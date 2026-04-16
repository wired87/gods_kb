"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``main``.

CHAR: runs in-process on the same ``UniprotKB`` instance (``self``); keep signatures aligned
with the class delegator in ``uniprot_kb.py``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import google.generativeai as genai
import httpx
import networkx as nx
import numpy as np

async def _enrich_go_term_metadata(self):
    """
    Batch-fetch GO term definitions, synonyms, obsolescence, comments
    from QuickGO /ontology/go/terms/{ids} (up to 25 IDs per call).
    Patches existing GO_TERM nodes in-place.
    """
    _TERM_BASE = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"
    _BATCH_SIZE = 25
    go_nodes = [(n, d) for n, d in self.g.G.nodes(data=True) if d.get("type") == "GO_TERM"]
    enriched = 0

    for i in range(0, len(go_nodes), _BATCH_SIZE):
        chunk = go_nodes[i : i + _BATCH_SIZE]
        ids_csv = ",".join(d["go_id"] for _, d in chunk if d.get("go_id"))
        if not ids_csv:
            continue

        try:
            res = await self.client.get(
                f"{_TERM_BASE}/{quote(ids_csv, safe='')}",
                headers={"Accept": "application/json"},
                timeout=25.0,
            )
            if res.status_code != 200:
                continue

            for term in res.json().get("results", []):
                tid = term.get("id")
                if not tid:
                    continue
                node_id = f"GO_{tid.replace(':', '_')}"
                if not self.g.G.has_node(node_id):
                    continue

                self.g.G.nodes[node_id].update({
                    "definition": (term.get("definition") or {}).get("text"),
                    "synonyms": [s.get("name") for s in term.get("synonyms", []) if s.get("name")],
                    "is_obsolete": term.get("isObsolete", False),
                    "comment": term.get("comment"),
                })
                enriched += 1

        except Exception as e:
            print(f"GO Term Metadata Error: {e}")

    print(f"Phase 13a+: {enriched}/{len(go_nodes)} GO_TERM nodes enriched with metadata")

# ── GO ONTOLOGY HIERARCHY ────────────────────────────────────────
