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

async def enrich_go_semantic_layer(self):
    """
    Gene Ontology via QuickGO: PROTEIN -[ANNOTATED_WITH]-> GO_TERM.
    Evidence codes resolved through short GO codes AND ECO URIs.
    Only annotations with reliability >= 0.5 are kept.
    """
    _QUICKGO_BASE = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
    _MIN_RELIABILITY = 0.5
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "PROTEIN" and self._is_active(k)]
    go_count = 0

    for node_id, protein in protein_nodes:
        accession = protein.get("id")
        if not accession:
            continue

        try:
            res = await self.client.get(
                _QUICKGO_BASE,
                params={"geneProductId": accession, "limit": 100, "taxonId": 9606},
                headers={"Accept": "application/json"},
                timeout=20.0,
            )
            if res.status_code != 200:
                continue
            results = res.json().get("results", [])

            for anno in results:
                go_id = anno.get("goId")
                if not go_id:
                    continue

                # RESOLVE: short GO evidence codes (IDA, IPI, …) + full ECO URIs
                raw_code = anno.get("goEvidence", "")
                reliability, evidence_type, eco_uri = self._resolve_go_evidence(raw_code)
                if reliability < _MIN_RELIABILITY:
                    continue

                go_node_id = f"GO_{go_id.replace(':', '_')}"
                if not self.g.G.has_node(go_node_id):
                    self.g.add_node({
                        "id": go_node_id,
                        "type": "GO_TERM",
                        "label": anno.get("goName", go_id),
                        "go_id": go_id,
                        "aspect": anno.get("goAspect"),
                    })

                self.g.add_edge(
                    src=node_id, trgt=go_node_id,
                    attrs={
                        "rel": "ANNOTATED_WITH",
                        "evidence_code": eco_uri,
                        "reliability": reliability,
                        "evidence_type": evidence_type,
                        "qualifier": anno.get("qualifier"),
                        "assigned_by": anno.get("assignedBy"),
                        "extension": anno.get("extensions"),
                        "src_layer": "PROTEIN",
                        "trgt_layer": "FUNCTIONAL",
                    },
                )
                go_count += 1

        except Exception as e:
            print(f"QuickGO Error for {accession}: {e}")

    print(f"Phase 13a: {go_count} GO annotations linked (min reliability={_MIN_RELIABILITY})")

# ── GO TERM METADATA ENRICHMENT ──────────────────────────────────
