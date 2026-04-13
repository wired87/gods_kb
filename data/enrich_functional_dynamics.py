"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``finalize_biological_graph``.

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

async def enrich_functional_dynamics(self):
    """Reactome Pfade und Kausalität."""
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True) if v.get("type") == "PROTEIN" and self._is_active(k)]
    for accession, node in protein_nodes:
        url = f"https://reactome.org/ContentService/data/pathways/low/entity/{accession}"
        try:
            res = await self.fetch_with_retry(url)
            for pw in res:
                pw_id = f"PATHWAY_{pw['dbId']}"
                pw_node = {"id": pw_id, "type": "REACTOME_PATHWAY", "label": pw["displayName"]}
                # stId is the stable Reactome identifier (R-HSA-…) — used by _bridge_reactome_nodes
                if pw.get("stId"):
                    pw_node["reactome_stable_id"] = pw["stId"]
                self.g.add_node(pw_node)
                self.g.add_edge(src=accession, trgt=pw_id,
                                attrs={"rel": "PARTICIPATES_IN", "causality": "CONTRIBUTORY"})
        except Exception:
            continue

