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

def _bridge_reactome_nodes(self):
    """
    Cross-link MOLECULE_CHAIN nodes (from UniProt xref stable IDs, e.g. R-HSA-…)
    with REACTOME_PATHWAY nodes (numeric dbId from Reactome ContentService) that
    represent the SAME biological pathway.

    Both node types are created independently during Phase 2 and Phase 6.
    This bridge adds a SAME_AS edge so downstream graph traversal can treat them
    as co-referential without merging them (preserving provenance).
    """
    # INDEX: stable_id → MOLECULE_CHAIN node id
    mol_by_stable: dict[str, str] = {}
    for n, d in self.g.G.nodes(data=True):
        if d.get("type") == "MOLECULE_CHAIN" and d.get("reactome_stable_id"):
            mol_by_stable[d["reactome_stable_id"]] = n

    bridged = 0
    for pw_node_id, pw_data in self.g.G.nodes(data=True):
        if pw_data.get("type") != "REACTOME_PATHWAY":
            continue
        stable_id = pw_data.get("reactome_stable_id")
        if not stable_id or stable_id not in mol_by_stable:
            continue
        mol_node_id = mol_by_stable[stable_id]
        # Avoid duplicate edges (multigraph may already have one from a prior run)
        existing = {d.get("rel") for _, _, d in self.g.G.edges(mol_node_id, data=True)}
        if "SAME_AS" not in existing:
            self.g.add_edge(src=mol_node_id, trgt=pw_node_id, attrs={
                "rel": "SAME_AS",
                "src_layer": "MOLECULE_CHAIN", "trgt_layer": "REACTOME_PATHWAY",
            })
        bridged += 1

    print(f"Phase 6+: {bridged} Reactome MOL↔PATHWAY bridge edges created")

