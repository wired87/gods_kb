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

def build_tissue_hierarchy_map(self, cfg: dict | None = None) -> nx.Graph:
    """
    Prompt: return a seamless map from organ layer to atomic structure / electron matrix.

    Undirected BFS from ORGAN (or ANATOMY_PART fallback) roots matching cfg['organs'];
    keeps only nodes in _ORGAN_TISSUE_CASCADE_TYPES plus any node carrying
    electron_density_matrix / atom_decomposition (Phase 18 output).
    """
    G = self.g.G
    want = set()
    if cfg:
        want = {str(o).strip().lower() for o in (cfg.get("organs") or []) if str(o).strip()}

    roots: set[str] = set()
    for nid, d in G.nodes(data=True):
        if d.get("type") != "ORGAN":
            continue
        it = (d.get("input_term") or "").strip().lower()
        if want and it in want:
            roots.add(nid)
    if want:
        for nid, d in G.nodes(data=True):
            if d.get("type") != "ORGAN":
                continue
            if (d.get("label") or "").strip().lower() in want:
                roots.add(nid)
    if not roots:
        roots = {n for n, d in G.nodes(data=True) if d.get("type") == "ORGAN"}
    if not roots and want:
        for nid, d in G.nodes(data=True):
            if d.get("type") != "ANATOMY_PART":
                continue
            lab = (d.get("label") or "").strip().lower()
            if any(w == lab or w in lab or lab in w for w in want):
                roots.add(nid)

    if not roots:
        print("  Tissue hierarchy map: no organ roots — returning empty graph")
        return nx.Graph()

    visited: set[str] = set()
    q: deque[str] = deque()
    for r in roots:
        if r in G:
            visited.add(r)
            q.append(r)

    while q:
        u = q.popleft()
        for v in G.neighbors(u):
            if v in visited:
                continue
            vd = G.nodes[v]
            if self._node_in_organ_tissue_cascade(vd):
                visited.add(v)
                q.append(v)

    H = G.subgraph(visited).copy()
    print(f"  Tissue hierarchy map: {H.number_of_nodes()}N / {H.number_of_edges()}E "
          f"(from {len(roots)} organ/anatomy root(s))")
    return H

# --- CONSOLIDATED WORKFLOW ---
