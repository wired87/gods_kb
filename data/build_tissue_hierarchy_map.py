"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``finalize_biological_graph``.

Prompt (user): data-dir graph hardening — one-pass organ root discovery.

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
    Build an organ-centric subgraph for tissue cascade + Phase-18 quantum outputs.

    Inputs: ``self.g.G``; optional ``cfg`` with ``organs`` list (string names, case-insensitive).
    Outputs: undirected subgraph spanning BFS from matching ORGAN (or ANATOMY_PART fallback) roots
    through nodes accepted by ``_node_in_organ_tissue_cascade`` plus electron-density carriers.
    Side effects: none on the live graph; returns a new ``nx.Graph`` copy of the subgraph.
    Empty result: returns ``nx.Graph()`` when no roots match.
    """
    G = self.g.G
    want = set()
    if cfg:
        want = {str(o).strip().lower() for o in (cfg.get("organs") or []) if str(o).strip()}

    roots: set[str] = set()
    all_organs: set[str] = set()
    anatomy_fallback: set[str] = set()

    for nid, d in G.nodes(data=True):
        t = d.get("type")
        if t == "ORGAN":
            all_organs.add(nid)
            if want:
                it = (d.get("input_term") or "").strip().lower()
                lab = (d.get("label") or "").strip().lower()
                if it in want or lab in want:
                    roots.add(nid)
        elif t == "ANATOMY_PART" and want:
            lab = (d.get("label") or "").strip().lower()
            if any(w == lab or w in lab or lab in w for w in want):
                anatomy_fallback.add(nid)

    if not roots:
        roots = set(all_organs)
    if not roots and want:
        roots = set(anatomy_fallback)

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
