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

def _wire_gene_go_edges(self):
    """
    Derive GENE -> GO_TERM edges by traversing
    PROTEIN -[ANNOTATED_WITH]-> GO_TERM and PROTEIN -[ENCODED_BY]-> GENE.
    Uses the strongest reliability per (gene, go_term) pair.
    """
    # COLLECT: protein -> gene mapping
    protein_to_genes: dict[str, list[str]] = {}
    for src, trgt, edata in self.g.G.edges(data=True):
        if edata.get("rel") == "ENCODED_BY":
            protein_to_genes.setdefault(src, []).append(trgt)

    # COLLECT: protein -> go_term annotations with best reliability
    # KEY: (gene_node_id, go_node_id) -> best reliability
    gene_go_best: dict[tuple[str, str], tuple[float, str, str]] = {}

    for src, trgt, edata in self.g.G.edges(data=True):
        if edata.get("rel") != "ANNOTATED_WITH":
            continue
        protein_id = src
        go_node_id = trgt
        reliability = edata.get("reliability", 0)
        evidence_type = edata.get("evidence_type", "")
        go_attrs = self.g.G.nodes.get(go_node_id, {})
        aspect = go_attrs.get("aspect", "")

        for gene_id in protein_to_genes.get(protein_id, []):
            key = (gene_id, go_node_id)
            if key not in gene_go_best or reliability > gene_go_best[key][0]:
                gene_go_best[key] = (reliability, evidence_type, protein_id)

    derived_count = 0
    for (gene_id, go_node_id), (reliability, evidence_type, protein_id) in gene_go_best.items():
        go_attrs = self.g.G.nodes.get(go_node_id, {})
        aspect = go_attrs.get("aspect", "")
        rel = self._ASPECT_TO_REL.get(aspect, "ANNOTATED_WITH_GENE")

        self.g.add_edge(
            src=gene_id, trgt=go_node_id,
            attrs={
                "rel": rel,
                "derived_from": protein_id,
                "reliability": reliability,
                "evidence_type": evidence_type,
                "src_layer": "GENE",
                "trgt_layer": "FUNCTIONAL",
            },
        )
        derived_count += 1

    print(f"Phase 13a+++: {derived_count} GENE -> GO_TERM derived edges created")

