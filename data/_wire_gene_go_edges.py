"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``finalize_biological_graph``.

Prompt (user): data-dir graph hardening — single-pass edge scan for gene–GO derivation.

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
    Derive GENE → GO_TERM edges from PROTEIN → GO_TERM (ANNOTATED_WITH) and
    PROTEIN → GENE (ENCODED_BY), keeping the strongest reliability per (gene, go_term).

    Inputs: existing ``ENCODED_BY`` and ``ANNOTATED_WITH`` edges; GO_TERM nodes with ``aspect``.
    Outputs: new edges GENE → GO_TERM with ``derived_from`` protein id and evidence fields.
    Side effects: mutates ``self.g.G`` only; single full edge pass for indexing.
    """
    protein_to_genes: dict[str, list[str]] = {}
    annotated: list[tuple[str, str, dict]] = []

    for src, trgt, edata in self.g.G.edges(data=True):
        rel = edata.get("rel")
        if rel == "ENCODED_BY":
            protein_to_genes.setdefault(src, []).append(trgt)
        elif rel == "ANNOTATED_WITH":
            annotated.append((src, trgt, edata))

    gene_go_best: dict[tuple[str, str], tuple[float, str, str]] = {}

    for protein_id, go_node_id, edata in annotated:
        reliability = edata.get("reliability", 0)
        evidence_type = edata.get("evidence_type", "")
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
