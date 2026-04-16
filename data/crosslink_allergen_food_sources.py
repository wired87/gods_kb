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

def crosslink_allergen_food_sources(self):
    """
    Cross-Link: FOOD_SOURCE -> ALLERGEN + CROSS_REACTIVITY zwischen Allergenen.
    Ermöglicht Vorhersage, welche Lebensmittel ähnliche Entzündungskaskaden auslösen.
    """
    # LOOKUP: source_protein_accession -> allergen_node_id (via IS_ALLERGEN edges)
    allergen_by_protein: dict[str, str] = {}
    for nid, ndata in self.g.G.nodes(data=True):
        if ndata.get("type") != "ALLERGEN" or not self._is_active(nid):
            continue
        for neighbor in self.g.G.neighbors(nid):
            if self.g.G.nodes.get(neighbor, {}).get("type") == "PROTEIN":
                allergen_by_protein[neighbor] = nid
                break

    if not allergen_by_protein:
        print("Phase 16: No allergen nodes for cross-linking")
        return

    # ── A: FOOD_SOURCE -> ALLERGEN via CONTAINS_NUTRIENT Kanten ──
    linked = 0
    for u, v, data in self.g.G.edges(data=True):
        if data.get("rel") != "CONTAINS_NUTRIENT":
            continue
        if self.g.G.nodes.get(u, {}).get("type") != "FOOD_SOURCE":
            continue

        allergen_id = allergen_by_protein.get(v)
        if not allergen_id:
            continue

        self.g.add_edge(
            src=u, trgt=allergen_id,
            attrs={
                "rel": "CONTAINS_ALLERGEN",
                "severity": "CRITICAL",
                "src_layer": "FOOD",
                "trgt_layer": "ALLERGEN",
            },
        )
        linked += 1
        food_label = self.g.G.nodes.get(u, {}).get("label", u)
        allergen_label = self.g.G.nodes.get(allergen_id, {}).get("label", allergen_id)
        print(f"Critical Path: {food_label} -> ALLERGEN {allergen_label}")

    # ── B: KREUZALLERGIE via gemeinsame Protein-Domänen ──────────
    # Wenn zwei Allergene dieselbe Domäne teilen -> Kreuzreaktivitätsrisiko
    allergen_domains: dict[str, set[str]] = {}
    for protein_acc, a_id in allergen_by_protein.items():
        domains: set[str] = set()
        if self.g.G.has_node(protein_acc):
            for _, neighbor, edata in self.g.G.edges(protein_acc, data=True):
                if edata.get("rel") == "CONTAINS_DOMAIN":
                    domains.add(neighbor)
        if domains:
            allergen_domains[a_id] = domains

    cross_links = 0
    allergen_list = list(allergen_domains.keys())
    for i, a_id in enumerate(allergen_list):
        for b_id in allergen_list[i + 1:]:
            shared = allergen_domains[a_id] & allergen_domains[b_id]
            if shared:
                self.g.add_edge(
                    src=a_id, trgt=b_id,
                    attrs={
                        "rel": "CROSS_REACTIVITY",
                        "shared_domains": list(shared),
                        "domain_overlap": len(shared),
                        "src_layer": "ALLERGEN",
                        "trgt_layer": "ALLERGEN",
                    },
                )
                cross_links += 1

    print(f"Phase 16: {linked} food-allergen links, {cross_links} cross-reactivity pairs")

# ═══════════════════════════════════════════════════════════════════
# PHASE 17: CELLULAR COMPONENTS + CODING / NON-CODING GENE MAPPING
# A) UniProt cc_subcellular_location -> CELLULAR_COMPONENT nodes
#    linked to PROTEIN and coding GENE nodes.
# B) Ensembl overlap -> NON_CODING_GENE nodes (lncRNA, miRNA, …)
#    linked to same CELLULAR_COMPONENT + nearby coding GENE.
# ═══════════════════════════════════════════════════════════════════

