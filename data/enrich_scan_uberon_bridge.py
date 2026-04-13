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

async def enrich_scan_uberon_bridge(self):
    """PHASE 22a: Connect SPATIAL_REGION nodes to existing TISSUE / ORGAN nodes
    via UBERON ID matching and Ubergraph part_of traversal."""
    _UBERGRAPH_SPARQL = "https://ubergraph.apps.renci.org/sparql"

    async def _ubergraph_parent_organ(uberon_id: str) -> dict | None:
        obo_uri = f"http://purl.obolibrary.org/obo/{uberon_id.replace(':', '_')}"
        query = (
            "SELECT DISTINCT ?organ ?organLabel WHERE { "
            f"<{obo_uri}> <http://purl.obolibrary.org/obo/BFO_0000050> ?organ . "
            "?organ <http://www.w3.org/2000/01/rdf-schema#label> ?organLabel . "
            "FILTER(STRSTARTS(STR(?organ), 'http://purl.obolibrary.org/obo/UBERON_')) "
            "} LIMIT 5"
        )
        try:
            res = await self.client.get(
                _UBERGRAPH_SPARQL,
                params={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=15.0,
            )
            if res.status_code != 200:
                return None
            bindings = res.json().get("results", {}).get("bindings", [])
            if not bindings:
                return None
            first = bindings[0]
            organ_uri = first.get("organ", {}).get("value", "")
            organ_label = first.get("organLabel", {}).get("value", "")
            organ_uberon = organ_uri.split("/")[-1].replace("_", ":")
            return {"uberon_id": organ_uberon, "label": organ_label}
        except Exception:
            return None

    # INDEX: existing TISSUE + ORGAN nodes by uberon_id for fast lookup
    tissue_by_uberon: dict[str, str] = {}
    organ_by_uberon: dict[str, str] = {}
    for nid, nd in self.g.G.nodes(data=True):
        ntype = nd.get("type")
        uid = nd.get("uberon_id")
        if not uid:
            continue
        if ntype == "TISSUE":
            tissue_by_uberon[uid] = nid
        elif ntype == "ORGAN":
            organ_by_uberon[uid] = nid

    spatial_nodes = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "SPATIAL_REGION" and v.get("uberon_id")
    ]

    _tissue_links = 0
    _organ_links = 0
    for reg_id, reg in spatial_nodes:
        uid = reg["uberon_id"]

        # DIRECT MATCH: SPATIAL_REGION → TISSUE
        if uid in tissue_by_uberon:
            self.g.add_edge(
                src=reg_id, trgt=tissue_by_uberon[uid],
                attrs={"rel": "MAPS_TO_TISSUE", "src_layer": "SPATIAL_REGION", "trgt_layer": "TISSUE"},
            )
            _tissue_links += 1

        # DIRECT MATCH: SPATIAL_REGION → ORGAN
        if uid in organ_by_uberon:
            self.g.add_edge(
                src=reg_id, trgt=organ_by_uberon[uid],
                attrs={"rel": "MAPS_TO_ORGAN", "src_layer": "SPATIAL_REGION", "trgt_layer": "ORGAN"},
            )
            _organ_links += 1
            continue

        # FALLBACK: Ubergraph part_of → parent ORGAN
        parent = await _ubergraph_parent_organ(uid)
        if parent and parent.get("uberon_id"):
            parent_uid = parent["uberon_id"]
            if parent_uid in organ_by_uberon:
                self.g.add_edge(
                    src=reg_id, trgt=organ_by_uberon[parent_uid],
                    attrs={"rel": "MAPS_TO_ORGAN", "src_layer": "SPATIAL_REGION", "trgt_layer": "ORGAN"},
                )
                _organ_links += 1

    print(f"  Phase 22a: {_tissue_links} tissue links, {_organ_links} organ links")

# ══════════════════════════════════════════════════════════════════
# PHASE 22b — MODALITY-SPECIFIC FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

