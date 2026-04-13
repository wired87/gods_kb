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

async def detect_allergen_proteins(self):
    """
    Scannt PROTEIN-Nodes auf allergenes Potenzial.
    Primär: UniProt Keyword KW-0020 ("Allergen") via API.
    Fallback: Description-basierte Detektion für bereits geladene Proteine.
    """
    # A: UniProt-Suche nach annotierten humanen Allergenen (KW-0020)
    url = (
        "https://rest.uniprot.org/uniprotkb/search"
        "?query=keyword:KW-0020+AND+organism_id:9606"
        "&fields=accession,id,protein_name,gene_names,keyword"
        "&format=json&size=500"
    )
    allergen_accessions: set[str] = set()
    try:
        res = await self.client.get(url, timeout=30.0)
        if res.status_code == 200:
            for entry in res.json().get("results", []):
                acc = entry.get("primaryAccession")
                if acc:
                    allergen_accessions.add(acc)
    except Exception as e:
        print(f"UniProt Allergen KW-0020 Query Error: {e}")

    # B: Match gegen bestehende PROTEIN-Nodes im Graph
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "PROTEIN" and self._is_active(k)]
    detected = 0

    for node_id, protein in protein_nodes:
        accession = protein.get("id", "")
        desc = (protein.get("description") or "").lower()

        # DETECTION: UniProt-Annotation ODER Beschreibung enthält Allergen-Signale
        is_allergen = (
            accession in allergen_accessions
            or "allergen" in desc
            or "ige-binding" in desc
            or "ige binding" in desc
        )
        if not is_allergen:
            continue

        allergen_id = f"ALLERGEN_{accession}"
        self.g.add_node({
            "id": allergen_id,
            "type": "ALLERGEN",
            "label": protein.get("label", accession),
            "description": protein.get("description"),
            "detection_method": "UNIPROT_KW0020" if accession in allergen_accessions else "DESCRIPTION_SCAN",
        })
        self.g.add_edge(
            src=accession, trgt=allergen_id,
            attrs={
                "rel": "IS_ALLERGEN",
                "src_layer": "PROTEIN",
                "trgt_layer": "ALLERGEN",
            },
        )
        detected += 1

    print(f"Phase 14: {detected} allergen proteins detected "
          f"({len(allergen_accessions)} UniProt KW-0020 matches)")

# ═══════════════════════════════════════════════════════════════════
# PHASE 15: ALLERGEN MOLECULAR IMPACT (CTD + Open Targets)
# A) CTD: Welche Gene werden durch das Allergen in der Expression
#    verändert? -> Zytokin-Haushalt (IL-4, IL-5, IL-13 etc.)
# B) Open Targets GraphQL: Immunologische Krankheitsassoziationen
#    -> ALLERGEN -[TRIGGERS]-> IMMUNE_RESPONSE
# ═══════════════════════════════════════════════════════════════════

_OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_OT_DISEASE_QUERY = """
query($ensgId: String!) {
  target(ensemblId: $ensgId) {
    approvedSymbol
    associatedDiseases(page: {size: 50, index: 0}) {
      rows {
        disease { id name }
        score
      }
    }
  }
}
"""

