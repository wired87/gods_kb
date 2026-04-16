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

async def enrich_compartment_localization(self):
    """
    COMPARTMENTS DB (JensenLab): PROTEIN -[LOCALIZED_IN]-> COMPARTMENT.
    Subzelluläre Lokalisation mit Konfidenz-Score.
    """
    _COMP_BASE = "https://compartments.jensenlab.org/Entity"
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "PROTEIN" and self._is_active(k)]
    loc_count = 0
    _MIN_CONFIDENCE = 2.0

    for node_id, protein in protein_nodes:
        accession = protein.get("id")
        gene_label = protein.get("label", "")
        if not accession:
            continue

        try:
            res = await self.client.get(
                _COMP_BASE,
                params={"query": gene_label, "type": "9606", "format": "json"},
                timeout=15.0,
            )
            if res.status_code != 200:
                continue
            entries = res.json()
            if not isinstance(entries, list):
                continue

            for entry in entries:
                compartment = entry.get("compartment", {})
                comp_id_raw = compartment.get("id") or entry.get("go_id")
                comp_name = compartment.get("name") or entry.get("name")
                confidence = float(entry.get("confidence", 0))

                if not comp_id_raw or not comp_name:
                    continue
                if confidence < _MIN_CONFIDENCE:
                    continue

                comp_node_id = f"COMP_{comp_id_raw.replace(':', '_')}"
                if not self.g.G.has_node(comp_node_id):
                    self.g.add_node({
                        "id": comp_node_id,
                        "type": "COMPARTMENT",
                        "label": comp_name,
                        "go_id": comp_id_raw,
                    })

                    # CROSS-LINK: if go_id matches an existing GO_TERM node, wire them
                    if comp_id_raw and comp_id_raw.startswith("GO:"):
                        mapped_go_node = f"GO_{comp_id_raw.replace(':', '_')}"
                        if self.g.G.has_node(mapped_go_node):
                            self.g.add_edge(
                                src=comp_node_id, trgt=mapped_go_node,
                                attrs={
                                    "rel": "MAPPED_TO_GO",
                                    "src_layer": "LOCALIZATION",
                                    "trgt_layer": "FUNCTIONAL",
                                },
                            )

                self.g.add_edge(
                    src=node_id, trgt=comp_node_id,
                    attrs={
                        "rel": "LOCALIZED_IN",
                        "confidence": confidence,
                        "source": entry.get("source"),
                        "src_layer": "PROTEIN",
                        "trgt_layer": "LOCALIZATION",
                    },
                )
                loc_count += 1

        except Exception as e:
            print(f"COMPARTMENTS Error for {gene_label}: {e}")

    print(f"Phase 13b: {loc_count} localization links created (min confidence={_MIN_CONFIDENCE})")

# ═══════════════════════════════════════════════════════════════════
# PHASE 13c: GO-CAM CAUSAL ACTIVITY MODELS (BioLink API)
# Fetches gene-function associations and builds GOCAM_ACTIVITY nodes
# with causal/structural edges into the GO semantic layer.
# API: http://api.geneontology.org/api/bioentity/gene/UniProtKB:{acc}/function
# ═══════════════════════════════════════════════════════════════════

_BIOLINK_BASE = "http://api.geneontology.org/api"
_GOCAM_ROWS = 50

