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

async def enrich_structural_layer(self):
    """
    AlphaFold DB Integration: PROTEIN -[HAS_PREDICTED_STRUCTURE]-> 3D_STRUCTURE.
    pLDDT-Score als Zuverlässigkeitsmetrik an der Kante.
    """
    _AF_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "PROTEIN" and self._is_active(k)]
    linked = 0

    for node_id, protein in protein_nodes:
        accession = protein.get("id")
        if not accession:
            continue

        try:
            res = await self.client.get(f"{_AF_BASE}/{accession}", timeout=20.0)
            if res.status_code != 200:
                continue
            entries = res.json()
            if not entries:
                continue
            data = entries[0] if isinstance(entries, list) else entries

            model_id = f"STRUCT_{accession}"
            plddt = data.get("globalMetrics", {}).get("globalPlddt") or data.get("uniprotScore")

            self.g.add_node({
                "id": model_id,
                "type": "3D_STRUCTURE",
                "label": f"AlphaFold_{accession}",
                "pLDDT_avg": plddt,
                "pdb_url": data.get("pdbUrl"),
                "cif_url": data.get("cifUrl"),
                "model_version": data.get("latestVersion"),
                "gene": data.get("gene"),
            })

            self.g.add_edge(
                src=node_id, trgt=model_id,
                attrs={
                    "rel": "HAS_PREDICTED_STRUCTURE",
                    "pLDDT": plddt,
                    "src_layer": "PROTEIN",
                    "trgt_layer": "STRUCTURAL",
                },
            )
            linked += 1

        except Exception as e:
            print(f"AlphaFold Error for {accession}: {e}")

    print(f"Phase 11: {linked} proteins linked to AlphaFold structures")

# ═══════════════════════════════════════════════════════════════════
# PHASE 12: DOMAIN DECOMPOSITION (InterPro)
# Bricht Proteine in funktionale Domänen auf.
# Ermöglicht funktionale Ähnlichkeitsschlüsse zwischen neuartigen
# und bekannten Proteinen über gemeinsame Domänen.
# ═══════════════════════════════════════════════════════════════════

