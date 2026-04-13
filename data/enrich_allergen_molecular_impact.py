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

async def enrich_allergen_molecular_impact(self):
    """
    Molecular Mapping: CTD Chemical-Gene Interactions + Open Targets Disease Links.
    Erzeugt ALTERS_EXPRESSION und TRIGGERS Kanten für den Allergie-Subgraph.
    """
    allergen_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                      if v.get("type") == "ALLERGEN" and self._is_active(k)]

    if not allergen_nodes:
        print("Phase 15: No allergen nodes found, skipping")
        return

    ctd_links = 0
    immune_links = 0

    for allergen_id, allergen in allergen_nodes:
        allergen_label = allergen.get("label", "")

        # RESOLVE source protein via IS_ALLERGEN edge (PROTEIN -> ALLERGEN)
        source_protein = ""
        for neighbor in self.g.G.neighbors(allergen_id):
            if self.g.G.nodes.get(neighbor, {}).get("type") == "PROTEIN":
                source_protein = neighbor
                break

        # ── A: CTD Chemical-Gene Interactions ──────────────────────
        # ZIEL: welche Gene werden durch dieses Allergen hochreguliert?
        ctd_url = (
            f"http://ctdbase.org/tools/batchQuery.go"
            f"?inputType=chem&inputTerms={quote(allergen_label)}"
            f"&report=genes_curated&format=json"
        )
        try:
            ctd_res = await self.client.get(ctd_url, timeout=30.0)
            if ctd_res.status_code == 200:
                content_type = ctd_res.headers.get("content-type", "")
                if "json" in content_type:
                    raw = ctd_res.json()
                    interactions = raw if isinstance(raw, list) else []

                    for inter in interactions:
                        target_gene = inter.get("GeneSymbol")
                        action = inter.get("InteractionActions", "")
                        organism = inter.get("Organism", "")

                        if "Homo sapiens" not in organism:
                            continue

                        gene_node_id = f"GENE_{target_gene}"
                        if not self.g.G.has_node(gene_node_id):
                            continue

                        impact = "INFLAMMATORY_CASCADE" if "increases" in action.lower() else "REGULATORY"

                        self.g.add_edge(
                            src=allergen_id,
                            trgt=gene_node_id,
                            attrs={
                                "rel": "ALTERS_EXPRESSION",
                                "mechanism": action,
                                "impact": impact,
                                "src_layer": "ALLERGEN",
                                "trgt_layer": "GENE",
                            },
                        )
                        ctd_links += 1
        except Exception as e:
            print(f"CTD Error for {allergen_label}: {e}")

        # ── B: Open Targets Disease Associations (GraphQL) ────────
        # Benötigt Ensembl-ID via verlinktem GENE-Node
        ensembl_id = None
        if source_protein and self.g.G.has_node(source_protein):
            for neighbor in self.g.G.neighbors(source_protein):
                n_data = self.g.G.nodes.get(neighbor, {})
                if n_data.get("type") == "GENE" and n_data.get("ensembl_id"):
                    ensembl_id = n_data["ensembl_id"]
                    break

        if not ensembl_id:
            continue

        try:
            ot_res = await self.client.post(
                self._OT_URL,
                json={"query": self._OT_DISEASE_QUERY, "variables": {"ensgId": ensembl_id}},
                timeout=20.0,
            )
            if ot_res.status_code != 200:
                continue

            target_data = ot_res.json().get("data", {}).get("target", {})
            rows = (target_data.get("associatedDiseases") or {}).get("rows", [])

            for row in rows:
                disease = row.get("disease", {})
                disease_name = disease.get("name", "")
                disease_id = disease.get("id", "")
                score = row.get("score", 0)

                # FILTER: immunologisch relevante Assoziationen mit Score > 0.3
                disease_lower = disease_name.lower()
                is_immune = any(t in disease_lower for t in (
                    "allerg", "hypersensitiv", "asthma", "dermatitis",
                    "rhinitis", "anaphyla", "urticaria", "eczema",
                    "atopic", "immun", "inflammat", "histamin", "mast cell",
                ))
                if not is_immune or score < 0.3:
                    continue

                response_id = f"IMMUNE_{disease_id.replace(':', '_')}"
                if not self.g.G.has_node(response_id):
                    self.g.add_node({
                        "id": response_id,
                        "type": "IMMUNE_RESPONSE",
                        "label": disease_name,
                        "disease_id": disease_id,
                        "association_score": score,
                    })

                self.g.add_edge(
                    src=allergen_id, trgt=response_id,
                    attrs={
                        "rel": "TRIGGERS",
                        "score": score,
                        "src_layer": "ALLERGEN",
                        "trgt_layer": "IMMUNE",
                    },
                )
                immune_links += 1

        except Exception as e:
            print(f"Open Targets Error for {allergen_label}: {e}")

    print(f"Phase 15: {ctd_links} gene expression links (CTD), "
          f"{immune_links} immune response links (Open Targets)")

# ═══════════════════════════════════════════════════════════════════
# PHASE 16: ALLERGEN-FOOD CROSS-LINKING + KREUZALLERGIE
# Verbindet FOOD_SOURCE -> ALLERGEN wenn ein Lebensmittel ein Protein
# enthält, das als Allergen markiert ist. Domänen-Überlappung zwischen
# Allergenen erzeugt CROSS_REACTIVITY Kanten (Kreuzallergie-Prädiktion).
# ═══════════════════════════════════════════════════════════════════

