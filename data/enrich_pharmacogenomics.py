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

async def enrich_pharmacogenomics(self):
    """
    Verknüpft PHARMA_COMPOUND-Nodes mit genetischen Varianten via ClinPGx.
    Pfad: PHARMA_COMPOUND -> CLINICAL_ANNOTATION -> GENETIC_VARIANT -> GENE
    Ermöglicht personalisierte Toxizitäts- und Metabolisierungswarnungen.
    """
    _CLINPGX_BASE = "https://api.clinpgx.org/v1"
    drug_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                  if v.get("type") == "PHARMA_COMPOUND" and self._is_active(k)]

    for node_id, drug in drug_nodes:
        drug_label = drug.get("label")
        if not drug_label:
            continue

        try:
            # A: ClinPGx Chemical-ID via Wirkstoffname
            chem_url = f"{_CLINPGX_BASE}/data/chemical?name={quote(drug_label)}"
            chem_res = await self.client.get(chem_url)
            if chem_res.status_code != 200:
                continue
            chem_data = chem_res.json().get("data", [])
            if not chem_data:
                continue

            pgkb_id = chem_data[0].get("id")
            if not pgkb_id:
                continue

            # B: Klinische Annotationen für diesen Wirkstoff
            # ClinPGx Rate Limit: 2 req/s
            await asyncio.sleep(0.5)
            annot_url = f"{_CLINPGX_BASE}/report/connectedObjects/{pgkb_id}/ClinicalAnnotation"
            annot_res = await self.client.get(annot_url)
            if annot_res.status_code != 200:
                continue
            annotations = annot_res.json().get("data", [])

            for annot in annotations:
                annot_id = annot.get("id")
                if not annot_id:
                    continue

                # C: CLINICAL_ANNOTATION Node
                ca_node_id = f"CLIN_ANNOT_{annot_id}"
                self.g.add_node({
                    "id": ca_node_id,
                    "type": "CLINICAL_ANNOTATION",
                    "label": annot.get("name", annot_id),
                    "evidence_level": annot.get("evidenceLevel"),
                    "phenotype_category": annot.get("phenotypeCategory"),
                    "pgkb_id": annot_id,
                })

                self.g.add_edge(
                    src=node_id,
                    trgt=ca_node_id,
                    attrs={
                        "rel": "CLINICAL_SIGNIFICANCE",
                        "src_layer": "PHARMA",
                        "trgt_layer": "GENETICS",
                    },
                )

                # D: GENETIC_VARIANT Nodes + Rückverlinkung zu GENE
                for var in annot.get("relatedVariants", []):
                    var_symbol = var.get("name") or var.get("symbol")
                    if not var_symbol:
                        continue

                    var_node_id = f"VARIANT_{var_symbol}"
                    self.g.add_node({
                        "id": var_node_id,
                        "type": "GENETIC_VARIANT",
                        "label": var_symbol,
                        "location": var.get("location"),
                        "pgkb_id": var.get("id"),
                    })

                    self.g.add_edge(
                        src=ca_node_id,
                        trgt=var_node_id,
                        attrs={"rel": "ASSOCIATED_VARIANT"},
                    )

                # E: Verlinkung Variante -> bestehendes GENE via relatedGenes
                for gene_ref in annot.get("relatedGenes", []):
                    gene_symbol = gene_ref.get("symbol") or gene_ref.get("name")
                    if not gene_symbol:
                        continue
                    target_gene_id = f"GENE_{gene_symbol}"
                    # Nur verlinken wenn der GENE-Node existiert
                    if self.g.G.has_node(target_gene_id):
                        for var in annot.get("relatedVariants", []):
                            vs = var.get("name") or var.get("symbol")
                            if vs:
                                self.g.add_edge(
                                    src=f"VARIANT_{vs}",
                                    trgt=target_gene_id,
                                    attrs={
                                        "rel": "VARIANT_OF",
                                        "src_layer": "GENETICS",
                                        "trgt_layer": "GENE",
                                    },
                                )

            print(f"PGx Enriched: {drug_label} ({len(annotations)} annotations)")

        except Exception as e:
            print(f"PGx Error for {drug_label}: {e}")

# --- SÄULE 2: BIOELEKTRISCHE KNOTENEIGENSCHAFTEN (GtoPdb) ──────
