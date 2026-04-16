"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``main``.

Prompt (user): No-op unless ``UniprotKB.workflow_create_cell_type_nodes`` is True (default off).

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
import data.main as _uk

async def enrich_cell_type_expression(self):
    """
    Zelluläre Integration: GENE -> CELL_TYPE via HPA Expression + OLS4 Metadaten.
    Two-Pass:
      1) HPA rtcte -> provisorische CELL_TYPE Nodes + EXPRESSED_IN_CELL Kanten
      2) OLS4 CL  -> description, CL-ID, parent + HAS_MARKER_GENE Rückverkettung
    """
    if not getattr(self, "workflow_create_cell_type_nodes", False):
        print(
            "enrich_cell_type_expression: skipped (workflow_create_cell_type_nodes=False)",
        )
        return

    _seen: set[str] = set()
    _count = 0

    gene_nodes = [(k, v) for k, v in self.g.G.nodes(data=True) if v.get("type") == "GENE" and self._is_active(k)]

    # ── PASS 1: HPA Expression Data ──────────────────────────────
    for gene_id, gene in gene_nodes:
        if _count >= _uk._MAX_CELL_NODES:
            print(f"CELL_TYPE cap reached ({_uk._MAX_CELL_NODES})")
            break

        gene_name = gene.get("label")
        if not gene_name:
            continue

        hpa_url = (
            f"https://www.proteinatlas.org/api/search_download.php"
            f"?search={quote(gene_name)}&format=json"
            f"&columns=g,eg,up,rtcte&compress=no"
        )
        try:
            res = await self.client.get(hpa_url, timeout=20.0)
            if res.status_code != 200:
                continue
            entries = res.json()
            if not isinstance(entries, list) or not entries:
                continue

            # HPA may return multiple genes; prefer exact name match
            matched = next(
                (e for e in entries if (e.get("Gene") or "").upper() == gene_name.upper()),
                entries[0],
            )
            rtcte = matched.get("RNA tissue cell type enrichment", "")
            cell_types = self._parse_hpa_cell_enrichment(rtcte)

            for ct_label in cell_types:
                ct_key = ct_label.lower().strip()
                cell_node_id = f"CELL_{ct_key.replace(' ', '_').upper()}"

                if ct_key not in _seen and _count < _uk._MAX_CELL_NODES:
                    self.g.add_node({
                        "id": cell_node_id,
                        "type": "CELL_TYPE",
                        "label": ct_label,
                        "ontology_prefix": "CL",
                        "cl_resolved": False,
                    })
                    _seen.add(ct_key)
                    _count += 1

                if self.g.G.has_node(cell_node_id):
                    self.g.add_edge(
                        src=gene_id, trgt=cell_node_id,
                        attrs={"rel": "EXPRESSED_IN_CELL", "src_layer": "GENE", "trgt_layer": "CELL"},
                    )

        except Exception as e:
            print(f"HPA Error for {gene_name}: {e}")

    print(f"Pass 1 done: {_count} CELL_TYPE nodes from HPA expression data")

    # ── PASS 2: OLS4 Cell Ontology Metadata ─────────────────────
    unresolved = [(k, v) for k, v in self.g.G.nodes(data=True)
                  if v.get("type") == "CELL_TYPE" and not v.get("cl_resolved")]

    resolved = 0
    for cell_id, cell in unresolved:
        cl_data = await self._resolve_cell_ontology(cell.get("label", ""))
        if not cl_data:
            continue

        cell["cl_id"] = cl_data["cl_id"]
        cell["label"] = cl_data["label"]
        cell["description"] = cl_data["description"]
        cell["cl_resolved"] = True
        resolved += 1

        # HAS_MARKER_GENE: CL annotation -> existing GENE nodes
        for marker in cl_data.get("marker_genes", []):
            target_gene = f"GENE_{marker}"
            if self.g.G.has_node(target_gene):
                self.g.add_edge(
                    src=cell_id, trgt=target_gene,
                    attrs={"rel": "HAS_MARKER_GENE", "src_layer": "CELL", "trgt_layer": "GENE"},
                )

    print(f"Pass 2 done: {resolved}/{_count} cells resolved via Cell Ontology (OLS4)")

