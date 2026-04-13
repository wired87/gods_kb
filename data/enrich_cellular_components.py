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
import data.main as _uk

async def enrich_cellular_components(self):
    """
    Two-pass cellular-component integration:
      A) PROTEIN -> CELLULAR_COMPONENT via UniProt cc_subcellular_location
         + GENE (coding) -> CELLULAR_COMPONENT back-link
      B) Ensembl overlap -> NON_CODING_GENE -> CELLULAR_COMPONENT + GENE
    """
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "PROTEIN" and self._is_active(k)]
    comp_count = 0
    coding_links = 0

    # ── PASS A: UniProt subcellular locations ─────────────────────
    for node_id, protein in protein_nodes:
        accession = protein.get("id")
        if not accession:
            continue

        url = (
            f"https://rest.uniprot.org/uniprotkb/{accession}"
            f"?fields=cc_subcellular_location&format=json"
        )
        try:
            res = await self.fetch_with_retry(url)
            comments = res.get("comments", [])

            # COLLECT: components this protein resides in (for Pass B back-ref)
            protein_comp_ids: list[str] = []

            for comment in comments:
                if comment.get("commentType") != "SUBCELLULAR LOCATION":
                    continue

                for sl in comment.get("subcellularLocations", []):
                    loc = sl.get("location", {})
                    sl_id = loc.get("id")
                    sl_label = loc.get("value")
                    if not sl_id or not sl_label:
                        continue

                    comp_node_id = f"CELLCOMP_{sl_id}"

                    # IDEMPOTENT: create component node once
                    if not self.g.G.has_node(comp_node_id):
                        topo = sl.get("topology", {})
                        self.g.add_node({
                            "id": comp_node_id,
                            "type": "CELLULAR_COMPONENT",
                            "label": sl_label,
                            "sl_id": sl_id,
                            "topology": topo.get("value"),
                        })
                        comp_count += 1

                    # ECO: best evidence score from this location's evidences
                    best_rel = 0.0
                    best_eco = None
                    for ev in loc.get("evidences", []):
                        eco_code = ev.get("evidenceCode")
                        rel, _ = _uk.ECO_RELIABILITY.get(eco_code, _uk.ECO_DEFAULT)
                        if rel > best_rel:
                            best_rel, best_eco = rel, eco_code

                    self.g.add_edge(
                        src=node_id, trgt=comp_node_id,
                        attrs={
                            "rel": "RESIDES_IN",
                            "eco_code": best_eco,
                            "reliability": best_rel,
                            "src_layer": "PROTEIN",
                            "trgt_layer": "CELLULAR_COMPONENT",
                        },
                    )
                    protein_comp_ids.append(comp_node_id)

            # BACK-LINK: coding GENE -> CELLULAR_COMPONENT
            for neighbor in self.g.G.neighbors(node_id):
                n_data = self.g.G.nodes.get(neighbor, {})
                if n_data.get("type") != "GENE":
                    continue
                for cid in protein_comp_ids:
                    self.g.add_edge(
                        src=neighbor, trgt=cid,
                        attrs={
                            "rel": "CODING_GENE_IN_COMPONENT",
                            "src_layer": "GENE",
                            "trgt_layer": "CELLULAR_COMPONENT",
                        },
                    )
                    coding_links += 1

        except Exception as e:
            print(f"CellComp Error for {accession}: {e}")

    print(f"Pass A done: {comp_count} CELLULAR_COMPONENT nodes, "
          f"{coding_links} coding-gene links")

    # ── PASS B: Non-coding genes via Ensembl overlap ──────────────
    gene_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                  if v.get("type") == "GENE"
                  and self._is_active(k)
                  and v.get("chromosome")
                  and v.get("gene_start") is not None
                  and v.get("gene_end") is not None]

    _seen_nc: set[str] = set()
    nc_count = 0
    nc_comp_links = 0

    for gene_id, gene in gene_nodes:
        if nc_count >= _uk._MAX_NC_GENE_NODES:
            print(f"NON_CODING_GENE cap reached ({_uk._MAX_NC_GENE_NODES})")
            break

        chrom = gene["chromosome"]
        start = max(1, gene["gene_start"] - _uk._OVERLAP_FLANK_BP)
        end = gene["gene_end"] + _uk._OVERLAP_FLANK_BP
        region = f"{chrom}:{start}-{end}"

        # BIOTYPE FILTER: only non-coding classes
        biotype_params = ";".join(f"biotype={bt}" for bt in _uk._NC_BIOTYPES)
        ensembl_url = (
            f"https://rest.ensembl.org/overlap/region/homo_sapiens/{region}"
            f"?feature=gene;{biotype_params};content-type=application/json"
        )

        try:
            overlap_res = await self.client.get(ensembl_url, timeout=20.0)
            if overlap_res.status_code != 200:
                continue
            nc_genes = overlap_res.json()
            if not isinstance(nc_genes, list):
                continue

            for nc in nc_genes:
                nc_ensembl = nc.get("id") or nc.get("gene_id")
                if not nc_ensembl or nc_ensembl in _seen_nc:
                    continue
                if nc_count >= _uk._MAX_NC_GENE_NODES:
                    break

                nc_node_id = f"NCGENE_{nc_ensembl}"
                nc_label = (nc.get("external_name")
                            or nc.get("Name")
                            or nc_ensembl)
                nc_biotype = nc.get("biotype", "unknown")

                self.g.add_node({
                    "id": nc_node_id,
                    "type": "NON_CODING_GENE",
                    "label": nc_label,
                    "biotype": nc_biotype,
                    "ensembl_id": nc_ensembl,
                    "chromosome": chrom,
                    "start": nc.get("start"),
                    "end": nc.get("end"),
                })
                _seen_nc.add(nc_ensembl)
                nc_count += 1

                # EDGE: NON_CODING_GENE -> nearby coding GENE
                self.g.add_edge(
                    src=nc_node_id, trgt=gene_id,
                    attrs={
                        "rel": "OVERLAPS_CODING_GENE",
                        "src_layer": "NON_CODING",
                        "trgt_layer": "GENE",
                    },
                )

                # INFERRED LOCALIZATION: share coding gene's components
                for _, neighbor, edata in self.g.G.edges(gene_id, data=True):
                    if edata.get("rel") == "CODING_GENE_IN_COMPONENT":
                        self.g.add_edge(
                            src=nc_node_id, trgt=neighbor,
                            attrs={
                                "rel": "NC_GENE_IN_COMPONENT",
                                "inferred_from": gene_id,
                                "src_layer": "NON_CODING",
                                "trgt_layer": "CELLULAR_COMPONENT",
                            },
                        )
                        nc_comp_links += 1

        except Exception as e:
            print(f"Ensembl Overlap Error for {gene.get('label')}: {e}")

    print(f"Pass B done: {nc_count} NON_CODING_GENE nodes, "
          f"{nc_comp_links} component links (inferred)")

