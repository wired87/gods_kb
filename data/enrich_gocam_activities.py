"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``finalize_biological_graph``.

Prompt (user): data-dir graph hardening — non-nested node ids, production logging, precise docs.

CHAR: runs in-process on the same ``UniprotKB`` instance (``self``); keep signatures aligned
with the class delegator in ``uniprot_kb.py``.
"""
from __future__ import annotations

import asyncio
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

from data.graph_identity import (
    canonical_node_id,
    go_term_node_id,
    phase_http_log,
    timed_ms,
)

async def enrich_gocam_activities(self):
    """
    PHASE 13c: GO-CAM / BioLink functional associations as ``GOCAM_ACTIVITY`` hub nodes.

    Inputs: active ``GENE`` nodes; each must have a neighboring ``PROTEIN`` (UniProt accession
    as node id) reachable via ``ENCODED_BY`` from protein to gene.
    Outputs: ``GOCAM_ACTIVITY`` nodes (opaque ids), ``GO_TERM`` nodes if missing, edges
    ``ENABLED_BY`` → GENE, ``INVOLVES_PROTEIN`` → PROTEIN, aspect-specific rel → GO_TERM.
    Side effects: GET ``{BIOLINK}/bioentity/gene/UniProtKB:{accession}/function``.
    Failures: non-200 responses skip the gene; exceptions logged via ``phase_http_log``.
    """
    gene_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                  if v.get("type") == "GENE" and self._is_active(k)]
    activity_count = 0
    edge_count = 0
    _seen: set[str] = set()

    for gene_id, gene_data in gene_nodes:
        accession = None
        for neighbor in self.g.G.neighbors(gene_id):
            if self.g.G.nodes.get(neighbor, {}).get("type") == "PROTEIN":
                accession = neighbor
                break
        if not accession:
            continue

        url = f"{self._BIOLINK_BASE}/bioentity/gene/UniProtKB:{accession}/function"
        t0 = timed_ms()
        try:
            res = await self.client.get(
                url,
                params={"rows": self._GOCAM_ROWS},
                headers={"Accept": "application/json"},
                timeout=25.0,
            )
            if res.status_code != 200:
                phase_http_log(
                    "phase13c_gocam", "biolink_function",
                    url, status_code=res.status_code, elapsed_ms=timed_ms() - t0,
                )
                continue

            associations = res.json().get("associations", [])
            for assoc in associations:
                obj = assoc.get("object", {})
                go_id = obj.get("id")
                go_label = obj.get("label", "")
                go_category = obj.get("category", [])
                if not go_id:
                    continue

                rel_label = (assoc.get("relation") or {}).get("label", "associated_with")

                act_id = canonical_node_id(
                    "GOCAMACT",
                    {
                        "uniprot_accession": accession,
                        "go_id": go_id,
                        "relation": rel_label,
                    },
                )
                if act_id in _seen:
                    continue
                _seen.add(act_id)

                evidence_types = [
                    ev.get("label") for ev in assoc.get("evidence_types", [])
                    if ev.get("label")
                ]
                provided_by = [
                    s for s in assoc.get("provided_by", [])
                    if isinstance(s, str)
                ]

                go_node_id = go_term_node_id(go_id)
                if not self.g.G.has_node(go_node_id):
                    self.g.add_node({
                        "id": go_node_id,
                        "type": "GO_TERM",
                        "label": go_label,
                        "go_id": go_id,
                        "aspect": self._infer_go_aspect(go_category),
                    })

                self.g.add_node({
                    "id": act_id,
                    "type": "GOCAM_ACTIVITY",
                    "label": f"{gene_data.get('label', '')} {rel_label} {go_label}",
                    "uniprot_accession": accession,
                    "go_id": go_id,
                    "activity_relation": rel_label,
                    "evidence_types": evidence_types,
                    "provided_by": provided_by,
                })
                activity_count += 1

                self.g.add_edge(
                    src=act_id, trgt=gene_id,
                    attrs={"rel": "ENABLED_BY", "src_layer": "GOCAM", "trgt_layer": "GENE"},
                )
                edge_count += 1

                if self.g.G.has_node(accession):
                    self.g.add_edge(
                        src=act_id, trgt=accession,
                        attrs={
                            "rel": "INVOLVES_PROTEIN",
                            "src_layer": "GOCAM",
                            "trgt_layer": "PROTEIN",
                        },
                    )
                    edge_count += 1

                go_rel = self._gocam_edge_rel(rel_label, go_category)
                self.g.add_edge(
                    src=act_id, trgt=go_node_id,
                    attrs={"rel": go_rel, "src_layer": "GOCAM", "trgt_layer": "FUNCTIONAL"},
                )
                edge_count += 1

            phase_http_log(
                "phase13c_gocam", "biolink_function_ok",
                url, status_code=res.status_code, elapsed_ms=timed_ms() - t0,
            )
        except Exception as e:
            phase_http_log(
                "phase13c_gocam", "biolink_function_err",
                url, status_code=None, elapsed_ms=timed_ms() - t0,
                err_class=type(e).__name__,
            )

    print(f"Phase 13c: {activity_count} GOCAM_ACTIVITY nodes, {edge_count} edges created")
