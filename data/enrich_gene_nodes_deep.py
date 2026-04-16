"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``main``.

Prompt (user): data-dir graph hardening — visibility when adding PPI protein stubs.

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

import httpx
import networkx as nx
import numpy as np

from data.graph_identity import phase_log

async def enrich_gene_nodes_deep(self):
    """
    PHASE 2: Deep UniProt fetch per active PROTEIN — cofactors, Reactome, PPI, disease.

    Inputs: ``PROTEIN`` nodes in ``active_subgraph``; UniProt JSON via ``get_uniprot_url_single_gene``.
    Outputs: ``MINERAL``, ``MOLECULE_CHAIN``, ``DISEASE`` nodes and typed edges; optional ``PROTEIN``
    stubs for off-graph interaction partners (``stub: True``).
    Side effects: parallel HTTP via ``fetch_with_retry``.
    Failures: per-protein exceptions skipped in gather; no global raise.
    """
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "PROTEIN" and self._is_active(k)]
    tasks = [self.fetch_with_retry(self.get_uniprot_url_single_gene(node_id))
             for node_id, _ in protein_nodes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    _ppi_stubs = 0

    for (node_id, _node), res in zip(protein_nodes, results):
        if isinstance(res, Exception):
            continue

        # COFACTORS → MINERAL nodes + REQUIRES_MINERAL edges
        for comment in res.get("comments", []):
            if comment.get("commentType") != "COFACTOR":
                continue
            for cofactor in comment.get("cofactors", []):
                name = cofactor.get("name")
                if not name:
                    continue
                m_id = f"MINERAL_{name}"
                self.g.add_node({"id": m_id, "type": "MINERAL", "label": name})
                self.g.add_edge(src=node_id, trgt=m_id, attrs={
                    "rel": "REQUIRES_MINERAL", "src_layer": "PROTEIN", "trgt_layer": "MINERAL",
                })

        # helper: properties -> dict
        def _props_to_dict(props_list):
            if not props_list:
                return {}
            return {p["key"]: p["value"] for p in props_list if "key" in p and "value" in p}

        # REACTOME XREFS → MOLECULE_CHAIN nodes + EDGES
        for ref in res.get("uniProtKBCrossReferences", []):
            print("ref", ref)

            ref_id = ref.get("id")
            db = ref.get("database")
            isoform = ref.get("isoformId")

            props = _props_to_dict(ref.get("properties", []))

            # unified label strategy (robust)
            label = (
                    props.get("PathwayName")
                    or props.get("GeneName")
                    or props.get("ProteinId")
                    or props.get("Description")
                    or ref_id
            )

            node_id_ref = f"{db}_{ref_id}"

            # build full attribute payload (NO DATA LOSS)
            node_attrs = {
                "id": node_id_ref,
                "type": "MOLECULE_CHAIN",
                "label": label,
                "source_db": db,
                "ref_id": ref_id,
                "isoform": isoform,
                **props,  # flatten ALL properties
            }

            # add node
            self.g.add_node(node_attrs)

            # add edge
            self.g.add_edge(
                src=node_id,
                trgt=node_id_ref,
                attrs={
                    "rel": "HAS_REFERENCE",
                    "src_layer": "PROTEIN",
                    "trgt_layer": "MOLECULE_CHAIN",
                    "source_db": db,
                },
            )

        # PROTEIN–PROTEIN INTERACTIONS (UniProt cc_interaction / IntAct)
        # Only wire edge when the partner accession is ALREADY a node in the graph,
        # keeping the PPI layer anchored to the active biological context.
        for comment in res.get("comments", []):
            if comment.get("commentType") != "INTERACTION":
                continue
            for iact in comment.get("interactions", []):
                # UniProt encodes both sides; the *other* side relative to node_id is the partner
                acc_one = iact.get("interactantOne", {}).get("uniProtKBAccession", "")
                acc_two = iact.get("interactantTwo", {}).get("uniProtKBAccession", "")
                partner = acc_two if acc_one == node_id else acc_one
                if not partner or partner == node_id:
                    continue
                # If partner not yet in graph, add a minimal PROTEIN stub so the edge is reachable
                if not self.g.G.has_node(partner):
                    gene_lbl = (iact.get("interactantTwo", {}).get("geneName")
                                if acc_one == node_id
                                else iact.get("interactantOne", {}).get("geneName")) or partner
                    self.g.add_node({"id": partner, "type": "PROTEIN", "label": gene_lbl, "stub": True})
                    _ppi_stubs += 1
                self.g.add_edge(src=node_id, trgt=partner, attrs={
                    "rel": "INTERACTS_WITH",
                    "experiments": iact.get("numberOfExperiments", 0),
                    "src_layer": "PROTEIN", "trgt_layer": "PROTEIN",
                })

        # DISEASE ANNOTATIONS (UniProt cc_disease / MIM)
        # Each entry carries curated disease associations → DISEASE node + directed edge
        for comment in res.get("comments", []):
            if comment.get("commentType") != "DISEASE":
                continue
            disease_info = comment.get("disease", {})
            # UniProt uses diseaseId (acronym) and diseaseAccession (MIM-style)
            dis_acc = disease_info.get("diseaseAccession", "")
            dis_name = disease_info.get("diseaseId") or dis_acc
            if not dis_name:
                continue
            d_node_id = f"DISEASE_UNIPROT_{dis_acc or dis_name}"
            if not self.g.G.has_node(d_node_id):
                self.g.add_node({
                    "id": d_node_id, "type": "DISEASE",
                    "label": disease_info.get("diseaseId", dis_name),
                    "description": disease_info.get("description", ""),
                    "mim_id": dis_acc,
                })
            self.g.add_edge(src=node_id, trgt=d_node_id, attrs={
                "rel": "ASSOCIATED_WITH_DISEASE",
                "note": comment.get("note", {}).get("texts", [{}])[0].get("value", ""),
                "src_layer": "PROTEIN", "trgt_layer": "DISEASE",
            })

    phase_log("phase2_gene_deep", "ppi_stub_summary", entity_type="PROTEIN", stub_nodes_added=_ppi_stubs)

