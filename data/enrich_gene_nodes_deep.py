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

async def enrich_gene_nodes_deep(self):
    """Fetches cofactors + Reactome pathways per PROTEIN and materializes them as edges.
    No reference data stored on nodes — cofactors become MINERAL nodes, pathways become MOLECULE_CHAIN nodes."""
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "PROTEIN" and self._is_active(k)]
    tasks = [self.fetch_with_retry(self.get_uniprot_url_single_gene(node_id))
             for node_id, _ in protein_nodes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

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

        # REACTOME XREFS → MOLECULE_CHAIN nodes + INVOLVED_IN_CHAIN edges
        for ref in res.get("uniProtKBCrossReferences", []):
            if ref.get("database") != "Reactome":
                continue
            pw_id = f"MOL_{ref['id']}"
            self.g.add_node({"id": pw_id, "type": "MOLECULE_CHAIN",
                             "label": ref.get("properties", {}).get("pathwayName", "Pathway"),
                             # store the stable Reactome ID so _bridge_reactome_nodes can match
                             "reactome_stable_id": ref["id"]})
            self.g.add_edge(src=node_id, trgt=pw_id, attrs={
                "rel": "INVOLVED_IN_CHAIN", "src_layer": "PROTEIN", "trgt_layer": "MOLECULE_CHAIN",
            })

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

