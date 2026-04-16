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

async def enrich_domain_decomposition(self):
    """
    InterPro Integration: PROTEIN -[CONTAINS_DOMAIN]-> DOMAIN.
    Domänen aus Pfam, SMART, CDD etc. werden als eigenständige Nodes modelliert.
    """
    _INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot"
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "PROTEIN" and self._is_active(k)]
    domain_count = 0

    for node_id, protein in protein_nodes:
        accession = protein.get("id")
        if not accession:
            continue

        try:
            res = await self.client.get(
                f"{_INTERPRO_BASE}/{accession}",
                headers={"Accept": "application/json"},
                timeout=20.0,
            )
            if res.status_code != 200:
                continue
            payload = res.json()
            results = payload.get("results", [])

            for entry in results:
                meta = entry.get("metadata", {})
                ipr_id = meta.get("accession")
                if not ipr_id:
                    continue

                domain_node_id = f"DOMAIN_{ipr_id}"
                # IDEMPOTENT: Domain-Node nur einmal anlegen, mehrfach verlinken
                if not self.g.G.has_node(domain_node_id):
                    self.g.add_node({
                        "id": domain_node_id,
                        "type": "PROTEIN_DOMAIN",
                        "label": meta.get("name", ipr_id),
                        "interpro_id": ipr_id,
                        "domain_type": meta.get("type"),
                        "source_database": meta.get("source_database"),
                    })

                    # EXPAND: InterPro GO terms -> proper GO_TERM nodes + edges
                    for go_raw in meta.get("go_terms", []):
                        go_ident = go_raw.get("identifier")
                        if not go_ident:
                            continue
                        go_node_id = f"GO_{go_ident.replace(':', '_')}"
                        if not self.g.G.has_node(go_node_id):
                            self.g.add_node({
                                "id": go_node_id,
                                "type": "GO_TERM",
                                "label": go_raw.get("name", go_ident),
                                "go_id": go_ident,
                                "aspect": go_raw.get("category", {}).get("name"),
                            })
                        self.g.add_edge(
                            src=domain_node_id, trgt=go_node_id,
                            attrs={
                                "rel": "ASSOCIATED_GO",
                                "src_layer": "DOMAIN",
                                "trgt_layer": "FUNCTIONAL",
                            },
                        )

                # POSITIONALE INFORMATION: wo sitzt die Domäne in der Sequenz?
                proteins_block = entry.get("proteins", [])
                locations = []
                for prot_entry in proteins_block:
                    for loc_group in prot_entry.get("entry_protein_locations", []):
                        for frag in loc_group.get("fragments", []):
                            locations.append({
                                "start": frag.get("start"),
                                "end": frag.get("end"),
                            })

                self.g.add_edge(
                    src=node_id, trgt=domain_node_id,
                    attrs={
                        "rel": "CONTAINS_DOMAIN",
                        "positions": locations if locations else None,
                        "src_layer": "PROTEIN",
                        "trgt_layer": "DOMAIN",
                    },
                )
                domain_count += 1

        except Exception as e:
            print(f"InterPro Error for {accession}: {e}")

    print(f"Phase 12: {domain_count} domain links created")
