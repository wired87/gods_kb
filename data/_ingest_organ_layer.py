"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``finalize_biological_graph``.

Prompt: Physical-layer gating for UniProt tissue protein seeds lives only in ``UniprotKB`` —
``finalize_biological_graph`` passes ``fetch_protein_seeds`` derived from
``filter_physical_compound``; this module must not call ``_physical_enrich_blocked``.

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

async def _ingest_organ_layer(self, organs: list[str], *, fetch_protein_seeds: bool = True) -> None:
    """STAGE A — Organ-first top-down ingestion.

    Prompt: Make Sure to First fetch Just Organ Data from Uniprot & co based on the input.

    fetch_protein_seeds: when False, organ + disease steps still run; UniProt tissue protein
    seeds are omitted (set by ``UniprotKB.finalize_biological_graph`` from filter_physical_compound).

    1. Resolves each organ term → UBERON ontology ID via OLS4 (exact match first, fuzzy fallback).
    2. Creates ORGAN node in the graph (type=ORGAN, uberon_id, description).
    3. Fetches disease/harmful ontology (HPO + MONDO) linked to each organ via OLS4;
       creates DISEASE nodes and ORGAN_ASSOCIATED_DISEASE edges.
    4. Protein seed fetch (parallel): UniProt (organism_id:9606) AND (tissue:{organ}) — if allowed.
    5. Stores (uberon_id, label) in self._organ_uberon_seeds for tissue layer Block A."""
    if not organs:
        print("  Stage A: no organs supplied — skipping")
        return

    if self.active_subgraph is None:
        self.active_subgraph = set()

    _OLS4_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
    organ_nodes_created = 0
    disease_edges = 0

    for organ in organs:
        # ── 1. Resolve UBERON ID — exact match first, fuzzy fallback ──
        uberon_meta: dict | None = None
        try:
            for exact in ("true", "false"):
                res = await self.client.get(
                    _OLS4_SEARCH,
                    params={"q": organ, "ontology": "uberon", "exact": exact, "rows": 1},
                    timeout=15.0,
                )
                if res.status_code == 200:
                    docs = res.json().get("response", {}).get("docs", [])
                    if docs:
                        hit = docs[0]
                        obo_id = hit.get("obo_id", "") or ""
                        sf = obo_id.replace(":", "_") if obo_id else hit.get("short_form", "")
                        if sf and sf.startswith("UBERON_"):
                            desc_raw = hit.get("description", [])
                            uberon_meta = {
                                "uberon_id": sf,
                                "label": hit.get("label", organ),
                                "description": ". ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw or ""),
                            }
                            break
        except Exception as e:
            print(f"  Stage A OLS4 UBERON error for '{organ}': {e}")

        organ_uberon = uberon_meta["uberon_id"] if uberon_meta else None
        organ_label  = uberon_meta["label"]     if uberon_meta else organ
        organ_nid    = f"ORGAN_{organ_uberon or organ.replace(' ', '_')}"

        # ── 2. Create ORGAN node (anchor for the entire organ hierarchy) ──
        if not self.g.G.has_node(organ_nid):
            self.g.add_node({
                "id": organ_nid,
                "type": "ORGAN",
                "label": organ_label,
                "input_term": organ,
                "uberon_id": organ_uberon or "",
                "description": (uberon_meta or {}).get("description", ""),
                "source": "OLS4_UBERON" if organ_uberon else "input",
            })
            organ_nodes_created += 1

        if organ_uberon:
            self._organ_uberon_seeds.append((organ_uberon, organ_label))

        # ── 3. Disease/harmful ontology for this organ (HPO + MONDO via OLS4) ──
        for ontology in ("hp", "mondo"):
            try:
                res = await self.client.get(
                    _OLS4_SEARCH,
                    params={"q": organ, "ontology": ontology, "rows": 5, "type": "class"},
                    timeout=10.0,
                )
                if res.status_code != 200:
                    continue
                for doc in res.json().get("response", {}).get("docs", []):
                    obo_id = doc.get("obo_id", "")
                    if not obo_id:
                        continue
                    d_nid = f"DISEASE_{obo_id.replace(':', '_')}"
                    if not self.g.G.has_node(d_nid):
                        dr = doc.get("description", [])
                        self.g.add_node({
                            "id": d_nid,
                            "type": "DISEASE",
                            "label": doc.get("label", obo_id),
                            "description": ". ".join(dr) if isinstance(dr, list) else str(dr or ""),
                            "obo_id": obo_id,
                            "ontology": ontology.upper(),
                            "organ_term": organ,
                        })
                    if not self.g.G.has_edge(organ_nid, d_nid):
                        self.g.add_edge(
                            src=organ_nid, trgt=d_nid,
                            attrs={
                                "rel": "ORGAN_ASSOCIATED_DISEASE",
                                "src_layer": "ORGAN",
                                "trgt_layer": "DISEASE",
                                "ontology": ontology.upper(),
                                "source": "OLS4",
                            },
                        )
                        disease_edges += 1
            except Exception as e:
                print(f"  Stage A OLS4 disease error (organ='{organ}', ont={ontology}): {e}")

    # ── 4. Protein seed fetch — parallel over all organs (caller gates via fetch_protein_seeds) ──
    if not fetch_protein_seeds:
        print("  Stage A: UniProt tissue protein seeds skipped — physical filter excludes gene/protein")
    else:
        tasks = [
            self.fetch_proteins_by_query(f"(organism_id:9606) AND (tissue:{organ})")
            for organ in organs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, list):
                self.active_subgraph.update(res)

    print(f"  Stage A: {organ_nodes_created} ORGAN nodes, "
          f"{disease_edges} ORGAN_ASSOCIATED_DISEASE edges, "
          f"{len(self.active_subgraph)} protein seeds from {len(organs)} organ(s)")

