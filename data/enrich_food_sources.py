"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``finalize_biological_graph``.

Prompt (user): data-dir graph hardening — bounded concurrency + structured HTTP logging for OFF.

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

from data.graph_identity import phase_http_log, phase_log, timed_ms

_OFF_MAX_CONCURRENCY = 2
_OFF_MAX_PRODUCTS_PER_TARGET = 3


async def enrich_food_sources(self):
    """
    PHASE 5: Open Food Facts (DE) → FOOD_SOURCE nodes linked to MINERAL / PROTEIN targets.

    Inputs: active ``MINERAL`` and ``PROTEIN`` nodes; search term from ``_extract_search_term``.
    Outputs: ``FOOD_*`` nodes (OFF product id), ``CONTAINS_NUTRIENT`` edges with amount attrs.
    Side effects: HTTP GET world.openfoodfacts.org; bounded parallel fetches (semaphore).
    Failures: non-200 or exceptions logged; target skipped without raising.
    """
    target_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                    if v.get("type") in ["MINERAL", "PROTEIN"] and self._is_active(k)]

    _VALID_GRADES = {"a", "b", "c"}
    sem = asyncio.Semaphore(_OFF_MAX_CONCURRENCY)

    async def _one_target(target_nid: str, node: dict) -> None:
        search_term = self._extract_search_term(node.get("label", ""))
        off_url = (
            f"https://world.openfoodfacts.org/cgi/search.pl"
            f"?search_terms={quote(search_term)}&cc=de"
            f"&search_simple=1&action=process&json=1&page_size=5"
        )
        t0 = timed_ms()
        async with sem:
            try:
                response = await self.client.get(off_url, timeout=15.0)
                if response.status_code != 200:
                    phase_http_log(
                        "phase5_food", "openfoodfacts",
                        off_url, status_code=response.status_code, elapsed_ms=timed_ms() - t0,
                    )
                    return
                products = response.json().get("products", [])
                phase_http_log(
                    "phase5_food", "openfoodfacts_ok",
                    off_url, status_code=response.status_code, elapsed_ms=timed_ms() - t0,
                )
                n_linked = 0
                for prod in products:
                    if n_linked >= _OFF_MAX_PRODUCTS_PER_TARGET:
                        break
                    food_name = prod.get("product_name_de") or prod.get("product_name")
                    if not food_name:
                        continue

                    grade = (prod.get("nutrition_grades") or "").lower()
                    nutriments = prod.get("nutriments", {})
                    val = nutriments.get(f"{search_term.lower()}_100g", 0) or nutriments.get("proteins_100g", 0)

                    if val <= 0:
                        continue
                    if grade not in _VALID_GRADES and not val:
                        continue

                    food_id = f"FOOD_{prod.get('_id')}"
                    self.g.add_node({
                        "id": food_id,
                        "type": "FOOD_SOURCE",
                        "label": food_name,
                        "brand": prod.get("brands"),
                        "nova_group": prod.get("nova_group"),
                        "ecoscore": prod.get("ecoscore_grade"),
                        "nutrition_grade": grade,
                        "image_url": prod.get("image_url"),
                    })

                    self.g.add_edge(
                        src=food_id,
                        trgt=target_nid,
                        attrs={
                            "rel": "CONTAINS_NUTRIENT",
                            "amount_per_100g": val,
                            "unit": nutriments.get(f"{search_term.lower()}_unit", "g"),
                            "src_layer": "FOOD",
                            "trgt_layer": node.get("type", "PROTEIN"),
                        },
                    )
                    n_linked += 1
                    print(f"Live Linked: {food_name} [{grade}] contains {val} of {node['label']}")

                phase_log(
                    "phase5_food", "target_done",
                    entity_type=node.get("type"),
                    products_linked=n_linked,
                    search_term=search_term[:40],
                )
            except Exception as e:
                phase_http_log(
                    "phase5_food", "openfoodfacts_err",
                    off_url, status_code=None, elapsed_ms=timed_ms() - t0,
                    err_class=type(e).__name__,
                )

    await asyncio.gather(*[_one_target(k, v) for k, v in target_nodes])
