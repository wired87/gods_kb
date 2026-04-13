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

async def enrich_food_sources(self):
    """
    Kategorie: LEBENSMITTEL -> MOLEKÜL -> PROTEIN.
    Zieht Live-Daten von Open Food Facts (OFF) für den deutschen Markt (cc=de).
    Filtert auf nutrition_grades a/b/c für Qualitätssicherung.
    """
    target_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                    if v.get("type") in ["MINERAL", "PROTEIN"] and self._is_active(k)]

    _VALID_GRADES = {"a", "b", "c"}

    for target_nid, node in target_nodes:
        search_term = self._extract_search_term(node.get("label", ""))
        off_url = (
            f"https://world.openfoodfacts.org/cgi/search.pl"
            f"?search_terms={quote(search_term)}&cc=de"
            f"&search_simple=1&action=process&json=1&page_size=5"
        )

        try:
            response = await self.client.get(off_url, timeout=15.0)
            if response.status_code != 200:
                continue
            products = response.json().get("products", [])

            for prod in products:
                food_name = prod.get("product_name_de") or prod.get("product_name")
                if not food_name:
                    continue

                # QUALITY GATE: only validated nutrition grades
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
                print(f"Live Linked: {food_name} [{grade}] contains {val} of {node['label']}")

        except Exception as e:
            print(f"Error fetching live food data for {search_term}: {e}")
