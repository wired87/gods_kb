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

async def enrich_genomic_data(self):
    """Ensembl Integration für chromosomale Daten (batched to avoid 429 storms)."""
    gene_nodes = [v for k, v in self.g.G.nodes(data=True) if v.get("type") == "GENE" and self._is_active(k)]
    _BATCH = 15  # ENSEMBL rate limit: ~15 req/s for anonymous users
    for i in range(0, len(gene_nodes), _BATCH):
        batch = gene_nodes[i : i + _BATCH]
        tasks = [self.fetch_with_retry(
            f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{n['label']}?content-type=application/json")
            for n in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for node, res in zip(batch, results):
            if not isinstance(res, Exception):
                node.update({
                    "ensembl_id": res.get("id"),
                    "chromosome": res.get("seq_region_name"),
                    "gene_start": res.get("start"),
                    "gene_end": res.get("end"),
                })
        if i + _BATCH < len(gene_nodes):
            await asyncio.sleep(1.1)

