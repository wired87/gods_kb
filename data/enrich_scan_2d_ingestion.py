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
from data.main import ScanIngestionLayer

async def enrich_scan_2d_ingestion(self, scan_path: str, modality_hint: str | None = None):
    """
    PHASE 20: Load a 2D medical image into the graph as a RAW_SCAN node.

    Inputs: filesystem path; optional ``modality_hint`` overrides auto-detected modality.
    Outputs: one ``RAW_SCAN`` node with ``path_key`` (sha256 of resolved path, 16 hex chars)
        for stable cross-references without embedding the RAW_SCAN node id in dependent nodes.
    Side effects: reads image via ``ScanIngestionLayer``; attaches ``_pixels`` on the NetworkX
    node dict for Phase 21 (not intended for JSON export). Idempotent on ``path_key``.
    """
    p = Path(scan_path)
    scan_desc = ScanIngestionLayer.load(p)

    # OVERRIDE modality if caller supplies explicit hint
    if modality_hint and modality_hint != "auto":
        scan_desc["modality"] = modality_hint.upper()

    # IDEMPOTENT: sha256 of absolute path → stable node ID
    path_hash = hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:16]
    scan_node_id = f"RAW_SCAN_{path_hash}"

    if self.g.G.has_node(scan_node_id):
        print(f"  RAW_SCAN already in graph: {scan_node_id}")
        return

    # STORE pixel data as transient attribute (_pixels not serialised)
    self.g.add_node({
        "id": scan_node_id,
        "type": "RAW_SCAN",
        "label": p.name,
        "path_key": path_hash,
        "modality": scan_desc["modality"],
        "slice_idx": scan_desc["slice_idx"],
        "spacing_mm": scan_desc["spacing_mm"],
        "pixel_shape": list(scan_desc["pixels"].shape),
        "orientation": scan_desc["orientation"],
        "source_path": scan_desc["source_path"],
    })
    # ATTACH pixel array on the nx node dict for downstream phases
    self.g.G.nodes[scan_node_id]["_pixels"] = scan_desc["pixels"]

    print(f"  RAW_SCAN ingested: {p.name} | modality={scan_desc['modality']} "
          f"| shape={scan_desc['pixels'].shape}")

# ══════════════════════════════════════════════════════════════════
# PHASE 21 — SPATIAL SEGMENTATION (SimpleITK Otsu + Connected Components)
# ══════════════════════════════════════════════════════════════════

