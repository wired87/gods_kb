"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``main``.

Prompt (user): data-dir graph hardening — non-nested node ids for SCAN_SIGNAL.

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

from data.graph_identity import canonical_node_id

async def enrich_scan_feature_extraction(self):
    """
    PHASE 22b: Build a 5D ``modality_feature_vec`` per ``SPATIAL_REGION`` as ``SCAN_SIGNAL``.

    Inputs: ``SPATIAL_REGION`` nodes with ``path_key`` and ``component_label_idx`` (Phase 21).
    Outputs: ``SCANSIG_*`` nodes (opaque ids), ``PRODUCES_SIGNAL`` edges from region to signal.
    Side effects: none external; pure graph + numeric features.
    """
    spatial_nodes = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "SPATIAL_REGION"
    ]

    all_means = [v.get("pixel_mean", 0.0) for _, v in spatial_nodes]
    global_max = max(all_means) if all_means else 1.0
    global_min = min(all_means) if all_means else 0.0
    intensity_range = (global_max - global_min) or 1.0

    _signal_count = 0
    for reg_id, reg in spatial_nodes:
        modality = reg.get("modality", "UNKNOWN")
        px_mean = reg.get("pixel_mean", 0.0)
        px_std = reg.get("pixel_std", 0.0)
        px_min = reg.get("pixel_min", 0.0)
        px_max = reg.get("pixel_max", 0.0)

        intensity = (px_mean - global_min) / intensity_range

        local_range = px_max - px_min
        contrast = local_range / intensity_range if intensity_range > 0 else 0.0

        if modality in ("MRI_T1", "MRI"):
            mod_proxy = intensity
        elif modality == "MRI_T2":
            mod_proxy = 1.0 - intensity
        elif modality == "CT":
            mod_proxy = (px_mean + 1000.0) / 4000.0
        elif modality == "PET":
            mod_proxy = intensity * 1.5
        elif modality == "FMRI":
            mod_proxy = min(px_std / (px_mean + 1e-6), 1.0)
        else:
            mod_proxy = intensity

        variance = min(px_std / (intensity_range + 1e-6), 1.0)

        edge_density = local_range / (px_mean + 1e-6)
        edge_density = min(edge_density, 1.0)

        feature_vec = [
            round(intensity, 6),
            round(contrast, 6),
            round(mod_proxy, 6),
            round(variance, 6),
            round(edge_density, 6),
        ]

        pk = reg.get("path_key") or ""
        li = int(reg.get("component_label_idx", -1))
        scan_signal_id = canonical_node_id(
            "SCANSIG",
            {"path_key": pk, "label_idx": li, "modality": modality},
        )

        self.g.add_node({
            "id": scan_signal_id,
            "type": "SCAN_SIGNAL",
            "label": f"ScanSig_{reg.get('label', reg_id)}",
            "sensor_type": modality,
            "modality": modality,
            "spatial_region_id": reg_id,
            "path_key": pk,
            "component_label_idx": li,
            "modality_feature_vec": feature_vec,
            "measured_frequency_hz": round(feature_vec[0] * 100.0, 4),
            "measured_amplitude_mv": round(feature_vec[1] * 50.0, 4),
            "measured_phase_rad": round(feature_vec[2] * 2.0 * math.pi, 6),
            "source_organ": reg.get("anatomy_label", "unknown"),
        })

        self.g.add_edge(
            src=reg_id, trgt=scan_signal_id,
            attrs={"rel": "PRODUCES_SIGNAL", "src_layer": "SPATIAL_REGION", "trgt_layer": "SCAN_SIGNAL"},
        )
        _signal_count += 1

    print(f"  Phase 22b: {_signal_count} SCAN_SIGNAL nodes from spatial regions")

# --- PHASE 22c — PATHOLOGY FINDING INFERENCE (HPO + cosine → DISEASE) ---
