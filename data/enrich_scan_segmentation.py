"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``main``.

Prompt (user): data-dir graph hardening — non-nested node ids, config-driven scan terms, logging.

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
import data.main as _uk

from execution_cfg import SCAN_ANATOMY_TERM_TIERS_BY_MODALITY
from data.graph_identity import canonical_node_id, phase_http_log, timed_ms

async def enrich_scan_segmentation(self):
    """
    PHASE 21: Segment ``RAW_SCAN`` nodes into ``SPATIAL_REGION`` nodes (SimpleITK).

    Inputs: ``RAW_SCAN`` nodes with ``_pixels`` and ``path_key`` (or legacy id prefix ``RAW_SCAN_``).
    Outputs: ``SPATREG_*`` region nodes (opaque ids), ``CONTAINS_REGION`` edges from scan;
    sets ``uberon_id`` / ``anatomy_label`` via OLS4 using tiers from
    ``execution_cfg.SCAN_ANATOMY_TERM_TIERS_BY_MODALITY``.
    Side effects: HTTP GET to EBI OLS4; CPU segmentation per scan.
    Failures: OLS4 non-200 or exceptions cache ``None`` for term; segmentation errors propagate from SimpleITK.
    """
    import SimpleITK as sitk

    _OLS4_UBERON = "https://www.ebi.ac.uk/ols4/api/search"
    _uberon_cache: dict[str, dict | None] = {}

    async def _resolve_uberon(term: str) -> dict | None:
        key = term.strip().lower()
        if key in _uberon_cache:
            return _uberon_cache[key]
        params = {"q": term, "ontology": "uberon", "rows": 3, "type": "class"}
        t0 = timed_ms()
        try:
            res = await self.client.get(_OLS4_UBERON, params=params, timeout=10.0)
            if res.status_code != 200:
                phase_http_log(
                    "phase21_scan_seg", "ols4_uberon",
                    _OLS4_UBERON, status_code=res.status_code, elapsed_ms=timed_ms() - t0,
                )
                _uberon_cache[key] = None
                return None
            docs = res.json().get("response", {}).get("docs", [])
            if not docs:
                _uberon_cache[key] = None
                return None
            best = docs[0]
            result = {
                "uberon_id": best.get("obo_id", ""),
                "label": best.get("label", term),
                "synonyms": best.get("synonym", []),
            }
            _uberon_cache[key] = result
            phase_http_log(
                "phase21_scan_seg", "ols4_uberon_ok",
                _OLS4_UBERON, status_code=res.status_code, elapsed_ms=timed_ms() - t0,
            )
            return result
        except Exception as e:
            phase_http_log(
                "phase21_scan_seg", "ols4_uberon_err",
                _OLS4_UBERON, status_code=None, elapsed_ms=timed_ms() - t0,
                err_class=type(e).__name__,
            )
            _uberon_cache[key] = None
            return None

    raw_scans = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "RAW_SCAN" and "_pixels" in v
    ]

    _total_regions = 0
    for scan_id, scan in raw_scans:
        pixels = scan["_pixels"]
        modality = scan.get("modality", "UNKNOWN")
        path_key = scan.get("path_key")
        if not path_key and isinstance(scan_id, str) and scan_id.startswith("RAW_SCAN_"):
            path_key = scan_id[len("RAW_SCAN_"):]
        if not path_key:
            path_key = hashlib.sha256(str(scan_id).encode()).hexdigest()[:16]

        sitk_img = sitk.GetImageFromArray(pixels.astype(np.float32))

        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        binary = otsu_filter.Execute(sitk_img)

        cc_filter = sitk.ConnectedComponentImageFilter()
        labelled = cc_filter.Execute(binary)
        n_labels = cc_filter.GetObjectCount()

        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(sitk_img, labelled)

        region_count = 0
        for label_idx in range(1, n_labels + 1):
            if _total_regions >= _uk._MAX_SPATIAL_REGIONS:
                break
            if not stats.HasLabel(label_idx):
                continue

            count = stats.GetCount(label_idx)
            if count < 10:
                continue

            bbox = stats.GetBoundingBox(label_idx)
            px_mean = stats.GetMean(label_idx)
            px_std = stats.GetSigma(label_idx)
            px_min = stats.GetMinimum(label_idx)
            px_max = stats.GetMaximum(label_idx)

            if len(bbox) >= 4:
                cx = bbox[0] + bbox[2] / 2.0
                cy = bbox[1] + bbox[3] / 2.0
            else:
                cx, cy = 0.0, 0.0

            region_id = canonical_node_id(
                "SPATREG", {"path_key": path_key, "label_idx": int(label_idx)},
            )
            self.g.add_node({
                "id": region_id,
                "type": "SPATIAL_REGION",
                "label": f"Region_{label_idx}_{scan.get('label', scan_id)}",
                "scan_source": scan_id,
                "path_key": path_key,
                "component_label_idx": int(label_idx),
                "modality": modality,
                "bounding_box": list(bbox),
                "pixel_mean": round(float(px_mean), 4),
                "pixel_std": round(float(px_std), 4),
                "pixel_min": round(float(px_min), 4),
                "pixel_max": round(float(px_max), 4),
                "area_px": int(count),
                "centroid_xy": [round(cx, 2), round(cy, 2)],
            })

            self.g.add_edge(
                src=scan_id, trgt=region_id,
                attrs={"rel": "CONTAINS_REGION", "src_layer": "RAW_SCAN", "trgt_layer": "SPATIAL_REGION"},
            )

            region_count += 1
            _total_regions += 1

        region_nodes = [
            (k, v) for k, v in self.g.G.nodes(data=True)
            if v.get("type") == "SPATIAL_REGION" and v.get("scan_source") == scan_id
        ]

        sorted_regions = sorted(region_nodes, key=lambda x: x[1].get("pixel_mean", 0), reverse=True)

        tiers = SCAN_ANATOMY_TERM_TIERS_BY_MODALITY.get(
            modality, ["tissue", "connective tissue", "background"],
        )

        for idx, (reg_id, reg) in enumerate(sorted_regions):
            tier_idx = min(idx, len(tiers) - 1)
            anatomy_term = tiers[tier_idx]

            uberon = await _resolve_uberon(anatomy_term)
            if uberon and uberon.get("uberon_id"):
                self.g.G.nodes[reg_id]["uberon_id"] = uberon["uberon_id"]
                self.g.G.nodes[reg_id]["anatomy_label"] = uberon["label"]

        print(f"  Scan {scan_id}: {region_count} SPATIAL_REGION nodes | path_key={path_key}")

    print(f"  Phase 21 complete: {_total_regions} total regions")

# --- PHASE 22a — UBERON BRIDGE (SPATIAL_REGION → TISSUE / ORGAN) ---
