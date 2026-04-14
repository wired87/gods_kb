"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``finalize_biological_graph``.

Prompt (user): data-dir graph hardening — PATHOLOGY_FINDING ids, config-driven HPO terms, logging.

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

from execution_cfg import SCAN_PATHOLOGY_HPO_TERMS_BY_MODALITY
from data.graph_identity import canonical_node_id, phase_http_log, timed_ms

async def enrich_pathology_findings(self):
    """
    PHASE 22c: SCAN_SIGNAL → PATHOLOGY_FINDING (HPO) → DISEASE via cosine similarity.

    Inputs: ``SCAN_SIGNAL`` nodes with ``modality_feature_vec``; ``DISEASE`` nodes with linked
    signals or bioelectric fallbacks; optional ``workflow_cfg['outsrc_criteria']`` for ranking boost.
    Outputs: ``PATHFIND_*`` finding nodes (opaque ids), ``SUGGESTS_FINDING``, ``INFERRED_DISEASE`` edges.
    Side effects: GET OLS4 HPO search; bounded fallback graph walks when building disease profiles.
    Failures: HPO lookup returns empty id/label fallback to search term; cosine matches threshold 0.1.
    """
    _OLS4_HPO = "https://www.ebi.ac.uk/ols4/api/search"
    _hpo_cache: dict[str, dict | None] = {}
    _FALLBACK_EDGE_BUDGET = 96

    async def _resolve_hpo(term: str) -> dict | None:
        key = term.strip().lower()
        if key in _hpo_cache:
            return _hpo_cache[key]
        t0 = timed_ms()
        try:
            res = await self.client.get(
                _OLS4_HPO,
                params={"q": term, "ontology": "hp", "rows": 3, "type": "class"},
                timeout=10.0,
            )
            if res.status_code != 200:
                phase_http_log(
                    "phase22c_pathology", "ols4_hpo",
                    _OLS4_HPO, status_code=res.status_code, elapsed_ms=timed_ms() - t0,
                )
                _hpo_cache[key] = None
                return None
            docs = res.json().get("response", {}).get("docs", [])
            if not docs:
                _hpo_cache[key] = None
                return None
            best = docs[0]
            result = {
                "hpo_id": best.get("obo_id", ""),
                "label": best.get("label", term),
                "description": (best.get("description", [""])[0]
                                if best.get("description") else ""),
            }
            _hpo_cache[key] = result
            phase_http_log(
                "phase22c_pathology", "ols4_hpo_ok",
                _OLS4_HPO, status_code=res.status_code, elapsed_ms=timed_ms() - t0,
            )
            return result
        except Exception as e:
            phase_http_log(
                "phase22c_pathology", "ols4_hpo_err",
                _OLS4_HPO, status_code=None, elapsed_ms=timed_ms() - t0,
                err_class=type(e).__name__,
            )
            _hpo_cache[key] = None
            return None

    def _cos_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) + 1e-12
        norm_b = math.sqrt(sum(x * x for x in b)) + 1e-12
        return dot / (norm_a * norm_b)

    organ_states = {
        k: v for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "ORGAN_STATE"
    }
    os_freqs = [v.get("mean_frequency_hz", 0.0) for v in organ_states.values()]
    os_amps = [v.get("mean_amplitude_mv", 0.0) for v in organ_states.values()]
    baseline_freq = np.mean(os_freqs) if os_freqs else 0.0
    baseline_amp = np.mean(os_amps) if os_amps else 0.0
    std_freq = max(np.std(os_freqs) if os_freqs else 1.0, 1e-6)
    std_amp = max(np.std(os_amps) if os_amps else 1.0, 1e-6)

    disease_profiles: dict[str, list[float]] = {}
    disease_nodes = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "DISEASE"
    ]
    for dis_id, dis in disease_nodes:
        linked_vecs: list[list[float]] = []
        for pred in self.g.G.predecessors(dis_id):
            pd = self.g.G.nodes.get(pred, {})
            if pd.get("type") == "SCAN_SIGNAL" and "modality_feature_vec" in pd:
                linked_vecs.append(pd["modality_feature_vec"])

        if not linked_vecs:
            freq = 0.0
            amp = 0.0
            count = 0
            budget = _FALLBACK_EDGE_BUDGET
            for pred in self.g.G.predecessors(dis_id):
                if budget <= 0:
                    break
                pt = self.g.G.nodes.get(pred, {}).get("type")
                if pt not in ("GENE", "PROTEIN"):
                    continue
                for nb in self.g.G.neighbors(pred):
                    if budget <= 0:
                        break
                    budget -= 1
                    nbd = self.g.G.nodes.get(nb, {})
                    if nbd.get("type") != "ELECTRICAL_COMPONENT":
                        continue
                    for bs in self.g.G.neighbors(nb):
                        if budget <= 0:
                            break
                        budget -= 1
                        bsd = self.g.G.nodes.get(bs, {})
                        if bsd.get("type") == "BIOELECTRIC_STATE":
                            freq += bsd.get("frequency_signature_hz", 0.0)
                            amp += abs(bsd.get("membrane_potential_mV", 0.0))
                            count += 1
            if count > 0:
                avg_f = freq / count
                avg_a = amp / count
                disease_profiles[dis_id] = [
                    avg_f / 100.0, avg_a / 50.0, 0.5, 0.5, 0.5,
                ]
            continue

        n = len(linked_vecs)
        avg_vec = [sum(v[d] for v in linked_vecs) / n for d in range(_uk._SCAN_FEAT_DIM)]
        disease_profiles[dis_id] = avg_vec

    _outsrc_terms_lc = {t.lower() for t in (self.workflow_cfg or {}).get("outsrc_criteria", [])}

    scan_signals = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "SCAN_SIGNAL" and v.get("modality_feature_vec")
    ]

    _finding_count = 0
    _disease_link_count = 0
    for sig_id, sig in scan_signals:
        fvec = sig["modality_feature_vec"]
        modality = sig.get("modality", "UNKNOWN")

        z_freq = abs(sig.get("measured_frequency_hz", 0.0) - baseline_freq) / std_freq
        z_amp = abs(sig.get("measured_amplitude_mv", 0.0) - baseline_amp) / std_amp
        anomaly_score = round((z_freq + z_amp) / 2.0, 4)

        if anomaly_score < 0.3:
            continue

        dominant_dim = int(np.argmax(fvec))
        anomaly_terms = SCAN_PATHOLOGY_HPO_TERMS_BY_MODALITY.get(modality, ["abnormality"])
        search_term = anomaly_terms[min(dominant_dim, len(anomaly_terms) - 1)]

        hpo = await _resolve_hpo(search_term)
        hpo_id = hpo["hpo_id"] if hpo else ""
        hpo_label = hpo["label"] if hpo else search_term

        finding_id = canonical_node_id(
            "PATHFIND",
            {
                "scan_signal_id": sig_id,
                "hpo_search": search_term,
                "dominant_dim": dominant_dim,
            },
        )
        self.g.add_node({
            "id": finding_id,
            "type": "PATHOLOGY_FINDING",
            "label": f"Finding_{hpo_label}",
            "hpo_id": hpo_id,
            "hpo_label": hpo_label,
            "anomaly_score": anomaly_score,
            "finding_type": search_term,
            "modality": modality,
            "dominant_feature_dim": dominant_dim,
            "source_scan_signal": sig_id,
        })

        self.g.add_edge(
            src=sig_id, trgt=finding_id,
            attrs={"rel": "SUGGESTS_FINDING", "src_layer": "SCAN_SIGNAL", "trgt_layer": "PATHOLOGY_FINDING"},
        )
        _finding_count += 1

        matches: list[tuple[str, float]] = []
        for dis_id, dis_vec in disease_profiles.items():
            if len(dis_vec) != _uk._SCAN_FEAT_DIM:
                continue
            sim = _cos_sim(fvec, dis_vec)
            if _outsrc_terms_lc and self.g.G.nodes.get(dis_id, {}).get("outsrc_match"):
                sim = min(sim * 1.25, 1.0)
            if sim > 0.1:
                matches.append((dis_id, sim))

        matches.sort(key=lambda x: x[1], reverse=True)

        for dis_id, confidence in matches[:5]:
            self.g.add_edge(
                src=finding_id, trgt=dis_id,
                attrs={
                    "rel": "INFERRED_DISEASE",
                    "confidence_score": round(confidence, 4),
                    "anomaly_score": anomaly_score,
                    "finding_type": search_term,
                    "src_layer": "PATHOLOGY_FINDING",
                    "trgt_layer": "DISEASE",
                },
            )
            _disease_link_count += 1

    print(f"  Phase 22c: {_finding_count} PATHOLOGY_FINDING nodes, "
          f"{_disease_link_count} INFERRED_DISEASE edges")

# --- CFG-DRIVEN HIERARCHICAL SEED LAYERS ---
