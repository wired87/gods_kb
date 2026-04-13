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
import data.main as _uk

async def enrich_pathology_findings(self):
    """PHASE 22c: Intermediate layer between SCAN_SIGNAL and DISEASE.
    Detects anomalies via z-score against ORGAN_STATE baseline,
    resolves HPO terms from OLS4, and infers DISEASE links by cosine similarity."""

    _OLS4_HPO = "https://www.ebi.ac.uk/ols4/api/search"
    _hpo_cache: dict[str, dict | None] = {}

    async def _resolve_hpo(term: str) -> dict | None:
        key = term.strip().lower()
        if key in _hpo_cache:
            return _hpo_cache[key]
        try:
            res = await self.client.get(
                _OLS4_HPO,
                params={"q": term, "ontology": "hp", "rows": 3, "type": "class"},
                timeout=10.0,
            )
            if res.status_code != 200:
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
            return result
        except Exception:
            _hpo_cache[key] = None
            return None

    # COSINE SIMILARITY helper
    def _cos_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) + 1e-12
        norm_b = math.sqrt(sum(x * x for x in b)) + 1e-12
        return dot / (norm_a * norm_b)

    # COLLECT: ORGAN_STATE baseline feature vectors for z-score
    organ_states = {
        k: v for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "ORGAN_STATE"
    }
    # BASELINE: mean frequency and amplitude across all organ states
    os_freqs = [v.get("mean_frequency_hz", 0.0) for v in organ_states.values()]
    os_amps = [v.get("mean_amplitude_mv", 0.0) for v in organ_states.values()]
    baseline_freq = np.mean(os_freqs) if os_freqs else 0.0
    baseline_amp = np.mean(os_amps) if os_amps else 0.0
    std_freq = np.std(os_freqs) if os_freqs else 1.0
    std_amp = np.std(os_amps) if os_amps else 1.0
    std_freq = max(std_freq, 1e-6)
    std_amp = max(std_amp, 1e-6)

    # COLLECT: disease EM profiles for cosine matching (from Phase 19.6 DISEASE nodes)
    disease_profiles: dict[str, list[float]] = {}
    disease_nodes = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "DISEASE"
    ]
    for dis_id, dis in disease_nodes:
        # BUILD 5D proxy from disease-linked SCAN_SIGNALs
        linked_vecs: list[list[float]] = []
        for pred in self.g.G.predecessors(dis_id):
            pd = self.g.G.nodes.get(pred, {})
            if pd.get("type") == "SCAN_SIGNAL" and "modality_feature_vec" in pd:
                linked_vecs.append(pd["modality_feature_vec"])

        if not linked_vecs:
            # FALLBACK: construct from EM signature aggregates
            freq = 0.0
            amp = 0.0
            count = 0
            for pred in self.g.G.predecessors(dis_id):
                pt = self.g.G.nodes.get(pred, {}).get("type")
                if pt in ("GENE", "PROTEIN"):
                    for nb in self.g.G.neighbors(pred):
                        nbd = self.g.G.nodes.get(nb, {})
                        if nbd.get("type") == "ELECTRICAL_COMPONENT":
                            for bs in self.g.G.neighbors(nb):
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

        # AVERAGE linked vectors
        n = len(linked_vecs)
        avg_vec = [sum(v[d] for v in linked_vecs) / n for d in range(_uk._SCAN_FEAT_DIM)]
        disease_profiles[dis_id] = avg_vec

    # OUTSRC CRITERIA: bias HPO resolution toward user-supplied harmful terms
    _outsrc_terms_lc = {t.lower() for t in (self.workflow_cfg or {}).get("outsrc_criteria", [])}

    # MODALITY → HPO search terms for anomaly classification
    _MODALITY_ANOMALY_TERMS: dict[str, list[str]] = {
        "MRI":        ["abnormal signal intensity", "brain lesion", "atrophy"],
        "MRI_T1":     ["white matter abnormality", "brain atrophy", "hypointensity"],
        "MRI_T2":     ["hyperintensity", "edema", "demyelination"],
        "CT":         ["calcification", "hyperdensity", "mass lesion"],
        "PET":        ["hypermetabolism", "hypometabolism", "abnormal uptake"],
        "ULTRASOUND": ["echogenic lesion", "cyst", "mass"],
        "FMRI":       ["abnormal activation", "reduced connectivity", "cortical dysfunction"],
    }

    # PROCESS: each SCAN_SIGNAL that has a modality_feature_vec
    scan_signals = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "SCAN_SIGNAL" and v.get("modality_feature_vec")
    ]

    _finding_count = 0
    _disease_link_count = 0
    for sig_id, sig in scan_signals:
        fvec = sig["modality_feature_vec"]
        modality = sig.get("modality", "UNKNOWN")

        # ANOMALY SCORE: z-score of signal features vs organ state baseline
        z_freq = abs(sig.get("measured_frequency_hz", 0.0) - baseline_freq) / std_freq
        z_amp = abs(sig.get("measured_amplitude_mv", 0.0) - baseline_amp) / std_amp
        anomaly_score = round((z_freq + z_amp) / 2.0, 4)

        # SKIP low-anomaly signals
        if anomaly_score < 0.3:
            continue

        # HPO TERM RESOLUTION: pick term based on dominant feature dimension
        dominant_dim = int(np.argmax(fvec))
        anomaly_terms = _MODALITY_ANOMALY_TERMS.get(modality, ["abnormality"])
        search_term = anomaly_terms[min(dominant_dim, len(anomaly_terms) - 1)]

        hpo = await _resolve_hpo(search_term)
        hpo_id = hpo["hpo_id"] if hpo else ""
        hpo_label = hpo["label"] if hpo else search_term

        # CREATE PATHOLOGY_FINDING node
        finding_id = f"PATHOLOGY_FINDING_{sig_id}"
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
        })

        # EDGE: SCAN_SIGNAL → PATHOLOGY_FINDING
        self.g.add_edge(
            src=sig_id, trgt=finding_id,
            attrs={"rel": "SUGGESTS_FINDING", "src_layer": "SCAN_SIGNAL", "trgt_layer": "PATHOLOGY_FINDING"},
        )
        _finding_count += 1

        # COSINE MATCHING: PATHOLOGY_FINDING → DISEASE (outsrc-boosted)
        matches: list[tuple[str, float]] = []
        for dis_id, dis_vec in disease_profiles.items():
            if len(dis_vec) != _uk._SCAN_FEAT_DIM:
                continue
            sim = _cos_sim(fvec, dis_vec)
            # BOOST: outsrc-matched diseases get priority in ranking
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

# ══════════════════════════════════════════════════════════════════
# CFG-DRIVEN HIERARCHICAL SEED LAYERS
# ══════════════════════════════════════════════════════════════════

