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

async def enrich_bioelectric_disease_signal_pipeline(self):
    """
    MASTER PIPELINE: Biology → Electromagnetic Signal → Disease Inference.

    Pass 1 — DISEASE nodes via OpenTargets + OLS4 MONDO
    Pass 2 — BIOELECTRIC_STATE from ELECTRICAL_COMPONENT (GHK-derived)
    Pass 3 — EM_SIGNATURE from BIOELECTRIC_STATE
    Pass 4 — Multi-scale: CELL_STATE → TISSUE_STATE → ORGAN_STATE
    Pass 5 — SCAN_SIGNAL per organ × sensor
    Pass 6 — Inverse inference: SCAN_SIGNAL → DISEASE (cosine similarity)
    """

    # ── CONSTANTS (all inside method – no module pollution) ───────
    _MAX_DISEASE_NODES = 2000
    _MAX_DISEASE_LINKS_PER_ENTITY = 5
    _MIN_OT_SCORE = 0.1
    _SCAN_SENSORS = ("EM", "ULTRASOUND", "MRI")
    _RESTING_VM_MV = -70.0          # DEFAULT resting membrane potential
    _MEMBRANE_AREA_UM2 = 10.0       # µm² – single-channel patch area
    _TWO_PI = 2.0 * math.pi

    # ION WEIGHTING: relative permeability ratios (GHK-like)
    _ION_WEIGHTS: dict[str, float] = {
        "na": 1.0, "na+": 1.0, "sodium": 1.0,
        "k": 0.8, "k+": 0.8, "potassium": 0.8,
        "ca": 1.5, "ca2+": 1.5, "calcium": 1.5,
        "cl": 0.6, "cl-": 0.6, "chloride": 0.6,
        "h": 0.3, "h+": 0.3, "proton": 0.3,
    }

    # ── OLS4 / UBERGRAPH ENDPOINTS for dynamic anatomy resolution ──
    _OLS4_UBERON = "https://www.ebi.ac.uk/ols4/api/search"
    _UBERGRAPH_SPARQL = "https://ubergraph.apps.renci.org/sparql"
    _MAX_ORGAN_NODES = 300
    _uberon_cache: dict[str, dict | None] = {}   # label → {uberon_id, label, synonyms}
    _organ_for_tissue: dict[str, str] = {}        # tissue_id → organ_node_id

    # ── HELPER: OLS4 UBERON lookup (tissue label → organ UBERON ID) ──
    async def _resolve_uberon_organ(term: str) -> dict | None:
        """OLS4 UBERON: term → {uberon_id, label, synonyms, description}."""
        key = term.strip().lower()
        if key in _uberon_cache:
            return _uberon_cache[key]

        try:
            res = await self.client.get(
                _OLS4_UBERON,
                params={"q": term, "ontology": "uberon", "rows": 3, "type": "class"},
                timeout=10.0,
            )
            if res.status_code != 200:
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
                "description": (best.get("description", [""])[0]
                                if best.get("description") else ""),
                "iri": best.get("iri", ""),
            }
            _uberon_cache[key] = result
            return result
        except Exception:
            _uberon_cache[key] = None
            return None

    # ── HELPER: Ubergraph SPARQL → parent organ for a UBERON tissue ──
    async def _ubergraph_parent_organ(uberon_id: str) -> dict | None:
        """Ubergraph: find the organ a tissue is part_of."""
        # BFO_0000050 = "part of"
        obo_uri = f"http://purl.obolibrary.org/obo/{uberon_id.replace(':', '_')}"
        query = (
            "SELECT DISTINCT ?organ ?organLabel WHERE { "
            f"<{obo_uri}> <http://purl.obolibrary.org/obo/BFO_0000050> ?organ . "
            "?organ <http://www.w3.org/2000/01/rdf-schema#label> ?organLabel . "
            "FILTER(STRSTARTS(STR(?organ), 'http://purl.obolibrary.org/obo/UBERON_')) "
            "} LIMIT 5"
        )
        try:
            res = await self.client.get(
                _UBERGRAPH_SPARQL,
                params={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=15.0,
            )
            if res.status_code != 200:
                return None

            bindings = res.json().get("results", {}).get("bindings", [])
            if not bindings:
                return None

            first = bindings[0]
            organ_uri = first.get("organ", {}).get("value", "")
            organ_label = first.get("organLabel", {}).get("value", "")
            # URI → UBERON:XXXXXXX
            organ_uberon = organ_uri.split("/")[-1].replace("_", ":")
            return {"uberon_id": organ_uberon, "label": organ_label}
        except Exception:
            return None

    # SENSOR PHYSICS: noise + resolution per modality
    _SENSOR_PROFILES: dict[str, dict] = {
        "EM":         {"noise_sigma": 0.05, "resolution_mm": 1.0},
        "ULTRASOUND": {"noise_sigma": 0.10, "resolution_mm": 0.5},
        "MRI":        {"noise_sigma": 0.02, "resolution_mm": 2.0},
    }

    # ── HELPER: deterministic RNG from node ID ───────────────────
    def _seeded_rng(node_id: str) -> random.Random:
        seed = int(hashlib.sha256(node_id.encode()).hexdigest(), 16) % (2**32)
        return random.Random(seed)

    # ── HELPER: cosine similarity on feature vectors ─────────────
    def _cos_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) + 1e-12
        norm_b = math.sqrt(sum(x * x for x in b)) + 1e-12
        return dot / (norm_a * norm_b)

    # ══════════════════════════════════════════════════════════════
    # PASS 1 — DISEASE ONTOLOGY (OpenTargets + OLS4 MONDO)
    # ══════════════════════════════════════════════════════════════
    print("  P19.1: Disease Ontology Integration (OpenTargets + MONDO)")

    # OUTSRC CRITERIA BIAS: lowercased terms for substring match on disease labels
    _outsrc_terms = {t.lower() for t in (self.workflow_cfg or {}).get("outsrc_criteria", [])}

    _disease_count = 0
    _disease_edge_count = 0
    _OLS4_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"

    # A: collect all GENE nodes with ensembl_id
    gene_nodes = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "GENE" and v.get("ensembl_id") and self._is_active(k)
    ]

    # B: OpenTargets → DISEASE nodes + GENE→DISEASE edges
    for gene_id, gene in gene_nodes:
        if _disease_count >= _MAX_DISEASE_NODES:
            break

        ensembl_id = gene["ensembl_id"]
        try:
            ot_res = await self.client.post(
                self._OT_URL,
                json={
                    "query": self._OT_FULL_DISEASE_QUERY,
                    "variables": {"ensgId": ensembl_id},
                },
                timeout=20.0,
            )
            if ot_res.status_code != 200:
                continue

            target_data = ot_res.json().get("data", {}).get("target", {})
            rows = (target_data.get("associatedDiseases") or {}).get("rows", [])

            gene_links = 0
            for row in rows:
                if _disease_count >= _MAX_DISEASE_NODES:
                    break
                if gene_links >= _MAX_DISEASE_LINKS_PER_ENTITY:
                    break

                disease_raw = row.get("disease", {})
                disease_ot_id = disease_raw.get("id", "")
                disease_name = disease_raw.get("name", "")
                score = row.get("score", 0)

                if score < _MIN_OT_SCORE or not disease_ot_id:
                    continue

                # THERAPEUTIC AREA → category
                areas = disease_raw.get("therapeuticAreas") or []
                category = areas[0].get("label", "UNCLASSIFIED") if areas else "UNCLASSIFIED"

                disease_node_id = f"DISEASE_{disease_ot_id.replace(':', '_')}"

                if not self.g.G.has_node(disease_node_id):
                    # OLS4 MONDO metadata enrichment
                    mondo_label = disease_name
                    mondo_desc = ""
                    try:
                        ols_res = await self.client.get(
                            _OLS4_SEARCH,
                            params={"q": disease_ot_id, "ontology": "mondo", "rows": 1},
                            timeout=10.0,
                        )
                        if ols_res.status_code == 200:
                            docs = ols_res.json().get("response", {}).get("docs", [])
                            if docs:
                                mondo_label = docs[0].get("label", disease_name)
                                mondo_desc = docs[0].get("description", [""])[0] if docs[0].get("description") else ""
                    except Exception:
                        pass

                    # OUTSRC MATCH: flag diseases matching harmful criteria
                    _label_lc = mondo_label.lower()
                    _outsrc_hit = any(ot in _label_lc for ot in _outsrc_terms) if _outsrc_terms else False

                    self.g.add_node({
                        "id": disease_node_id,
                        "type": "DISEASE",
                        "label": mondo_label,
                        "category": category,
                        "disease_ot_id": disease_ot_id,
                        "description": mondo_desc,
                        "max_association_score": score,
                        "outsrc_match": _outsrc_hit,
                    })
                    _disease_count += 1

                else:
                    # UPDATE max score if higher
                    existing = self.g.G.nodes[disease_node_id]
                    if score > existing.get("max_association_score", 0):
                        existing["max_association_score"] = score

                # EDGE: GENE → DISEASE
                self.g.add_edge(
                    src=gene_id, trgt=disease_node_id,
                    attrs={
                        "rel": "ASSOCIATED_WITH",
                        "score": score,
                        "src_layer": "GENE",
                        "trgt_layer": "DISEASE",
                    },
                )
                _disease_edge_count += 1
                gene_links += 1

                # EDGE: PROTEIN → DISEASE (traverse GENE→PROTEIN link back)
                for neighbor in self.g.G.neighbors(gene_id):
                    nb = self.g.G.nodes[neighbor]
                    if nb.get("type") == "PROTEIN":
                        self.g.add_edge(
                            src=neighbor, trgt=disease_node_id,
                            attrs={
                                "rel": "IMPLICATED_IN",
                                "score": score,
                                "src_layer": "PROTEIN",
                                "trgt_layer": "DISEASE",
                            },
                        )
                        _disease_edge_count += 1

        except Exception as e:
            print(f"    OT Error for {gene.get('label')}: {e}")

    # C: CELL_TYPE → DISEASE + TISSUE → DISEASE (via shared GENE neighbours)
    for disease_nid in [
        k for k, v in self.g.G.nodes(data=True) if v.get("type") == "DISEASE"
    ]:
        # find GENEs linked to this disease
        linked_genes = [
            n for n in self.g.G.neighbors(disease_nid)
            if self.g.G.nodes[n].get("type") == "GENE"
        ] if disease_nid in self.g.G else []

        for gid in linked_genes:
            cell_links = 0
            tissue_links = 0
            for neighbor in self.g.G.neighbors(gid):
                nb = self.g.G.nodes[neighbor]
                ntype = nb.get("type")

                if ntype == "CELL_TYPE" and cell_links < _MAX_DISEASE_LINKS_PER_ENTITY:
                    self.g.add_edge(
                        src=neighbor, trgt=disease_nid,
                        attrs={
                            "rel": "DYSFUNCTION_IN",
                            "src_layer": "CELL",
                            "trgt_layer": "DISEASE",
                        },
                    )
                    _disease_edge_count += 1
                    cell_links += 1

                elif ntype == "TISSUE" and tissue_links < _MAX_DISEASE_LINKS_PER_ENTITY:
                    self.g.add_edge(
                        src=neighbor, trgt=disease_nid,
                        attrs={
                            "rel": "AFFECTED_IN",
                            "src_layer": "TISSUE",
                            "trgt_layer": "DISEASE",
                        },
                    )
                    _disease_edge_count += 1
                    tissue_links += 1

    print(f"    → {_disease_count} DISEASE nodes, {_disease_edge_count} disease edges")

    # ══════════════════════════════════════════════════════════════
    # PASS 2 — BIOELECTRIC STATE (from ELECTRICAL_COMPONENT)
    # GHK-derived: membrane potential, ion flux, frequency signature
    # ══════════════════════════════════════════════════════════════
    print("  P19.2: Bioelectric State Derivation (GHK model)")

    elec_nodes = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "ELECTRICAL_COMPONENT"
    ]

    _bstate_count = 0
    for elec_id, elec in elec_nodes:
        # MEMBRANE POTENTIAL: use v_half if measured, else resting default
        v_half = elec.get("v_half_activation")
        v_m = float(v_half) if v_half is not None else _RESTING_VM_MV

        # CONDUCTANCE
        cond_ps = elec.get("conductance_pS")
        cond = float(cond_ps) if cond_ps is not None else 0.0

        # ION FLUX per selective ion species
        ion_sel = elec.get("ion_selectivity") or []
        ion_flux_map: dict[str, float] = {}
        total_flux = 0.0
        for ion_raw in ion_sel:
            ion_key = str(ion_raw).strip().lower()
            weight = _ION_WEIGHTS.get(ion_key, 0.5)
            flux = cond * abs(v_m) * weight / 1000.0  # pA equivalent
            ion_flux_map[ion_raw] = round(flux, 4)
            total_flux += flux

        # FREQUENCY SIGNATURE: f ≈ total_ion_flux / (2π × membrane_area)
        freq_hz = total_flux / (_TWO_PI * _MEMBRANE_AREA_UM2) if total_flux > 0 else 0.0

        bstate_id = f"BSTATE_{elec_id}"
        self.g.add_node({
            "id": bstate_id,
            "type": "BIOELECTRIC_STATE",
            "label": f"BState_{elec.get('label', elec_id)}",
            "membrane_potential_mV": round(v_m, 2),
            "ion_flux": ion_flux_map,
            "total_ion_flux_pA": round(total_flux, 4),
            "conductance_pS": cond,
            "frequency_signature_hz": round(freq_hz, 4),
            "source_target_class": elec.get("target_class", "UNKNOWN"),
        })

        self.g.add_edge(
            src=elec_id, trgt=bstate_id,
            attrs={
                "rel": "MANIFESTS_AS",
                "src_layer": "BIOELECTRIC",
                "trgt_layer": "BIOELECTRIC_STATE",
            },
        )
        _bstate_count += 1

    print(f"    → {_bstate_count} BIOELECTRIC_STATE nodes")

    # ══════════════════════════════════════════════════════════════
    # PASS 3 — EM SIGNATURE (from BIOELECTRIC_STATE)
    # frequency spectrum, amplitude, phase, harmonics
    # ══════════════════════════════════════════════════════════════
    print("  P19.3: Electromagnetic Signature Generation")

    bstate_nodes = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "BIOELECTRIC_STATE"
    ]

    _emsig_count = 0
    for bs_id, bs in bstate_nodes:
        freq_hz = bs.get("frequency_signature_hz", 0.0)
        v_m = bs.get("membrane_potential_mV", _RESTING_VM_MV)
        ion_flux = bs.get("ion_flux", {})

        # AMPLITUDE: |V_m| × sum of ion permeability weights
        ion_weight_sum = sum(
            _ION_WEIGHTS.get(str(ion).strip().lower(), 0.5)
            for ion in ion_flux.keys()
        ) or 1.0
        amplitude_mv = abs(v_m) * ion_weight_sum

        # PHASE: deterministic from node hash
        rng = _seeded_rng(bs_id)
        phase_rad = rng.uniform(0, _TWO_PI)

        # HARMONICS: first 5 integer multiples of base frequency
        harmonics = [round(freq_hz * n, 4) for n in range(1, 6)] if freq_hz > 0 else []

        emsig_id = f"EMSIG_{bs_id}"
        self.g.add_node({
            "id": emsig_id,
            "type": "EM_SIGNATURE",
            "label": f"EMSig_{bs.get('label', bs_id)}",
            "frequency_spectrum_hz": round(freq_hz, 4),
            "amplitude_mv": round(amplitude_mv, 4),
            "phase_rad": round(phase_rad, 6),
            "harmonics": harmonics,
        })

        self.g.add_edge(
            src=bs_id, trgt=emsig_id,
            attrs={
                "rel": "GENERATES_SIGNATURE",
                "src_layer": "BIOELECTRIC_STATE",
                "trgt_layer": "EM_SIGNATURE",
            },
        )
        _emsig_count += 1

    print(f"    → {_emsig_count} EM_SIGNATURE nodes")

    # ══════════════════════════════════════════════════════════════
    # PASS 4 — MULTI-SCALE AGGREGATION
    # CELL_STATE → TISSUE_STATE → ORGAN_STATE
    # ══════════════════════════════════════════════════════════════
    print("  P19.4: Multi-Scale Aggregation (Cell → Tissue → Organ)")

    # 4A: CELL_STATE — aggregate BIOELECTRIC_STATEs reaching each CELL_TYPE
    #     Path: CELL_TYPE ←EXPRESSED_IN_CELL← GENE →ENCODED_BY← PROTEIN
    #           →DESCRIBED_AS_COMPONENT→ ELECTRICAL_COMPONENT →MANIFESTS_AS→ BSTATE
    cell_nodes = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "CELL_TYPE"
    ]

    _cstate_count = 0
    _cell_bstate_map: dict[str, list[str]] = {}  # cell_id → [bstate_ids]
    for cell_id, cell in cell_nodes:
        # TRAVERSE: CELL ← GENE → PROTEIN → ELEC → BSTATE
        bstate_ids: list[str] = []
        # MultiGraph is undirected → use neighbors() instead of predecessors()
        for gene_neighbor in self.g.G.neighbors(cell_id):
            if self.g.G.nodes[gene_neighbor].get("type") != "GENE":
                continue
            for prot_neighbor in self.g.G.neighbors(gene_neighbor):
                if self.g.G.nodes[prot_neighbor].get("type") != "PROTEIN":
                    continue
                for elec_neighbor in self.g.G.neighbors(prot_neighbor):
                    if self.g.G.nodes[elec_neighbor].get("type") != "ELECTRICAL_COMPONENT":
                        continue
                    for bs_neighbor in self.g.G.neighbors(elec_neighbor):
                        if self.g.G.nodes[bs_neighbor].get("type") == "BIOELECTRIC_STATE":
                            bstate_ids.append(bs_neighbor)

        if not bstate_ids:
            continue

        # AGGREGATE: weighted mean of bioelectric parameters
        vm_vals = []
        flux_vals = []
        freq_vals = []
        for bid in bstate_ids:
            bdata = self.g.G.nodes[bid]
            vm_vals.append(bdata.get("membrane_potential_mV", _RESTING_VM_MV))
            flux_vals.append(bdata.get("total_ion_flux_pA", 0.0))
            freq_vals.append(bdata.get("frequency_signature_hz", 0.0))

        n = len(bstate_ids)
        cstate_id = f"CSTATE_{cell_id}"
        self.g.add_node({
            "id": cstate_id,
            "type": "CELL_STATE",
            "label": f"CState_{cell.get('label', cell_id)}",
            "mean_membrane_potential_mV": round(sum(vm_vals) / n, 2),
            "total_ion_flux_pA": round(sum(flux_vals), 4),
            "mean_frequency_hz": round(sum(freq_vals) / n, 4),
            "bioelectric_channel_count": n,
        })

        self.g.add_edge(
            src=cell_id, trgt=cstate_id,
            attrs={"rel": "HAS_STATE", "src_layer": "CELL", "trgt_layer": "CELL_STATE"},
        )
        for bid in bstate_ids:
            self.g.add_edge(
                src=bid, trgt=cstate_id,
                attrs={"rel": "CONTRIBUTES_TO", "src_layer": "BIOELECTRIC_STATE", "trgt_layer": "CELL_STATE"},
            )

        _cell_bstate_map[cell_id] = bstate_ids
        _cstate_count += 1

    print(f"    → {_cstate_count} CELL_STATE nodes")

    # 4B: TISSUE_STATE — aggregate CELL_STATEs per TISSUE
    tissue_nodes = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "TISSUE"
    ]

    _tstate_count = 0
    for tissue_id, tissue in tissue_nodes:
        # FIND: CELL_STATEs linked to this tissue via cell types
        cstate_ids: list[str] = []
        ntpm_weights: list[float] = []
        for neighbor in self.g.G.neighbors(tissue_id):
            nb = self.g.G.nodes[neighbor]
            if nb.get("type") == "CELL_TYPE":
                candidate = f"CSTATE_{neighbor}"
                if self.g.G.has_node(candidate):
                    cstate_ids.append(candidate)
                    ntpm_weights.append(tissue.get("ntpm", 1.0))

        # Also check neighbors (CELL_TYPE_PART_OF_TISSUE etc.)
        for pred in self.g.G.neighbors(tissue_id):
            pb = self.g.G.nodes[pred]
            if pb.get("type") == "CELL_TYPE":
                candidate = f"CSTATE_{pred}"
                if self.g.G.has_node(candidate) and candidate not in cstate_ids:
                    cstate_ids.append(candidate)
                    ntpm_weights.append(tissue.get("ntpm", 1.0))

        if not cstate_ids:
            continue

        # WEIGHTED AGGREGATE
        w_total = sum(ntpm_weights) or 1.0
        vm_agg = 0.0
        freq_agg = 0.0
        flux_agg = 0.0
        for cid, w in zip(cstate_ids, ntpm_weights):
            cd = self.g.G.nodes[cid]
            vm_agg += cd.get("mean_membrane_potential_mV", _RESTING_VM_MV) * w
            freq_agg += cd.get("mean_frequency_hz", 0.0) * w
            flux_agg += cd.get("total_ion_flux_pA", 0.0) * w

        tstate_id = f"TSTATE_{tissue_id}"
        self.g.add_node({
            "id": tstate_id,
            "type": "TISSUE_STATE",
            "label": f"TState_{tissue.get('label', tissue_id)}",
            "mean_membrane_potential_mV": round(vm_agg / w_total, 2),
            "em_frequency_hz": round(freq_agg / w_total, 4),
            "total_flux_pA": round(flux_agg, 4),
            "ntpm_weight": round(w_total, 2),
            "cell_state_count": len(cstate_ids),
        })

        self.g.add_edge(
            src=tissue_id, trgt=tstate_id,
            attrs={"rel": "HAS_STATE", "src_layer": "TISSUE", "trgt_layer": "TISSUE_STATE"},
        )
        for cid in cstate_ids:
            self.g.add_edge(
                src=cid, trgt=tstate_id,
                attrs={"rel": "AGGREGATES_INTO", "src_layer": "CELL_STATE", "trgt_layer": "TISSUE_STATE"},
            )

        _tstate_count += 1

    print(f"    → {_tstate_count} TISSUE_STATE nodes")

    # 4C: ORGAN RESOLUTION via OLS4 UBERON + Ubergraph part_of
    #     Creates ORGAN nodes dynamically, then ORGAN_STATE aggregation.
    print("  P19.4c: Dynamic Organ Resolution (OLS4 UBERON + Ubergraph)")

    _organ_count = 0
    _organ_node_ids: dict[str, str] = {}  # uberon_id → organ_node_id

    for tissue_id, tissue in tissue_nodes:
        tissue_label = tissue.get("label", "")
        if not tissue_label:
            continue

        # STEP 1: resolve tissue label → UBERON organ via OLS4
        uberon_data = await _resolve_uberon_organ(tissue_label)
        if not uberon_data or not uberon_data.get("uberon_id"):
            continue

        tissue_uberon = uberon_data["uberon_id"]

        # STEP 2: Ubergraph → find parent organ this tissue is part_of
        parent = await _ubergraph_parent_organ(tissue_uberon)

        # USE parent organ if found, otherwise the tissue itself IS the organ
        organ_uberon = parent["uberon_id"] if parent else tissue_uberon
        organ_label = parent["label"] if parent else uberon_data["label"]
        organ_node_id = f"ORGAN_{organ_uberon.replace(':', '_')}"

        # CREATE ORGAN node if new
        if organ_uberon not in _organ_node_ids and _organ_count < _MAX_ORGAN_NODES:
            self.g.add_node({
                "id": organ_node_id,
                "type": "ORGAN",
                "label": organ_label,
                "uberon_id": organ_uberon,
            })
            _organ_node_ids[organ_uberon] = organ_node_id
            _organ_count += 1

        actual_organ_nid = _organ_node_ids.get(organ_uberon)
        if not actual_organ_nid:
            continue

        # EDGE: TISSUE → ORGAN (PART_OF)
        self.g.add_edge(
            src=tissue_id, trgt=actual_organ_nid,
            attrs={"rel": "PART_OF_ORGAN", "src_layer": "TISSUE", "trgt_layer": "ORGAN"},
        )

        # MAP: tissue → organ for ORGAN_STATE aggregation
        _organ_for_tissue[tissue_id] = actual_organ_nid

        # LINK ORGAN to existing CELL_TYPEs in this tissue
        for neighbor in self.g.G.neighbors(tissue_id):
            nb = self.g.G.nodes.get(neighbor, {})
            if nb.get("type") == "CELL_TYPE":
                self.g.add_edge(
                    src=neighbor, trgt=actual_organ_nid,
                    attrs={"rel": "LOCATED_IN", "src_layer": "CELL", "trgt_layer": "ORGAN"},
                )

        # LINK ORGAN to PROTEINs expressed in this tissue (via GENE)
        for neighbor in self.g.G.neighbors(tissue_id):
            nb = self.g.G.nodes.get(neighbor, {})
            if nb.get("type") == "GENE":
                for gpred in self.g.G.neighbors(neighbor):
                    if self.g.G.nodes.get(gpred, {}).get("type") == "PROTEIN":
                        self.g.add_edge(
                            src=gpred, trgt=actual_organ_nid,
                            attrs={"rel": "EXPRESSED_IN_ORGAN", "src_layer": "PROTEIN", "trgt_layer": "ORGAN"},
                        )

        # LINK ORGAN to chemical compounds (MINERAL, PHARMA_COMPOUND)
        for neighbor in self.g.G.neighbors(tissue_id):
            nb = self.g.G.nodes.get(neighbor, {})
            if nb.get("type") == "GENE":
                for gpred in self.g.G.neighbors(neighbor):
                    prot = self.g.G.nodes.get(gpred, {})
                    if prot.get("type") != "PROTEIN":
                        continue
                    for chem_n in self.g.G.neighbors(gpred):
                        chem_type = self.g.G.nodes.get(chem_n, {}).get("type")
                        if chem_type in (
                            "MINERAL", "PHARMA_COMPOUND", "ATOMIC_STRUCTURE",
                            "VITAMIN", "FATTY_ACID", "COFACTOR",
                        ):
                            self.g.add_edge(
                                src=chem_n, trgt=actual_organ_nid,
                                attrs={
                                    "rel": "ACTIVE_IN_ORGAN",
                                    "src_layer": chem_type,
                                    "trgt_layer": "ORGAN",
                                },
                            )

        # LINK ORGAN to DISEASEs already in graph
        for disease_nid in [
            n for n in self.g.G.neighbors(tissue_id)
            if self.g.G.nodes.get(n, {}).get("type") == "DISEASE"
        ]:
            self.g.add_edge(
                src=actual_organ_nid, trgt=disease_nid,
                attrs={"rel": "PATHOLOGY_IN", "src_layer": "ORGAN", "trgt_layer": "DISEASE"},
            )

    print(f"    → {_organ_count} ORGAN nodes (OLS4 UBERON), "
          f"{len(_organ_for_tissue)} tissue→organ links")

    # 4D: ORGAN_STATE — aggregate TISSUE_STATEs per resolved ORGAN
    organ_buckets: dict[str, list[str]] = {}  # organ_node_id → [tstate_ids]
    for tissue_id, organ_nid in _organ_for_tissue.items():
        tstate_candidate = f"TSTATE_{tissue_id}"
        if self.g.G.has_node(tstate_candidate):
            organ_buckets.setdefault(organ_nid, []).append(tstate_candidate)

    _ostate_count = 0
    for organ_nid, tstate_ids in organ_buckets.items():
        n = len(tstate_ids)
        vm_sum = 0.0
        freq_sum = 0.0
        amp_sum = 0.0
        for tid in tstate_ids:
            td = self.g.G.nodes[tid]
            vm_sum += td.get("mean_membrane_potential_mV", _RESTING_VM_MV)
            freq_sum += td.get("em_frequency_hz", 0.0)
            amp_sum += abs(td.get("mean_membrane_potential_mV", _RESTING_VM_MV))

        organ_data = self.g.G.nodes.get(organ_nid, {})
        organ_slug = organ_data.get("uberon_id", organ_nid).replace(":", "_").lower()

        ostate_id = f"OSTATE_{organ_slug}"
        self.g.add_node({
            "id": ostate_id,
            "type": "ORGAN_STATE",
            "label": f"OState_{organ_data.get('label', organ_nid)}",
            "organ_slug": organ_slug,
            "organ_uberon_id": organ_data.get("uberon_id", ""),
            "mean_frequency_hz": round(freq_sum / n, 4),
            "mean_amplitude_mv": round(amp_sum / n, 4),
            "mean_membrane_potential_mV": round(vm_sum / n, 2),
            "tissue_count": n,
        })

        # EDGE: ORGAN → ORGAN_STATE
        self.g.add_edge(
            src=organ_nid, trgt=ostate_id,
            attrs={"rel": "HAS_STATE", "src_layer": "ORGAN", "trgt_layer": "ORGAN_STATE"},
        )

        for tid in tstate_ids:
            self.g.add_edge(
                src=tid, trgt=ostate_id,
                attrs={"rel": "AGGREGATES_TISSUE", "src_layer": "TISSUE_STATE", "trgt_layer": "ORGAN_STATE"},
            )

        # LINK: EM_SIGNATURES → ORGAN_STATE (via BSTATE→EMSIG chain)
        for tid in tstate_ids:
            for pred in self.g.G.neighbors(tid):
                if self.g.G.nodes[pred].get("type") != "CELL_STATE":
                    continue
                for bpred in self.g.G.neighbors(pred):
                    if self.g.G.nodes[bpred].get("type") != "BIOELECTRIC_STATE":
                        continue
                    emsig_candidate = f"EMSIG_{bpred}"
                    if self.g.G.has_node(emsig_candidate):
                        self.g.add_edge(
                            src=emsig_candidate, trgt=ostate_id,
                            attrs={
                                "rel": "FEEDS_INTO",
                                "src_layer": "EM_SIGNATURE",
                                "trgt_layer": "ORGAN_STATE",
                            },
                        )

        _ostate_count += 1

    print(f"    → {_ostate_count} ORGAN_STATE nodes")

    # ══════════════════════════════════════════════════════════════
    # PASS 5 — SCAN SIGNAL (per organ × sensor type)
    # Simulated measurement with seeded Gaussian noise
    # ══════════════════════════════════════════════════════════════
    print("  P19.5: Scan Signal Simulation")

    organ_states = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "ORGAN_STATE"
    ]

    _scan_count = 0
    for ostate_id, ostate in organ_states:
        organ_slug = ostate.get("organ_slug", ostate_id)
        base_freq = ostate.get("mean_frequency_hz", 0.0)
        base_amp = ostate.get("mean_amplitude_mv", 0.0)

        for sensor in _SCAN_SENSORS:
            profile = _SENSOR_PROFILES[sensor]
            rng = _seeded_rng(f"{ostate_id}_{sensor}")

            # MEASURED SIGNAL = base + Gaussian noise
            noise_freq = rng.gauss(0, profile["noise_sigma"] * max(base_freq, 1.0))
            noise_amp = rng.gauss(0, profile["noise_sigma"] * max(base_amp, 1.0))
            measured_freq = max(0.0, base_freq + noise_freq)
            measured_amp = max(0.0, base_amp + noise_amp)
            measured_phase = rng.uniform(0, _TWO_PI)

            scan_id = f"SCAN_{organ_slug}_{sensor}"
            self.g.add_node({
                "id": scan_id,
                "type": "SCAN_SIGNAL",
                "label": f"Scan_{organ_slug}_{sensor}",
                "sensor_type": sensor,
                "measured_frequency_hz": round(measured_freq, 4),
                "measured_amplitude_mv": round(measured_amp, 4),
                "measured_phase_rad": round(measured_phase, 6),
                "noise_sigma": profile["noise_sigma"],
                "resolution_mm": profile["resolution_mm"],
                "source_organ": organ_slug,
            })

            self.g.add_edge(
                src=ostate_id, trgt=scan_id,
                attrs={
                    "rel": "PRODUCES_SIGNAL",
                    "src_layer": "ORGAN_STATE",
                    "trgt_layer": "SCAN_SIGNAL",
                },
            )
            _scan_count += 1

    print(f"    → {_scan_count} SCAN_SIGNAL nodes")

    # ══════════════════════════════════════════════════════════════
    # PASS 6 — INVERSE INFERENCE: SCAN_SIGNAL → DISEASE
    # Cosine similarity between scan feature vector and
    # disease-linked EM signature aggregates
    # ══════════════════════════════════════════════════════════════
    print("  P19.6: Inverse Inference (Signal → Disease)")

    # PRE-COMPUTE: per-disease aggregate EM feature vector
    # Path: DISEASE ← GENE ← PROTEIN → ELEC → BSTATE → EMSIG
    disease_em_profiles: dict[str, list[float]] = {}  # disease_node_id → [freq, amp, phase]
    disease_nodes = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "DISEASE"
    ]

    for dis_id, _ in disease_nodes:
        freq_acc = []
        amp_acc = []
        phase_acc = []

        # TRAVERSE back: DISEASE - GENE - PROTEIN - ELEC - BSTATE - EMSIG
        for pred in self.g.G.neighbors(dis_id):
            pred_type = self.g.G.nodes[pred].get("type")
            # genes and proteins linked to this disease
            gene_ids = []
            if pred_type == "GENE":
                gene_ids.append(pred)
            elif pred_type == "PROTEIN":
                for gn in self.g.G.neighbors(pred):
                    if self.g.G.nodes[gn].get("type") == "GENE":
                        gene_ids.append(gn)

            for gid in gene_ids:
                for gpred in self.g.G.neighbors(gid):
                    if self.g.G.nodes[gpred].get("type") != "PROTEIN":
                        continue
                    for elec_n in self.g.G.neighbors(gpred):
                        if self.g.G.nodes[elec_n].get("type") != "ELECTRICAL_COMPONENT":
                            continue
                        for bs_n in self.g.G.neighbors(elec_n):
                            if self.g.G.nodes[bs_n].get("type") != "BIOELECTRIC_STATE":
                                continue
                            emsig_cand = f"EMSIG_{bs_n}"
                            if self.g.G.has_node(emsig_cand):
                                em = self.g.G.nodes[emsig_cand]
                                freq_acc.append(em.get("frequency_spectrum_hz", 0.0))
                                amp_acc.append(em.get("amplitude_mv", 0.0))
                                phase_acc.append(em.get("phase_rad", 0.0))

        if freq_acc:
            n = len(freq_acc)
            disease_em_profiles[dis_id] = [
                sum(freq_acc) / n,
                sum(amp_acc) / n,
                sum(phase_acc) / n,
            ]

    # MATCH: each SCAN_SIGNAL against all disease profiles
    scan_nodes = [
        (k, v) for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "SCAN_SIGNAL"
    ]

    _inference_count = 0
    for scan_id, scan in scan_nodes:
        scan_vec = [
            scan.get("measured_frequency_hz", 0.0),
            scan.get("measured_amplitude_mv", 0.0),
            scan.get("measured_phase_rad", 0.0),
        ]

        if all(v == 0.0 for v in scan_vec):
            continue

        # RANK diseases by cosine similarity, keep top matches
        matches: list[tuple[str, float]] = []
        for dis_id, dis_vec in disease_em_profiles.items():
            sim = _cos_sim(scan_vec, dis_vec)
            if sim > 0.1:
                matches.append((dis_id, sim))

        matches.sort(key=lambda x: x[1], reverse=True)

        for dis_id, confidence in matches[:_MAX_DISEASE_LINKS_PER_ENTITY]:
            self.g.add_edge(
                src=scan_id, trgt=dis_id,
                attrs={
                    "rel": "INFERRED_DISEASE",
                    "confidence_score": round(confidence, 4),
                    "sensor_type": scan.get("sensor_type"),
                    "source_organ": scan.get("source_organ"),
                    "src_layer": "SCAN_SIGNAL",
                    "trgt_layer": "DISEASE",
                },
            )
            _inference_count += 1

    print(f"    → {_inference_count} INFERRED_DISEASE edges")
    print(f"  P19 complete: {_disease_count} diseases, {_bstate_count} bioelectric states, "
          f"{_emsig_count} EM signatures, {_cstate_count} cell states, "
          f"{_tstate_count} tissue states, {_ostate_count} organ states, "
          f"{_scan_count} scan signals, {_inference_count} inferences")

# ══════════════════════════════════════════════════════════════════
# PHASE 20 — 2D SCAN INGESTION
# ══════════════════════════════════════════════════════════════════

