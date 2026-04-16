"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``main``.

Prompt (user): data-dir graph hardening — TISSUE_2D_LAYER and CELL_POSITION use opaque ids.

Prompt (user): When ``UniprotKB.workflow_create_cell_type_nodes`` is False (default), do not create
    ``CELL_TYPE`` / dependent grid nodes in this step; tissue/anatomy passes still run.

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

async def enrich_tissue_expression_layer(self):
    """
    PHASE 10b: Tissue layer — HPA nTPM, UBERON (OLS4), Cell Ontology bridge, 2D layers, cell grid.

    Inputs: graph after organ seeding; uses ``ORGAN`` / ``TISSUE`` / ``CELL_TYPE`` patterns already present.
    Outputs: ``TISSUE`` nodes and expression edges; ``TISSUE_2D_LAYER`` as ``TSLAYER_*`` opaque ids linked
    via ``TISSUE_HAS_LAYER``; ``CELL_POSITION`` as ``CELLPOS_*`` with ``CELL_HAS_POSITION`` / ``CELL_IN_LAYER``.
    Side effects: HTTP to Protein Atlas and OLS4; deterministic RNG positions from MD5 seeds.
    Id policy: layer and grid nodes use ``data.graph_identity.canonical_node_id`` (no string embedding of
    multiple graph ids in the node key).
    """
    # CHAR: anatomy CL-composition +2D grid are optional — matches ``main`` policy.
    _allow_cell_type_nodes = getattr(self, "workflow_create_cell_type_nodes", False)

    # ── gien prompt: primärer Abschnitt Tissue-Layer (HPA + Uberon + CL-Brücke) ──
    HPA_SEARCH_DOWNLOAD = "https://www.proteinatlas.org/api/search_download.php"
    OLS4_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
    OLS4_ONTOLOGY_API = "https://www.ebi.ac.uk/ols4/api/ontologies"

    _MAX_TISSUE_HPA_GENES = 500
    _MIN_NTPM = 0.0  # Ausdruck nur wenn nTPM strikt größer (nach float-Parse)
    # Offizielle HPA search_download-Spalten: Kürzel + exakter JSON-Spaltenname (Data access table)
    _HPA_TISSUE_NTPM_FIELDS: list[tuple[str, str]] = [
        ("t_RNA_adipose_tissue", "Tissue RNA - adipose tissue [nTPM]"),
        ("t_RNA_adrenal_gland", "Tissue RNA - adrenal gland [nTPM]"),
        ("t_RNA_amygdala", "Tissue RNA - amygdala [nTPM]"),
        ("t_RNA_appendix", "Tissue RNA - appendix [nTPM]"),
        ("t_RNA_basal_ganglia", "Tissue RNA - basal ganglia [nTPM]"),
        ("t_RNA_blood_vessel", "Tissue RNA - blood vessel [nTPM]"),
        ("t_RNA_bone_marrow", "Tissue RNA - bone marrow [nTPM]"),
        ("t_RNA_breast", "Tissue RNA - breast [nTPM]"),
        ("t_RNA_cerebellum", "Tissue RNA - cerebellum [nTPM]"),
        ("t_RNA_cerebral_cortex", "Tissue RNA - cerebral cortex [nTPM]"),
        ("t_RNA_cervix", "Tissue RNA - cervix [nTPM]"),
        ("t_RNA_choroid_plexus", "Tissue RNA - choroid plexus [nTPM]"),
        ("t_RNA_colon", "Tissue RNA - colon [nTPM]"),
        ("t_RNA_duodenum", "Tissue RNA - duodenum [nTPM]"),
        ("t_RNA_endometrium_1", "Tissue RNA - endometrium 1 [nTPM]"),
        ("t_RNA_epididymis", "Tissue RNA - epididymis [nTPM]"),
        ("t_RNA_esophagus", "Tissue RNA - esophagus [nTPM]"),
        ("t_RNA_fallopian_tube", "Tissue RNA - fallopian tube [nTPM]"),
        ("t_RNA_gallbladder", "Tissue RNA - gallbladder [nTPM]"),
        ("t_RNA_heart_muscle", "Tissue RNA - heart muscle [nTPM]"),
        ("t_RNA_hippocampal_formation", "Tissue RNA - hippocampal formation [nTPM]"),
        ("t_RNA_hypothalamus", "Tissue RNA - hypothalamus [nTPM]"),
        ("t_RNA_kidney", "Tissue RNA - kidney [nTPM]"),
        ("t_RNA_liver", "Tissue RNA - liver [nTPM]"),
        ("t_RNA_lung", "Tissue RNA - lung [nTPM]"),
        ("t_RNA_lymph_node", "Tissue RNA - lymph node [nTPM]"),
        ("t_RNA_midbrain", "Tissue RNA - midbrain [nTPM]"),
        ("t_RNA_ovary", "Tissue RNA - ovary [nTPM]"),
        ("t_RNA_pancreas", "Tissue RNA - pancreas [nTPM]"),
        ("t_RNA_parathyroid_gland", "Tissue RNA - parathyroid gland [nTPM]"),
        ("t_RNA_pituitary_gland", "Tissue RNA - pituitary gland [nTPM]"),
        ("t_RNA_placenta", "Tissue RNA - placenta [nTPM]"),
        ("t_RNA_prostate", "Tissue RNA - prostate [nTPM]"),
        ("t_RNA_rectum", "Tissue RNA - rectum [nTPM]"),
        ("t_RNA_retina", "Tissue RNA - retina [nTPM]"),
        ("t_RNA_salivary_gland", "Tissue RNA - salivary gland [nTPM]"),
        ("t_RNA_seminal_vesicle", "Tissue RNA - seminal vesicle [nTPM]"),
        ("t_RNA_skeletal_muscle", "Tissue RNA - skeletal muscle [nTPM]"),
        ("t_RNA_skin_1", "Tissue RNA - skin 1 [nTPM]"),
        ("t_RNA_small_intestine", "Tissue RNA - small intestine [nTPM]"),
        ("t_RNA_smooth_muscle", "Tissue RNA - smooth muscle [nTPM]"),
        ("t_RNA_spinal_cord", "Tissue RNA - spinal cord [nTPM]"),
        ("t_RNA_spleen", "Tissue RNA - spleen [nTPM]"),
        ("t_RNA_stomach_1", "Tissue RNA - stomach 1 [nTPM]"),
        ("t_RNA_testis", "Tissue RNA - testis [nTPM]"),
        ("t_RNA_thymus", "Tissue RNA - thymus [nTPM]"),
        ("t_RNA_thyroid_gland", "Tissue RNA - thyroid gland [nTPM]"),
        ("t_RNA_tongue", "Tissue RNA - tongue [nTPM]"),
        ("t_RNA_tonsil", "Tissue RNA - tonsil [nTPM]"),
        ("t_RNA_urinary_bladder", "Tissue RNA - urinary bladder [nTPM]"),
        ("t_RNA_vagina", "Tissue RNA - vagina [nTPM]"),
    ]
    # Minimal: HPA-Fragment → bessere OLS-Uberon-Suche (HPA-Name vs. Uberon-Preferred-Label)
    _HPA_TISSUE_OLS_OVERRIDES: dict[str, str] = {
        "endometrium 1": "endometrium",
        "skin 1": "skin of body",
        "stomach 1": "stomach",
        "heart muscle": "cardiac muscle tissue",
    }

    def _hpa_col_to_ols_query(human_col: str) -> str:
        if not human_col.startswith("Tissue RNA - ") or not human_col.endswith(" [nTPM]"):
            return human_col
        inner = human_col[len("Tissue RNA - "): -len(" [nTPM]")].strip()
        return _HPA_TISSUE_OLS_OVERRIDES.get(inner, inner)

    def _parse_ntpm(raw) -> float | None:
        if raw is None:
            return None
        s = str(raw).strip().lower()
        if s in {"", "n/a", "na", "not detected"}:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _ols_href_https(href: str) -> str:
        if href.startswith("http://"):
            return "https://" + href[len("http://"):]
        return href

    # CHAR: wie fetch_with_retry — 429 von HPA/OLS abfangen, sonst bricht der Tissue-Pass stumm weg
    async def _get_retry(url: str, *, params=None, timeout: float = 20.0):
        while True:
            if params is not None:
                response = await self.client.get(url, params=params, timeout=timeout)
            else:
                response = await self.client.get(url, timeout=timeout)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                print(f"Tissue-layer rate limited ({retry_after}s)")
                await asyncio.sleep(retry_after)
                continue
            return response

    uberon_cache: dict[str, dict | None] = {}
    cl_uberon_ancestors_cache: dict[str, frozenset[str]] = {}

    async def _resolve_uberon(tissue_query: str) -> dict | None:
        if tissue_query in uberon_cache:
            return uberon_cache[tissue_query]
        try:
            res = await _get_retry(
                OLS4_SEARCH,
                params={"q": tissue_query, "ontology": "uberon", "exact": "true", "rows": 1},
                timeout=15.0,
            )
            docs = res.json().get("response", {}).get("docs", []) if res.status_code == 200 else []
            if not docs:
                res = await _get_retry(
                    OLS4_SEARCH,
                    params={"q": tissue_query, "ontology": "uberon", "rows": 1},
                    timeout=15.0,
                )
                docs = res.json().get("response", {}).get("docs", []) if res.status_code == 200 else []
            if not docs:
                uberon_cache[tissue_query] = None
                return None
            hit = docs[0]
            obo_id = hit.get("obo_id", "") or ""
            short_form = (obo_id.replace(":", "_") if obo_id else "") or hit.get("short_form", "")
            if not short_form or not short_form.startswith("UBERON_"):
                uberon_cache[tissue_query] = None
                return None
            desc_raw = hit.get("description", [])
            description = ". ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw or "")
            out = {
                "uberon_id": short_form,
                "label": hit.get("label", tissue_query),
                "description": description,
            }
            uberon_cache[tissue_query] = out
            return out
        except Exception as e:
            print(f"OLS4 uberon error for '{tissue_query}': {e}")
            uberon_cache[tissue_query] = None
            return None

    async def _cl_hierarchical_uberon_short_forms(cl_short: str) -> frozenset[str]:
        if cl_short in cl_uberon_ancestors_cache:
            return cl_uberon_ancestors_cache[cl_short]
        try:
            res = await _get_retry(
                f"{OLS4_ONTOLOGY_API}/cl/terms",
                params={"short_form": cl_short},
                timeout=20.0,
            )
            if res.status_code != 200:
                cl_uberon_ancestors_cache[cl_short] = frozenset()
                return frozenset()
            embedded = res.json().get("_embedded", {}).get("terms", [])
            if not embedded:
                cl_uberon_ancestors_cache[cl_short] = frozenset()
                return frozenset()
            anc_href = (
                embedded[0].get("_links", {}).get("hierarchicalAncestors", {}).get("href", "")
            )
            if not anc_href:
                cl_uberon_ancestors_cache[cl_short] = frozenset()
                return frozenset()
            anc_res = await _get_retry(_ols_href_https(anc_href), timeout=30.0)
            if anc_res.status_code != 200:
                cl_uberon_ancestors_cache[cl_short] = frozenset()
                return frozenset()
            terms = anc_res.json().get("_embedded", {}).get("terms", [])
            ubs = {t.get("short_form", "") for t in terms if str(t.get("short_form", "")).startswith("UBERON_")}
            ubs.discard("")
            frozen = frozenset(ubs)
            cl_uberon_ancestors_cache[cl_short] = frozen
            return frozen
        except Exception as e:
            print(f"OLS4 CL ancestors error for {cl_short}: {e}")
            cl_uberon_ancestors_cache[cl_short] = frozenset()
            return frozenset()

    _tissue_col_params = ",".join(x[0] for x in _HPA_TISSUE_NTPM_FIELDS)
    _hpa_columns = f"g,eg,up,rnats,rnatd,rnatsm,rnatss,{_tissue_col_params}"

    tissue_nodes_created = 0
    expr_edges = 0
    gene_nodes = [(k, v) for k, v in self.g.G.nodes(data=True) if v.get("type") == "GENE" and self._is_active(k)]
    genes_tried = 0

    for gene_id, gene in gene_nodes:
        if genes_tried >= _MAX_TISSUE_HPA_GENES:
            print(f"Tissue HPA cap reached ({_MAX_TISSUE_HPA_GENES} genes queried)")
            break
        gene_name = gene.get("label")
        if not gene_name:
            continue
        genes_tried += 1

        hpa_url = (
            f"{HPA_SEARCH_DOWNLOAD}?search={quote(gene_name)}&format=json"
            f"&columns={_hpa_columns}&compress=no"
        )
        try:
            res = await _get_retry(hpa_url, timeout=25.0)
            if res.status_code != 200:
                continue
            entries = res.json()
            if not isinstance(entries, list) or not entries:
                continue
            matched = next(
                (e for e in entries if (e.get("Gene") or "").upper() == gene_name.upper()),
                entries[0],
            )
            rnats = matched.get("RNA tissue specificity", "")
            rnatd = matched.get("RNA tissue distribution", "")
        except Exception as e:
            print(f"HPA tissue error for {gene_name}: {e}")
            continue

        for _spec, human_key in _HPA_TISSUE_NTPM_FIELDS:
            ntpm = _parse_ntpm(matched.get(human_key))
            if ntpm is None or ntpm <= _MIN_NTPM:
                continue
            ols_q = _hpa_col_to_ols_query(human_key)
            umeta = await _resolve_uberon(ols_q)
            if not umeta:
                continue
            tid = f"TISSUE_{umeta['uberon_id']}"
            if not self.g.G.has_node(tid):
                self.g.add_node({
                    "id": tid,
                    "type": "TISSUE",
                    "label": umeta["label"],
                    "uberon_id": umeta["uberon_id"],
                    "ontology_prefix": "UBERON",
                    "description": umeta.get("description", ""),
                    "uberon_resolved": True,
                })
                tissue_nodes_created += 1
            self.g.add_edge(
                src=gene_id,
                trgt=tid,
                attrs={
                    "rel": "EXPRESSED_IN_TISSUE",
                    "src_layer": "GENE",
                    "trgt_layer": "TISSUE",
                    "ntpm": ntpm,
                    "hpa_column": human_key,
                    "tissue_label_raw": ols_q,
                    "source": "HPA",
                    "rna_tissue_specificity": rnats,
                    "rna_tissue_distribution": rnatd,
                },
            )
            expr_edges += 1

    print(f"Tissue pass: {tissue_nodes_created} TISSUE nodes, {expr_edges} EXPRESSED_IN_TISSUE edges")

    # Brücke: CELL_TYPE → TISSUE wenn UBERON in hierarchischen Vorfahren von CL vorkommt
    bridge_edges = 0
    tissue_ids_in_graph = {
        n for n, d in self.g.G.nodes(data=True) if d.get("type") == "TISSUE"
    }
    if tissue_ids_in_graph:
        for cell_id, cell in self.g.G.nodes(data=True):
            if cell.get("type") != "CELL_TYPE" or not cell.get("cl_resolved"):
                continue
            cl_s = cell.get("cl_id")
            if not cl_s:
                continue
            ancestors = await _cl_hierarchical_uberon_short_forms(cl_s)
            for ub in ancestors:
                tid = f"TISSUE_{ub}"
                if tid not in tissue_ids_in_graph or not self.g.G.has_node(tid):
                    continue
                if self.g.G.has_edge(cell_id, tid):
                    continue
                self.g.add_edge(
                    src=cell_id,
                    trgt=tid,
                    attrs={
                        "rel": "PART_OF_TISSUE",
                        "src_layer": "CELL",
                        "trgt_layer": "TISSUE",
                        "source": "OLS4_CL_hierarchicalAncestors",
                    },
                )
                bridge_edges += 1
    print(f"Tissue bridge: {bridge_edges} PART_OF_TISSUE edges (CELL_TYPE → TISSUE)")

    # ══════════════════════════════════════════════════════════════
    # BLOCK A: ANATOMY_PART → TISSUE hierarchy (macro layer)
    # CHAR: OLS4 UBERON seed terms → crawl hierarchicalChildren to
    #       discover tissues/sub-organs dynamically, then link to
    #       existing TISSUE nodes + create new ones + crawl CL
    #       cell-type composition per tissue
    # ══════════════════════════════════════════════════════════════

    # CHAR: static fallback — used only when Stage A produced no ORGAN nodes
    _ANATOMY_SEEDS_FALLBACK: list[tuple[str, str]] = [
        ("UBERON_0001460", "arm"),
        ("UBERON_0000978", "leg"),
        ("UBERON_0000955", "brain"),
        ("UBERON_0000948", "heart"),
        ("UBERON_0002048", "lung"),
        ("UBERON_0002107", "liver"),
        ("UBERON_0002113", "kidney"),
        ("UBERON_0000160", "intestine"),
        ("UBERON_0000945", "stomach"),
        ("UBERON_0002097", "skin"),
        ("UBERON_0002240", "spinal cord"),
        ("UBERON_0002046", "thyroid gland"),
        ("UBERON_0000473", "testis"),
        ("UBERON_0000992", "ovary"),
        ("UBERON_0000995", "uterus"),
        ("UBERON_0002370", "thymus"),
        ("UBERON_0002106", "spleen"),
        ("UBERON_0000029", "lymph node"),
        ("UBERON_0002368", "endocrine gland"),
        ("UBERON_0000970", "eye"),
        ("UBERON_0000174", "excretory gland"),  # includes salivary
        ("UBERON_0001723", "tongue"),
        ("UBERON_0001264", "pancreas"),
        ("UBERON_0002369", "adrenal gland"),
    ]
    _MAX_CHILDREN_PER_SEED = 30
    _MAX_CL_PER_TISSUE = 10
    # CHAR: prefer ORGAN nodes from Stage A; fall back to static list if Stage A was skipped
    _seeds_to_use: list[tuple[str, str]] = [
        (nd.get("uberon_id"), nd.get("label", ""))
        for _, nd in self.g.G.nodes(data=True)
        if nd.get("type") == "ORGAN" and nd.get("uberon_id")
    ] or _ANATOMY_SEEDS_FALLBACK

    # ── nested helper: fetch OLS4 term by short_form ──
    async def _fetch_uberon_term(short_form: str) -> dict | None:
        """Resolve UBERON short_form → label + description via OLS4."""
        if short_form in uberon_cache:
            return uberon_cache[short_form]
        try:
            res = await _get_retry(
                f"{OLS4_ONTOLOGY_API}/uberon/terms",
                params={"short_form": short_form},
                timeout=15.0,
            )
            if res.status_code != 200:
                uberon_cache[short_form] = None
                return None
            terms = res.json().get("_embedded", {}).get("terms", [])
            if not terms:
                uberon_cache[short_form] = None
                return None
            t = terms[0]
            desc_raw = t.get("description", [])
            out = {
                "uberon_id": short_form,
                "label": t.get("label", short_form),
                "description": ". ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw or ""),
                "_links": t.get("_links", {}),
            }
            uberon_cache[short_form] = out
            return out
        except Exception as e:
            print(f"OLS4 term fetch error {short_form}: {e}")
            uberon_cache[short_form] = None
            return None

    # ── nested helper: get hierarchicalChildren of a UBERON term ──
    async def _fetch_children(term_meta: dict, max_children: int) -> list[dict]:
        """Fetch hierarchicalChildren from OLS4 _links, return list of child dicts."""
        href = term_meta.get("_links", {}).get("hierarchicalChildren", {}).get("href", "")
        if not href:
            return []
        try:
            res = await _get_retry(_ols_href_https(href), params={"size": max_children}, timeout=20.0)
            if res.status_code != 200:
                return []
            children = res.json().get("_embedded", {}).get("terms", [])
            result = []
            for ch in children:
                sf = ch.get("short_form", "")
                if not sf:
                    continue
                desc_raw = ch.get("description", [])
                result.append({
                    "uberon_id": sf,
                    "label": ch.get("label", sf),
                    "description": ". ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw or ""),
                    "is_uberon": sf.startswith("UBERON_"),
                    "is_cl": sf.startswith("CL_"),
                    "_links": ch.get("_links", {}),
                })
            return result
        except Exception as e:
            print(f"OLS4 children fetch error: {e}")
            return []

    # ── nested helper: find CL cell types related to a UBERON tissue ──
    async def _fetch_cl_for_tissue(uberon_short: str) -> list[dict]:
        """Search OLS4 CL terms whose annotations reference this UBERON term."""
        try:
            res = await _get_retry(
                OLS4_SEARCH,
                params={
                    "q": uberon_short.replace("_", ":"),
                    "ontology": "cl",
                    "rows": _MAX_CL_PER_TISSUE,
                    "fieldList": "short_form,label,description,obo_id",
                },
                timeout=15.0,
            )
            if res.status_code != 200:
                return []
            docs = res.json().get("response", {}).get("docs", [])
            out = []
            for d in docs:
                sf = (d.get("obo_id", "") or "").replace(":", "_")
                if not sf.startswith("CL_"):
                    continue
                desc_raw = d.get("description", [])
                out.append({
                    "cl_id": sf,
                    "label": d.get("label", sf),
                    "description": ". ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw or ""),
                })
            return out
        except Exception as e:
            print(f"OLS4 CL-for-tissue error {uberon_short}: {e}")
            return []

    # ── reverse index: uberon_id → graph TISSUE node id ──
    _uberon_to_tid: dict[str, str] = {}
    _tissue_label_to_id: dict[str, str] = {}
    for nid, nd in self.g.G.nodes(data=True):
        if nd.get("type") == "TISSUE":
            ub = nd.get("uberon_id", "")
            if ub:
                _uberon_to_tid[ub] = nid
            _tissue_label_to_id[nd.get("label", "").lower()] = nid

    anat_nodes = 0
    anat_edges = 0
    tissue_discovered = 0
    cl_composition_edges = 0
    tissue_disease_edges = 0
    neighbor_edges = 0

    # CHAR: reverse map uberon_id → ORGAN node id (from Stage A) for fast lookup
    _uberon_to_organ_nid: dict[str, str] = {
        nd.get("uberon_id"): nid
        for nid, nd in self.g.G.nodes(data=True)
        if nd.get("type") == "ORGAN" and nd.get("uberon_id")
    }

    for seed_uberon, seed_label in _seeds_to_use:
        # CHAR: if Stage A created an ORGAN node for this uberon, reuse it as the anatomy root;
        #       otherwise create a legacy ANATOMY_PART (fallback path)
        organ_nid = _uberon_to_organ_nid.get(seed_uberon)
        if organ_nid:
            anat_id = organ_nid
            term_meta = await _fetch_uberon_term(seed_uberon)
        else:
            anat_id = f"ANAT_{seed_label.replace(' ', '_')}"
            if not self.g.G.has_node(anat_id):
                term_meta = await _fetch_uberon_term(seed_uberon)
                self.g.add_node({
                    "id": anat_id,
                    "type": "ANATOMY_PART",
                    "label": term_meta["label"] if term_meta else seed_label,
                    "uberon_id": seed_uberon,
                    "description": (term_meta or {}).get("description", ""),
                    "source": "OLS4_UBERON",
                })
                anat_nodes += 1
            else:
                term_meta = await _fetch_uberon_term(seed_uberon)

        if not term_meta:
            continue

        # ── link seed directly to matching TISSUE if it exists ──
        direct_tid = _uberon_to_tid.get(seed_uberon) or _tissue_label_to_id.get(seed_label.lower())
        if direct_tid and not self.g.G.has_edge(anat_id, direct_tid):
            self.g.add_edge(
                src=anat_id, trgt=direct_tid,
                attrs={"rel": "ANATOMY_HAS_TISSUE", "src_layer": "ANATOMY",
                       "trgt_layer": "TISSUE", "source": "OLS4_UBERON_direct"},
            )
            anat_edges += 1

        # ── crawl hierarchicalChildren → tissues + sub-anatomy ──
        children = await _fetch_children(term_meta, _MAX_CHILDREN_PER_SEED)

        # CHAR: collect sibling tissue ids under this organ → NEIGHBOR_TISSUE edges below
        _sibling_tids: list[str] = []

        for child in children:
            child_ub = child["uberon_id"]
            child_label = child["label"]

            if child.get("is_uberon"):
                # CHAR: check if this child matches an existing TISSUE node
                existing_tid = _uberon_to_tid.get(child_ub) or _tissue_label_to_id.get(child_label.lower())

                if not existing_tid:
                    # CHAR: create new TISSUE node from OLS4 discovery
                    new_tid = f"TISSUE_{child_ub}"
                    if not self.g.G.has_node(new_tid):
                        self.g.add_node({
                            "id": new_tid,
                            "type": "TISSUE",
                            "label": child_label,
                            "uberon_id": child_ub,
                            "ontology_prefix": "UBERON",
                            "description": child.get("description", ""),
                            "uberon_resolved": True,
                            "source": "OLS4_anatomy_crawl",
                        })
                        tissue_discovered += 1
                        _uberon_to_tid[child_ub] = new_tid
                        _tissue_label_to_id[child_label.lower()] = new_tid
                    existing_tid = new_tid

                if not self.g.G.has_edge(anat_id, existing_tid):
                    self.g.add_edge(
                        src=anat_id, trgt=existing_tid,
                        attrs={"rel": "ANATOMY_HAS_TISSUE", "src_layer": "ANATOMY",
                               "trgt_layer": "TISSUE", "source": "OLS4_UBERON_child"},
                    )
                    anat_edges += 1
                _sibling_tids.append(existing_tid)

                # CHAR: fetch disease/harmful ontology (HPO + MONDO) for this tissue,
                #       referencing its parent organ — creates TISSUE_ASSOCIATED_DISEASE edges
                for _ont in ("hp", "mondo"):
                    try:
                        _d_res = await _get_retry(
                            OLS4_SEARCH,
                            params={"q": child_label, "ontology": _ont, "rows": 3, "type": "class"},
                            timeout=10.0,
                        )
                        if _d_res.status_code != 200:
                            continue
                        for _doc in _d_res.json().get("response", {}).get("docs", []):
                            _obo = _doc.get("obo_id", "")
                            if not _obo:
                                continue
                            _d_nid = f"DISEASE_{_obo.replace(':', '_')}"
                            if not self.g.G.has_node(_d_nid):
                                _dr = _doc.get("description", [])
                                self.g.add_node({
                                    "id": _d_nid,
                                    "type": "DISEASE",
                                    "label": _doc.get("label", _obo),
                                    "description": ". ".join(_dr) if isinstance(_dr, list) else str(_dr or ""),
                                    "obo_id": _obo,
                                    "ontology": _ont.upper(),
                                    "tissue_term": child_label,
                                    "parent_organ": seed_label,
                                })
                            if not self.g.G.has_edge(existing_tid, _d_nid):
                                self.g.add_edge(
                                    src=existing_tid, trgt=_d_nid,
                                    attrs={
                                        "rel": "TISSUE_ASSOCIATED_DISEASE",
                                        "src_layer": "TISSUE",
                                        "trgt_layer": "DISEASE",
                                        "ontology": _ont.upper(),
                                        "parent_organ_id": anat_id,
                                        "source": "OLS4",
                                    },
                                )
                                tissue_disease_edges += 1
                    except Exception:
                        pass

                if _allow_cell_type_nodes:
                    # ── crawl CL cell types for this tissue ──
                    cl_hits = await _fetch_cl_for_tissue(child_ub)
                    for cl_h in cl_hits:
                        cell_nid = f"CELL_{cl_h['cl_id']}"
                        if not self.g.G.has_node(cell_nid):
                            self.g.add_node({
                                "id": cell_nid,
                                "type": "CELL_TYPE",
                                "label": cl_h["label"],
                                "cl_id": cl_h["cl_id"],
                                "description": cl_h.get("description", ""),
                                "cl_resolved": True,
                                "source": "OLS4_anatomy_cl_crawl",
                            })
                        if not self.g.G.has_edge(cell_nid, existing_tid):
                            self.g.add_edge(
                                src=cell_nid, trgt=existing_tid,
                                attrs={"rel": "CELL_TYPE_PART_OF_TISSUE",
                                       "src_layer": "CELL", "trgt_layer": "TISSUE",
                                       "source": "OLS4_CL_tissue_crawl"},
                            )
                            cl_composition_edges += 1

            elif child.get("is_cl") and _allow_cell_type_nodes:
                # CHAR: direct CL child of anatomy part (rare but possible)
                cell_nid = f"CELL_{child_ub}"
                if not self.g.G.has_node(cell_nid):
                    self.g.add_node({
                        "id": cell_nid,
                        "type": "CELL_TYPE",
                        "label": child_label,
                        "cl_id": child_ub,
                        "description": child.get("description", ""),
                        "cl_resolved": True,
                        "source": "OLS4_anatomy_direct_cl",
                    })
                if direct_tid and not self.g.G.has_edge(cell_nid, direct_tid):
                    self.g.add_edge(
                        src=cell_nid, trgt=direct_tid,
                        attrs={"rel": "CELL_TYPE_PART_OF_TISSUE",
                               "src_layer": "CELL", "trgt_layer": "TISSUE",
                               "source": "OLS4_anatomy_direct_cl"},
                    )
                    cl_composition_edges += 1

        # CHAR: link all sibling tissues that share the same organ root
        for _i, _ta in enumerate(_sibling_tids):
            for _tb in _sibling_tids[_i + 1:]:
                if not self.g.G.has_edge(_ta, _tb):
                    self.g.add_edge(
                        src=_ta, trgt=_tb,
                        attrs={
                            "rel": "NEIGHBOR_TISSUE",
                            "src_layer": "TISSUE",
                            "trgt_layer": "TISSUE",
                            "shared_organ": anat_id,
                            "source": "OLS4_sibling",
                        },
                    )
                    neighbor_edges += 1

    print(f"Anatomy pass (OLS4): {anat_nodes} ANATOMY_PART, {anat_edges} ANATOMY_HAS_TISSUE, "
          f"{tissue_discovered} new TISSUE discovered, {cl_composition_edges} CELL_TYPE_PART_OF_TISSUE, "
          f"{tissue_disease_edges} TISSUE_ASSOCIATED_DISEASE, {neighbor_edges} NEIGHBOR_TISSUE")

    # ══════════════════════════════════════════════════════════════
    # BLOCK B: TISSUE_2D_LAYER nodes per TISSUE (meso layer)
    # CHAR: keyword-based layer classification — epithelial/outer
    #       on top (0), stromal (1), vascular/muscular (2), neural (3)
    # ══════════════════════════════════════════════════════════════
    _LAYER_RULES: list[tuple[int, str, set[str]]] = [
        (0, "epithelial",  {"epithelial", "mucosa", "endometrium", "skin", "retina", "cortex"}),
        (1, "stromal",     {"stromal", "connective", "bone marrow", "adipose", "cartilage"}),
        (2, "vascular",    {"vascular", "blood vessel", "cardiac", "smooth muscle", "skeletal muscle"}),
        (3, "neural",      {"neural", "spinal", "hippocampal", "amygdala", "hypothalamus",
                            "midbrain", "cerebellum", "choroid plexus", "basal ganglia"}),
    ]

    layer_nodes = 0
    layer_edges = 0
    # CHAR: tissue_id → list of layer_ids for quick lookup in Block C
    _tissue_layers: dict[str, list[str]] = {}

    for tid, td in list(self.g.G.nodes(data=True)):
        if td.get("type") != "TISSUE":
            continue
        t_label = (td.get("label") or "").lower()
        matched_any = False
        _tissue_layers[tid] = []
        for layer_idx, layer_type, keywords in _LAYER_RULES:
            if any(kw in t_label for kw in keywords):
                lid = canonical_node_id(
                    "TSLAYER",
                    {
                        "tissue_node_id": tid,
                        "layer_idx": layer_idx,
                        "layer_type": layer_type,
                    },
                )
                if not self.g.G.has_node(lid):
                    self.g.add_node({
                        "id": lid,
                        "type": "TISSUE_2D_LAYER",
                        "label": f"{td.get('label', '')} — {layer_type} (L{layer_idx})",
                        "layer_idx": layer_idx,
                        "layer_type": layer_type,
                        "source_tissue": tid,
                        "uberon_id": td.get("uberon_id"),
                    })
                    layer_nodes += 1
                if not self.g.G.has_edge(tid, lid):
                    self.g.add_edge(
                        src=tid, trgt=lid,
                        attrs={
                            "rel": "TISSUE_HAS_LAYER",
                            "src_layer": "TISSUE",
                            "trgt_layer": "TISSUE_2D_LAYER",
                        },
                    )
                    layer_edges += 1
                _tissue_layers[tid].append(lid)
                matched_any = True
        # CHAR: fallback — tissues without keyword match get a single generic layer
        if not matched_any:
            lid = canonical_node_id(
                "TSLAYER",
                {"tissue_node_id": tid, "layer_idx": 0, "layer_type": "generic"},
            )
            if not self.g.G.has_node(lid):
                self.g.add_node({
                    "id": lid,
                    "type": "TISSUE_2D_LAYER",
                    "label": f"{td.get('label', '')} — generic (L0)",
                    "layer_idx": 0,
                    "layer_type": "generic",
                    "source_tissue": tid,
                    "uberon_id": td.get("uberon_id"),
                })
                layer_nodes += 1
            if not self.g.G.has_edge(tid, lid):
                self.g.add_edge(
                    src=tid, trgt=lid,
                    attrs={
                        "rel": "TISSUE_HAS_LAYER",
                        "src_layer": "TISSUE",
                        "trgt_layer": "TISSUE_2D_LAYER",
                    },
                )
                layer_edges += 1
            _tissue_layers[tid] = [lid]
    print(f"Layer pass: {layer_nodes} TISSUE_2D_LAYER nodes, {layer_edges} TISSUE_HAS_LAYER edges")

    # ══════════════════════════════════════════════════════════════
    # BLOCK C: CELL_POSITION + CELL_IN_LAYER (micro / 2D grid)
    # CHAR: deterministic 2D mapping  (TISSUE, CELL_TYPE) → (x,y,layer)
    #       seed = md5(cell_id + tissue_id)  → reproducible positions
    # ══════════════════════════════════════════════════════════════
    GRID_SIZE = 100
    _MAX_POS_PER_CELL = 25

    # Pre-compute max nTPM per tissue for density normalisation
    _tissue_max_ntpm: dict[str, float] = {}
    for u, v, ed in self.g.G.edges(data=True):
        if ed.get("rel") == "EXPRESSED_IN_TISSUE":
            ntpm_val = ed.get("ntpm")
            if ntpm_val is not None:
                cur = _tissue_max_ntpm.get(v, 0.0)
                if ntpm_val > cur:
                    _tissue_max_ntpm[v] = ntpm_val

    # CHAR: classify cell label → preferred layer_idx for CELL_IN_LAYER assignment
    def _cell_layer_idx(cell_label: str) -> int:
        cl = cell_label.lower()
        for lidx, _, kws in _LAYER_RULES:
            if any(kw in cl for kw in kws):
                return lidx
        return 0  # default to outermost

    pos_nodes = 0
    cil_edges = 0
    chp_edges = 0

    for cell_id, cell_d in list(self.g.G.nodes(data=True)):
        if cell_d.get("type") != "CELL_TYPE":
            continue
        # Walk PART_OF_TISSUE edges from this cell
        for _, tgt, edata in list(self.g.G.edges(cell_id, data=True)):
            if edata.get("rel") != "PART_OF_TISSUE":
                continue
            tissue_id = tgt
            tissue_d = self.g.G.nodes.get(tissue_id, {})
            if tissue_d.get("type") != "TISSUE":
                continue

            # ── density from nTPM ──
            max_ntpm = _tissue_max_ntpm.get(tissue_id, 1.0)
            cell_ntpm = _tissue_max_ntpm.get(tissue_id, 0.0)
            density = (cell_ntpm / max_ntpm) if max_ntpm > 0 else 0.1
            density = max(density, 0.1)  # minimum representation

            n_pos = max(1, min(int(density * _MAX_POS_PER_CELL), _MAX_POS_PER_CELL))

            # ── deterministic seed ──
            seed = int(hashlib.md5((cell_id + tissue_id).encode()).hexdigest(), 16) % (2 ** 31)
            rng = random.Random(seed)

            for i in range(n_pos):
                x = rng.randint(0, GRID_SIZE - 1)
                y = rng.randint(0, GRID_SIZE - 1)
                cp_id = canonical_node_id(
                    "CELLPOS",
                    {"cell_id": cell_id, "tissue_id": tissue_id, "slot_index": i},
                )
                self.g.add_node({
                    "id": cp_id,
                    "type": "CELL_POSITION",
                    "x": x,
                    "y": y,
                    "layer_idx": _cell_layer_idx(cell_d.get("label", "")),
                    "source_tissue": tissue_id,
                    "tissue_uberon_id": tissue_d.get("uberon_id"),
                    "slot_index": i,
                })
                pos_nodes += 1
                self.g.add_edge(
                    src=cell_id, trgt=cp_id,
                    attrs={
                        "rel": "CELL_HAS_POSITION",
                        "src_layer": "CELL",
                        "trgt_layer": "CELL_POSITION",
                        "grid_size": GRID_SIZE,
                    },
                )
                chp_edges += 1

            # ── CELL_IN_LAYER edge (one per matching layer) ──
            preferred_lidx = _cell_layer_idx(cell_d.get("label", ""))
            for lid in _tissue_layers.get(tissue_id, []):
                layer_d = self.g.G.nodes.get(lid, {})
                if layer_d.get("layer_idx") == preferred_lidx:
                    if not self.g.G.has_edge(cell_id, lid):
                        self.g.add_edge(
                            src=cell_id, trgt=lid,
                            attrs={
                                "rel": "CELL_IN_LAYER",
                                "src_layer": "CELL",
                                "trgt_layer": "TISSUE_2D_LAYER",
                            },
                        )
                        cil_edges += 1

    print(f"2D Grid pass: {pos_nodes} CELL_POSITION nodes, {chp_edges} CELL_HAS_POSITION, {cil_edges} CELL_IN_LAYER edges")
