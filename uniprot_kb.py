"""
uniprot_kb — Config-Driven Hierarchical Biological Knowledge Graph Builder.

Prompt: Make Sure to First fetch Just Organ Data from Uniprot & co based on the input
    -> Within the Tissue Method use the Organ nodes to fetch Tissue and ontological
       (disease, harmful, description) entries Referenced to the overlying Organ/Body Part
    -> Follow the hierarchy Till you have mapped the Tissue (over Proteins, genes, molecules,
       Chemicals Up to good etc), include neighbor Tissue, to atomic structures and electron Matrix

Receives: dict[db:str, organs:list, function_annotation:list, outsrc_criteria:list]
    - db: "uniprot" | "pubchem" as starting point
    -> fetch organ-specific data (ORGAN nodes + disease ontology first)
    -> fetch function-specific data
    -> fetch harmful data

Hierarchical extraction workflow (top-down):
    Stage A: organ → ORGAN node (UBERON) + disease/harmful ontology (HPO/MONDO)
             → protein seeds (UniProt tissue query)
    Stage B: function → proteins & genes
    Stage C: outsrc_criteria → disease pre-seeding
    Tissue pass (Phase 10b):
        ORGAN nodes (Stage A) → TISSUE discovery (UBERON children, HPA nTPM)
        + disease/harmful ontology per TISSUE (referenced to parent ORGAN)
        + NEIGHBOR_TISSUE edges between sibling tissues under same organ
        + CELL_TYPE composition (Cell Ontology)
        + TISSUE_2D_LAYER / CELL_POSITION micro-grid
    Then enrichment phases 2–22c build the full interconnected graph:
        proteins & genes → functions → chemicals → atomic → electron_matrix
        → disease and harmful results

The entire graph is built stable from all relevant components and
interconnected for a production-ready system.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections import deque
import math
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import google.generativeai as genai
import httpx
import networkx as nx
import numpy as np
from firegraph.graph import GUtils

# ── ECO Evidence Ontology: differentiated reliability scoring ──────
# EXPERIMENTAL > HIGH_THROUGHPUT > CURATED > COMPUTATIONAL
ECO_RELIABILITY: dict[str, tuple[float, str]] = {
    # (reliability_score, evidence_type)
    "ECO:0000269": (1.0,  "EXPERIMENTAL"),   # experimental evidence – manual assertion
    "ECO:0007005": (1.0,  "EXPERIMENTAL"),   # immunofluorescence
    "ECO:0007001": (0.95, "EXPERIMENTAL"),   # immunoprecipitation
    "ECO:0000006": (0.7,  "HIGH_THROUGHPUT"),# high-throughput experimental
    "ECO:0006056": (0.7,  "HIGH_THROUGHPUT"),# high-throughput mass spec
    "ECO:0000305": (0.6,  "CURATED"),        # curator inference – manual assertion
    "ECO:0000313": (0.6,  "CURATED"),        # imported information – automatic assertion
    "ECO:0000250": (0.5,  "CURATED"),        # sequence similarity (ISS)
    "ECO:0000256": (0.3,  "COMPUTATIONAL"),  # match to sequence model
    "ECO:0000259": (0.3,  "COMPUTATIONAL"),  # match to InterPro member signature
    "ECO:0007669": (0.3,  "COMPUTATIONAL"),  # computational proteomics
    "ECO:0000501": (0.2,  "COMPUTATIONAL"),  # automatic assertion (IEA)
}
ECO_DEFAULT = (0.4, "UNKNOWN")

# ── GO short evidence codes -> ECO URIs (QuickGO returns these) ────
_GO_EV_TO_ECO: dict[str, str] = {
    "EXP": "ECO:0000269", "IDA": "ECO:0000314", "IPI": "ECO:0000353",
    "IMP": "ECO:0000315", "IGI": "ECO:0000316", "IEP": "ECO:0000270",
    "HTP": "ECO:0006056", "HDA": "ECO:0007005", "HMP": "ECO:0007001",
    "TAS": "ECO:0000304", "NAS": "ECO:0000303", "IC":  "ECO:0000305",
    "ISS": "ECO:0000250", "ISO": "ECO:0000266", "ISA": "ECO:0000247",
    "ISM": "ECO:0000255", "RCA": "ECO:0000245", "IEA": "ECO:0000501",
    "ND":  "ECO:0000307", "IBA": "ECO:0000318",
}

# ── Compound-phrase prefixes for multi-word nutrient labels ────────
COMPOUND_PHRASES = {"vitamin", "omega", "alpha", "beta", "gamma", "delta", "coenzyme", "co-enzym"}

# ── CELL TYPE INTEGRATION: memory guard ────────────────────────
_MAX_CELL_NODES = 500

# ── NON-CODING GENE DISCOVERY: memory guard ───────────────────
_MAX_NC_GENE_NODES = 1000
_NC_BIOTYPES = {"lncRNA", "miRNA", "snRNA", "snoRNA", "antisense", "lincRNA"}
_OVERLAP_FLANK_BP = 500_000  # ±500 kb around coding gene

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
}

# ── DICOM MODALITY TAG (0008,0060) → canonical modality string ───
_DICOM_MODALITY_MAP: dict[str, str] = {
    "MR": "MRI", "CT": "CT", "PT": "PET", "NM": "NM",
    "US": "ULTRASOUND", "CR": "XRAY", "DX": "XRAY",
    "MG": "MAMMOGRAPHY", "OT": "OTHER",
}

# ── MODALITY FEATURE VECTOR DIMENSION: [intensity, contrast, t1/hu/suv_proxy, variance, edge_density] ──
_SCAN_FEAT_DIM = 5

# ── MAX SPATIAL REGIONS per scan (memory guard) ──────────────────
_MAX_SPATIAL_REGIONS = 200


class ScanIngestionLayer:
    """FORMAT-AGNOSTIC 2D MEDICAL IMAGE LOADER.
    Autodetects DICOM / NIfTI / PNG+TIFF and extracts modality from file metadata.
    Returns a normalised dict ready for graph ingestion — no hardcoded anatomy data."""

    @staticmethod
    def load(path: str | Path) -> dict:
        """Load a 2D medical image and return a normalised descriptor.
        Raises ValueError when format or modality cannot be determined."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Scan file not found: {p}")

        suffix = p.suffix.lower()

        # ── DICOM ────────────────────────────────────────────────
        if suffix in (".dcm", ".dicom", ""):
            return ScanIngestionLayer._load_dicom(p)

        # ── NIfTI ────────────────────────────────────────────────
        if suffix in (".nii", ".gz"):
            return ScanIngestionLayer._load_nifti(p)

        # ── RASTER (PNG / TIFF / JPG) ────────────────────────────
        if suffix in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
            return ScanIngestionLayer._load_raster(p)

        raise ValueError(f"Unsupported scan format: {suffix}")

    # ── PRIVATE LOADERS ──────────────────────────────────────────

    @staticmethod
    def _load_dicom(p: Path) -> dict:
        import pydicom
        ds = pydicom.dcmread(str(p))
        pixels = ds.pixel_array.astype(np.float64)
        # MODALITY from DICOM tag (0008,0060)
        raw_mod = str(getattr(ds, "Modality", "")).upper().strip()
        modality = _DICOM_MODALITY_MAP.get(raw_mod)
        if not modality:
            raise ValueError(f"Unknown DICOM modality tag: '{raw_mod}' in {p}")
        spacing = [float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0])]
        return {
            "pixels": pixels,
            "modality": modality,
            "slice_idx": int(getattr(ds, "InstanceNumber", 0)),
            "spacing_mm": spacing,
            "orientation": str(getattr(ds, "ImageOrientationPatient", "UNKNOWN")),
            "source_path": str(p),
        }

    @staticmethod
    def _load_nifti(p: Path) -> dict:
        import nibabel as nib
        img = nib.load(str(p))
        data = np.asarray(img.dataobj, dtype=np.float64)
        # NIfTI: take middle slice if 3D volume
        if data.ndim == 3:
            slice_idx = data.shape[2] // 2
            pixels = data[:, :, slice_idx]
        elif data.ndim == 2:
            slice_idx = 0
            pixels = data
        else:
            raise ValueError(f"NIfTI has unexpected {data.ndim}D shape in {p}")
        header = img.header
        zooms = header.get_zooms() if hasattr(header, "get_zooms") else (1.0, 1.0)
        spacing = [float(zooms[0]), float(zooms[1])]
        # NIfTI: modality must be inferred from description or filename
        desc = str(getattr(header, "descrip", b"")).lower()
        modality = ScanIngestionLayer._infer_modality_from_text(desc, p.stem)
        return {
            "pixels": pixels,
            "modality": modality,
            "slice_idx": slice_idx,
            "spacing_mm": spacing,
            "orientation": "RAS",
            "source_path": str(p),
        }

    @staticmethod
    def _load_raster(p: Path) -> dict:
        from PIL import Image
        img = Image.open(p).convert("L")
        pixels = np.array(img, dtype=np.float64)
        modality = ScanIngestionLayer._infer_modality_from_text("", p.stem)
        return {
            "pixels": pixels,
            "modality": modality,
            "slice_idx": 0,
            "spacing_mm": [1.0, 1.0],
            "orientation": "PLANAR",
            "source_path": str(p),
        }

    @staticmethod
    def _infer_modality_from_text(desc: str, filename: str) -> str:
        """KEYWORD MATCH against description + filename — no hardcoded fallback."""
        combined = f"{desc} {filename}".lower()
        # ORDER MATTERS: more specific first
        for kw, mod in (
            ("fmri", "FMRI"), ("bold", "FMRI"),
            ("t1", "MRI_T1"), ("t2", "MRI_T2"),
            ("mri", "MRI"), ("mr", "MRI"),
            ("ct", "CT"), ("hounsfield", "CT"),
            ("pet", "PET"), ("suv", "PET"),
            ("ultrasound", "ULTRASOUND"), ("us_", "ULTRASOUND"),
        ):
            if kw in combined:
                return mod
        raise ValueError(
            f"Cannot infer modality from description='{desc}', filename='{filename}'. "
            "Provide modality_hint explicitly."
        )


class UniprotKB:
    # ── CONTEXT-DRIVEN EXPANSION: allowed edge relations for BFS ──
    # INTERACTS_WITH: PPI edges from UniProt cc_interaction let BFS traverse interaction partners
    # ASSOCIATED_WITH_DISEASE: disease links from UniProt cc_disease anchor context to pathology
    _ALLOWED_EXPAND_RELS = frozenset({
        "ENCODED_BY", "MODULATES_TARGET", "HAS_STRUCTURE",
        "PARTICIPATES_IN", "REQUIRES_MINERAL", "CONTAINS_NUTRIENT",
        "INTERACTS_WITH", "ASSOCIATED_WITH_DISEASE",
    })
    # CHAR: node types kept when extracting organ→tissue→molecular→atomic/electron view
    _ORGAN_TISSUE_CASCADE_TYPES = frozenset({
        "ORGAN", "ANATOMY_PART", "TISSUE", "TISSUE_2D_LAYER", "CELL_POSITION", "CELL_TYPE",
        "DISEASE", "GENE", "PROTEIN", "NON_CODING_GENE",
        "SEQUENCE_HASH", "3D_STRUCTURE", "PROTEIN_DOMAIN",
        "PHARMA_COMPOUND", "MINERAL", "MOLECULE_CHAIN", "VMH_METABOLITE", "FOOD_SOURCE",
        "ATOMIC_STRUCTURE", "EXCITATION_FREQUENCY", "MICROBIAL_STRAIN",
        "COMPARTMENT", "CELLULAR_COMPONENT", "GO_TERM", "REACTOME_PATHWAY",
        "ECO_EVIDENCE", "ALLERGEN", "IMMUNE_RESPONSE",
    })

    def __init__(self, g):
        self.g = g
        self.client = httpx.AsyncClient(headers=_BROWSER_HEADERS, timeout=60.0)
        # None → all nodes active (legacy behaviour); set → context-driven subgraph
        self.active_subgraph: set[str] | None = None
        # WORKFLOW CONFIG — set by finalize_biological_graph(cfg=...)
        self.workflow_cfg: dict | None = None
        # ORGAN seeds resolved in Stage A; consumed by enrich_tissue_expression_layer Block A
        self._organ_uberon_seeds: list[tuple[str, str]] = []

    # --- ASYNC HELPER ---
    async def fetch_with_retry(self, url, max_retries: int = 5):
        for attempt in range(max_retries):
            response = await self.client.get(url)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 3))
                print(f"Rate limited. Waiting {retry_after}s for {url}")
                await asyncio.sleep(retry_after)
                continue
            response.raise_for_status()
            return response.json()
        print(f"Max retries ({max_retries}) exceeded for {url}")
        return None

    async def close(self):
        if self.client is None:
            return
        await self.client.aclose()
        self.client = None

    def _is_active(self, node_id: str) -> bool:
        """Guard: True when no subgraph filter is set OR node belongs to active set."""
        return self.active_subgraph is None or node_id in self.active_subgraph

    # --- CORE INGESTION ---
    async def get_all_proteins(self):
        """Initialer Fetch des menschlichen Proteoms."""
        url = "https://rest.uniprot.org/uniprotkb/search?query=proteome:UP000005640"
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()

            graph_nodes = []
            graph_edges = []
            for entry in data.get("results", []):
                protein_id = entry.get("primaryAccession")
                protein_node = {
                    "id": protein_id,
                    "type": "PROTEIN",
                    "label": entry.get("uniProtkbId"),
                    "description": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName",
                                                                                                      {}).get("value"),
                    "taxonId": entry.get("organism", {}).get("taxonId")
                }
                graph_nodes.append(protein_node)
                self._gather_genes(entry, protein_id, graph_nodes, graph_edges)
                self.apply_eco_weighting(entry, protein_id)  # Evidence direkt beim Ingest

            for node in graph_nodes: self.g.add_node(node)
            for edge in graph_edges: self.g.add_edge(**edge)
        except Exception as e:
            print(f"Error in Protein Ingestion: {e}")

    def _gather_genes(self, entry, protein_id, nodes, edges):
        for gene in entry.get("genes", []):
            gene_val = gene.get("geneName", {}).get("value")
            if not gene_val: continue
            gene_node_id = f"GENE_{gene_val}"
            nodes.append({"id": gene_node_id, "type": "GENE", "label": gene_val})
            edges.append({"src": protein_id, "trgt": gene_node_id,
                          "attrs": {"rel": "ENCODED_BY", "src_layer": "PROTEIN", "trgt_layer": "GENE"}})

    # ══════════════════════════════════════════════════════════════════
    # CONTEXT-DRIVEN GRAPH PIPELINE  (seed → expand → enrich)
    # ══════════════════════════════════════════════════════════════════

    async def fetch_proteins_by_query(self, query: str, limit: int = 100) -> list[str]:
        """TARGETED PROTEIN FETCH — replaces global proteome ingestion for context builds.
        Returns list of ingested protein accession IDs."""
        url = (
            f"https://rest.uniprot.org/uniprotkb/search"
            f"?query={quote(query)}&format=json&size={min(limit, 500)}"
        )
        ingested: list[str] = []
        try:
            data = await self.fetch_with_retry(url)
            graph_nodes: list[dict] = []
            graph_edges: list[dict] = []

            for entry in data.get("results", []):
                protein_id = entry.get("primaryAccession")
                if not protein_id:
                    continue
                graph_nodes.append({
                    "id": protein_id,
                    "type": "PROTEIN",
                    "label": entry.get("uniProtkbId"),
                    "description": entry.get("proteinDescription", {})
                                        .get("recommendedName", {})
                                        .get("fullName", {})
                                        .get("value"),
                    "taxonId": entry.get("organism", {}).get("taxonId"),
                })
                self._gather_genes(entry, protein_id, graph_nodes, graph_edges)
                self.apply_eco_weighting(entry, protein_id)
                ingested.append(protein_id)

            for node in graph_nodes:
                self.g.add_node(node)
            for edge in graph_edges:
                self.g.add_edge(**edge)

            print(f"Context fetch: {len(ingested)} proteins ingested for query [{query[:80]}]")
        except Exception as e:
            print(f"Error in context protein fetch: {e}")
        return ingested

    async def resolve_seed_nodes(
        self,
        organs: list[str] | None = None,
        functions: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> list[str]:
        """THREE-PATH SEED RESOLUTION — organs, functions, keywords → protein accessions.
        A) Organ → UniProt tissue query
        B) Functions → QuickGO molecular_function + UniProt keyword fallback
        C) Keywords → label match on existing graph nodes
        """
        seed_ids: set[str] = set()
        fetch_tasks: list = []

        # ── A: ORGAN → PROTEINS (UniProt tissue filter) ──────────────
        for organ in (organs or []):
            q = f"(organism_id:9606) AND (tissue:{organ})"
            fetch_tasks.append(self.fetch_proteins_by_query(q))

        # ── B: FUNCTIONS → PROTEINS (QuickGO + UniProt keyword) ──────
        for fn in (functions or []):
            # QuickGO: molecular_function annotations matching free-text
            fetch_tasks.append(self._resolve_function_seeds(fn))
            # UniProt keyword fallback
            q = f"(organism_id:9606) AND (keyword:{fn})"
            fetch_tasks.append(self.fetch_proteins_by_query(q))

        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, list):
                seed_ids.update(res)

        # ── C: KEYWORDS → label match on existing graph nodes ────────
        for kw in (keywords or []):
            kw_lower = kw.lower()
            for nid, attrs in self.g.G.nodes(data=True):
                if kw_lower in (attrs.get("label") or "").lower():
                    seed_ids.add(nid)

        print(f"Seed resolution complete: {len(seed_ids)} unique seeds")
        return list(seed_ids)

    async def _resolve_function_seeds(self, function_term: str) -> list[str]:
        """QuickGO aspect=molecular_function search → protein accessions."""
        _QGO = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
        accessions: list[str] = []
        try:
            res = await self.client.get(
                _QGO,
                params={
                    "goUsage": "exact",
                    "aspect": "molecular_function",
                    "taxonId": 9606,
                    "query": function_term,
                    "limit": 100,
                },
                headers={"Accept": "application/json"},
                timeout=25.0,
            )
            if res.status_code != 200:
                return accessions
            for anno in res.json().get("results", []):
                gp = anno.get("geneProductId", "")
                # FORMAT: UniProtKB:P12345 → P12345
                if gp.startswith("UniProtKB:"):
                    acc = gp.split(":", 1)[1]
                    if acc not in {n for n in self.g.G.nodes}:
                        accessions.append(acc)
            # BATCH INGEST discovered accessions
            if accessions:
                q = " OR ".join(f"accession:{a}" for a in accessions[:300])
                return await self.fetch_proteins_by_query(f"({q})")
        except Exception as e:
            print(f"QuickGO function resolve error ({function_term}): {e}")
        return accessions

    def expand_graph(self, seeds: list[str], max_depth: int = 1) -> set[str]:
        """BFS EXPANSION — follows only _ALLOWED_EXPAND_RELS edges up to max_depth hops.
        Returns all newly discovered node IDs (excludes seeds already known)."""
        visited: set[str] = set(seeds)
        frontier: set[str] = set(seeds)

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid not in self.g.G:
                    continue
                for neighbor in self.g.G.neighbors(nid):
                    if neighbor in visited:
                        continue
                    # CHECK: at least one edge between nid↔neighbor has an allowed rel
                    edge_data = self.g.G.get_edge_data(nid, neighbor)
                    if edge_data is None:
                        continue
                    # MULTIGRAPH: edge_data is dict[key, attrs]
                    edges = edge_data.values() if isinstance(edge_data, dict) else [edge_data]
                    for eattr in edges:
                        if not isinstance(eattr, dict):
                            continue
                        rel = (eattr.get("rel") or "").upper()
                        if rel in self._ALLOWED_EXPAND_RELS:
                            next_frontier.add(neighbor)
                            break

            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        return visited - set(seeds)

    def is_relevant(self, node_id: str, node_attrs: dict, context: dict) -> bool:
        """PER-TYPE RELEVANCE FILTER — decides if a node belongs in the context graph.
        context keys: organs (set[str]), functions (set[str]), keywords (set[str])."""
        ntype = (node_attrs.get("type") or "").upper()
        label = (node_attrs.get("label") or "").lower()
        organs_lc = {o.lower() for o in context.get("organs", [])}
        funcs_lc = {f.lower() for f in context.get("functions", [])}
        kws_lc = {k.lower() for k in context.get("keywords", [])}
        all_terms = organs_lc | funcs_lc | kws_lc

        if ntype == "PROTEIN":
            # MUST match at least one context term in label, description, or tissue
            desc = (node_attrs.get("description") or "").lower()
            tissue = (node_attrs.get("tissue") or "").lower()
            return any(t in label or t in desc or t in tissue for t in all_terms) or node_id in (self.active_subgraph or set())

        if ntype == "GENE":
            # ONLY if connected to a protein in active_subgraph
            if self.active_subgraph is None:
                return True
            return any(nb in self.active_subgraph for nb in self.g.G.neighbors(node_id))

        if ntype == "PHARMA_COMPOUND":
            # ONLY if modulates a protein in active_subgraph
            if self.active_subgraph is None:
                return True
            return any(nb in self.active_subgraph for nb in self.g.G.neighbors(node_id))

        if ntype == "FOOD_SOURCE":
            # ONLY if contains nutrient linked to active nodes
            if self.active_subgraph is None:
                return True
            return any(nb in self.active_subgraph for nb in self.g.G.neighbors(node_id))

        # ALL OTHER TYPES: pass through
        return True

    @staticmethod
    def score_node(node_attrs: dict) -> float:
        """RELEVANCE SCORE — simple additive metric for node ranking."""
        return float(node_attrs.get("relevance", 0)) + float(node_attrs.get("evidence_score", 0))

    async def build_context_graph(
        self,
        organs: list[str] | None = None,
        functions: list[str] | None = None,
        keywords: list[str] | None = None,
        max_depth: int = 4,
    ):
        """CONTEXT-DRIVEN GRAPH CONSTRUCTION — seed → expand → enrich pipeline.
        Builds a minimal, high-signal graph tailored to organ/function/keyword context.
        Does NOT fetch full proteome. Does NOT enrich disconnected nodes.
        CALLER is responsible for calling close() after graph export/visualization."""
        print("═══ CONTEXT GRAPH: Seed Resolution ═══")
        seeds = await self.resolve_seed_nodes(organs, functions, keywords)
        if not seeds:
            print("WARNING: No seeds resolved — graph will be empty")
            return

        self.active_subgraph = set(seeds)

        # ── BFS EXPANSION (multi-hop traversal) ──────────────────
        print(f"═══ CONTEXT GRAPH: Expanding from {len(seeds)} seeds (depth={max_depth}) ═══")
        for depth in range(max_depth):
            new_nodes = self.expand_graph(list(self.active_subgraph), 1)
            if not new_nodes:
                print(f"  Expansion saturated at depth {depth + 1}")
                break
            self.active_subgraph.update(new_nodes)
            print(f"  Depth {depth + 1}: +{len(new_nodes)} nodes (total: {len(self.active_subgraph)})")

        # ── CONDITIONAL ENRICHMENT (only active subgraph) ────────
        print(f"═══ CONTEXT GRAPH: Enriching {len(self.active_subgraph)} active nodes ═══")

        print("  Phase 1: Gene deep enrichment")
        await self.enrich_gene_nodes_deep()

        print("  Phase 2: Functional dynamics (Reactome)")
        await self.enrich_functional_dynamics()

        print("  Phase 3: Pharmacology (ChEMBL + PubChem)")
        await self.enrich_pharmacology_quantum_adme()

        print("  Phase 4: Molecular structures (PubChem/ChEBI)")
        await self.enrich_molecular_structures()

        print("  Phase 5: Genomic + Food + Bioelectric")
        await asyncio.gather(
            self.enrich_genomic_data(),
            self.enrich_food_sources(),
            self.enrich_bioelectric_properties(),
        )

        print("  Phase 6: Structural + GO + Compartments")
        await asyncio.gather(
            self.enrich_structural_layer(),
            self.enrich_go_semantic_layer(),
            self.enrich_compartment_localization(),
        )

        # ── POST-EXPANSION: add newly enriched nodes to subgraph ─
        for nid in list(self.g.G.nodes):
            if any(nb in self.active_subgraph for nb in self.g.G.neighbors(nid)):
                self.active_subgraph.add(nid)

        # ── RELEVANCE PRUNE: remove nodes that fail context filter ─
        context = {
            "organs": organs or [],
            "functions": functions or [],
            "keywords": keywords or [],
        }
        to_remove = [
            nid for nid, attrs in self.g.G.nodes(data=True)
            if not self.is_relevant(nid, attrs, context)
        ]
        for nid in to_remove:
            self.g.G.remove_node(nid)
            self.active_subgraph.discard(nid)

        print(f"═══ CONTEXT GRAPH COMPLETE: {self.g.G.number_of_nodes()} nodes, "
              f"{self.g.G.number_of_edges()} edges ═══")
        self.g.print_status_G()

    # --- ENRICHMENT CATEGORIES ---
    async def enrich_gene_nodes_deep(self):
        """Fetches cofactors + Reactome pathways per PROTEIN and materializes them as edges.
        No reference data stored on nodes — cofactors become MINERAL nodes, pathways become MOLECULE_CHAIN nodes."""
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN" and self._is_active(k)]
        tasks = [self.fetch_with_retry(self.get_uniprot_url_single_gene(node_id))
                 for node_id, _ in protein_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (node_id, _node), res in zip(protein_nodes, results):
            if isinstance(res, Exception):
                continue

            # COFACTORS → MINERAL nodes + REQUIRES_MINERAL edges
            for comment in res.get("comments", []):
                if comment.get("commentType") != "COFACTOR":
                    continue
                for cofactor in comment.get("cofactors", []):
                    name = cofactor.get("name")
                    if not name:
                        continue
                    m_id = f"MINERAL_{name}"
                    self.g.add_node({"id": m_id, "type": "MINERAL", "label": name})
                    self.g.add_edge(src=node_id, trgt=m_id, attrs={
                        "rel": "REQUIRES_MINERAL", "src_layer": "PROTEIN", "trgt_layer": "MINERAL",
                    })

            # REACTOME XREFS → MOLECULE_CHAIN nodes + INVOLVED_IN_CHAIN edges
            for ref in res.get("uniProtKBCrossReferences", []):
                if ref.get("database") != "Reactome":
                    continue
                pw_id = f"MOL_{ref['id']}"
                self.g.add_node({"id": pw_id, "type": "MOLECULE_CHAIN",
                                 "label": ref.get("properties", {}).get("pathwayName", "Pathway"),
                                 # store the stable Reactome ID so _bridge_reactome_nodes can match
                                 "reactome_stable_id": ref["id"]})
                self.g.add_edge(src=node_id, trgt=pw_id, attrs={
                    "rel": "INVOLVED_IN_CHAIN", "src_layer": "PROTEIN", "trgt_layer": "MOLECULE_CHAIN",
                })

            # PROTEIN–PROTEIN INTERACTIONS (UniProt cc_interaction / IntAct)
            # Only wire edge when the partner accession is ALREADY a node in the graph,
            # keeping the PPI layer anchored to the active biological context.
            for comment in res.get("comments", []):
                if comment.get("commentType") != "INTERACTION":
                    continue
                for iact in comment.get("interactions", []):
                    # UniProt encodes both sides; the *other* side relative to node_id is the partner
                    acc_one = iact.get("interactantOne", {}).get("uniProtKBAccession", "")
                    acc_two = iact.get("interactantTwo", {}).get("uniProtKBAccession", "")
                    partner = acc_two if acc_one == node_id else acc_one
                    if not partner or partner == node_id:
                        continue
                    # If partner not yet in graph, add a minimal PROTEIN stub so the edge is reachable
                    if not self.g.G.has_node(partner):
                        gene_lbl = (iact.get("interactantTwo", {}).get("geneName")
                                    if acc_one == node_id
                                    else iact.get("interactantOne", {}).get("geneName")) or partner
                        self.g.add_node({"id": partner, "type": "PROTEIN", "label": gene_lbl, "stub": True})
                    self.g.add_edge(src=node_id, trgt=partner, attrs={
                        "rel": "INTERACTS_WITH",
                        "experiments": iact.get("numberOfExperiments", 0),
                        "src_layer": "PROTEIN", "trgt_layer": "PROTEIN",
                    })

            # DISEASE ANNOTATIONS (UniProt cc_disease / MIM)
            # Each entry carries curated disease associations → DISEASE node + directed edge
            for comment in res.get("comments", []):
                if comment.get("commentType") != "DISEASE":
                    continue
                disease_info = comment.get("disease", {})
                # UniProt uses diseaseId (acronym) and diseaseAccession (MIM-style)
                dis_acc = disease_info.get("diseaseAccession", "")
                dis_name = disease_info.get("diseaseId") or dis_acc
                if not dis_name:
                    continue
                d_node_id = f"DISEASE_UNIPROT_{dis_acc or dis_name}"
                if not self.g.G.has_node(d_node_id):
                    self.g.add_node({
                        "id": d_node_id, "type": "DISEASE",
                        "label": disease_info.get("diseaseId", dis_name),
                        "description": disease_info.get("description", ""),
                        "mim_id": dis_acc,
                    })
                self.g.add_edge(src=node_id, trgt=d_node_id, attrs={
                    "rel": "ASSOCIATED_WITH_DISEASE",
                    "note": comment.get("note", {}).get("texts", [{}])[0].get("value", ""),
                    "src_layer": "PROTEIN", "trgt_layer": "DISEASE",
                })

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

    async def enrich_functional_dynamics(self):
        """Reactome Pfade und Kausalität."""
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True) if v.get("type") == "PROTEIN" and self._is_active(k)]
        for accession, node in protein_nodes:
            url = f"https://reactome.org/ContentService/data/pathways/low/entity/{accession}"
            try:
                res = await self.fetch_with_retry(url)
                for pw in res:
                    pw_id = f"PATHWAY_{pw['dbId']}"
                    pw_node = {"id": pw_id, "type": "REACTOME_PATHWAY", "label": pw["displayName"]}
                    # stId is the stable Reactome identifier (R-HSA-…) — used by _bridge_reactome_nodes
                    if pw.get("stId"):
                        pw_node["reactome_stable_id"] = pw["stId"]
                    self.g.add_node(pw_node)
                    self.g.add_edge(src=accession, trgt=pw_id,
                                    attrs={"rel": "PARTICIPATES_IN", "causality": "CONTRIBUTORY"})
            except Exception:
                continue

    def _bridge_reactome_nodes(self):
        """
        Cross-link MOLECULE_CHAIN nodes (from UniProt xref stable IDs, e.g. R-HSA-…)
        with REACTOME_PATHWAY nodes (numeric dbId from Reactome ContentService) that
        represent the SAME biological pathway.

        Both node types are created independently during Phase 2 and Phase 6.
        This bridge adds a SAME_AS edge so downstream graph traversal can treat them
        as co-referential without merging them (preserving provenance).
        """
        # INDEX: stable_id → MOLECULE_CHAIN node id
        mol_by_stable: dict[str, str] = {}
        for n, d in self.g.G.nodes(data=True):
            if d.get("type") == "MOLECULE_CHAIN" and d.get("reactome_stable_id"):
                mol_by_stable[d["reactome_stable_id"]] = n

        bridged = 0
        for pw_node_id, pw_data in self.g.G.nodes(data=True):
            if pw_data.get("type") != "REACTOME_PATHWAY":
                continue
            stable_id = pw_data.get("reactome_stable_id")
            if not stable_id or stable_id not in mol_by_stable:
                continue
            mol_node_id = mol_by_stable[stable_id]
            # Avoid duplicate edges (multigraph may already have one from a prior run)
            existing = {d.get("rel") for _, _, d in self.g.G.edges(mol_node_id, data=True)}
            if "SAME_AS" not in existing:
                self.g.add_edge(src=mol_node_id, trgt=pw_node_id, attrs={
                    "rel": "SAME_AS",
                    "src_layer": "MOLECULE_CHAIN", "trgt_layer": "REACTOME_PATHWAY",
                })
            bridged += 1

        print(f"Phase 6+: {bridged} Reactome MOL↔PATHWAY bridge edges created")

    def apply_eco_weighting(self, entry, protein_id):
        evidences = entry.get("organism", {}).get("evidences", [])
        for ev in evidences:
            eco_code = ev.get("evidenceCode")
            if eco_code:
                reliability, evidence_type = ECO_RELIABILITY.get(eco_code, ECO_DEFAULT)
                self.g.add_node({
                    "id": eco_code, "type": "ECO_EVIDENCE", "label": eco_code,
                    "reliability": reliability,
                    "evidence_type": evidence_type,
                })
                self.g.add_edge(src=protein_id, trgt=eco_code, attrs={
                    "rel": "VALIDATED_BY", "src_layer": "PROTEIN", "trgt_layer": "ECO_EVIDENCE",
                })

    def get_uniprot_url_single_gene(self, id: str):
        # cc_interaction: binary PPI from IntAct/BioGRID curated in UniProt
        # cc_disease: MIM/disease association annotations on the entry
        fields = "accession%2Cid%2Cgene_names%2Ccc_cofactor%2Ccc_pathway%2Ccc_interaction%2Ccc_disease%2Cxref_reactome"
        return f"https://rest.uniprot.org/uniprotkb/{id}?fields={fields}"

    # --- CATEGORY: MOLECULAR STRUCTURE (SMILES) ---
    async def _pubchem_properties_by_name(self, name: str) -> dict | None:
        """PubChem PUG REST: Name -> CanonicalSMILES + MolecularWeight + InChIKey."""
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{quote(name)}/property/CanonicalSMILES,MolecularWeight,InChIKey/JSON"
        )
        try:
            res = await self.client.get(url)
            if res.status_code == 200:
                return res.json()["PropertyTable"]["Properties"][0]
        except Exception:
            pass
        return None

    async def _chebi_fallback(self, label: str) -> dict | None:
        """ChEBI lite search as fallback when PubChem yields no hit."""
        url = f"https://www.ebi.ac.uk/chebi/advancedSearchFwd.do?searchString={quote(label)}&queryBean.stars=ALL&format=json"
        try:
            res = await self.client.get(url)
            if res.status_code == 200:
                data = res.json()
                hits = data.get("ListElement", data.get("searchResults", []))
                if hits:
                    first = hits[0] if isinstance(hits, list) else hits
                    return {"CanonicalSMILES": first.get("smiles"), "MolecularWeight": first.get("mass")}
        except Exception:
            pass
        return None

    async def enrich_molecular_structures(self):
        """
        Kategorie: MOLEKÜL -> SMILES.
        Zieht den atomaren Bauplan live von PubChem (Fallback: ChEBI).
        InChIKey dient als stabiler Primärschlüssel für Node-Deduplication.
        """
        molecule_nodes = [v for k, v in self.g.G.nodes(data=True) if v.get("type") in ["MINERAL", "MOLECULE_CHAIN"] and self._is_active(k)]

        for node in molecule_nodes:
            label = node.get("label", "")

            # PRIMARY: PubChem by compound name
            props = await self._pubchem_properties_by_name(label)

            # FALLBACK: ChEBI if PubChem returned nothing
            if not props or not props.get("CanonicalSMILES"):
                props = await self._chebi_fallback(label)

            if props and props.get("CanonicalSMILES"):
                node["smiles"] = props["CanonicalSMILES"]
                node["atomic_weight"] = props.get("MolecularWeight")
                node["inchikey"] = props.get("InChIKey")
                node["smiles_source"] = "PUBCHEM" if props.get("InChIKey") else "CHEBI"
                print(f"Atomic Structure Attached: {label} (SMILES: {node['smiles']})")
            else:
                node["smiles"] = None
                node["atomic_weight"] = None
                node["smiles_source"] = "NOT_FOUND"
                print(f"SMILES not resolved for: {label}")



    @staticmethod
    def _extract_search_term(label: str) -> str:
        """Phrase-aware extraction: keeps 'Vitamin C' intact instead of splitting to 'C'."""
        tokens = label.strip().split()
        if len(tokens) >= 2 and tokens[0].lower() in COMPOUND_PHRASES:
            return " ".join(tokens[:2])
        return label

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
    async def _fetch_smiles_for_drug(self, drug_name):
        """Hilfsmethode: Holt SMILES für den Wirkstoff von PubChem."""
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(drug_name)}/property/CanonicalSMILES/JSON"
        try:
            res = await self.client.get(url)
            if res.status_code == 200:
                return res.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        except Exception:
            return "N/A"

    async def enrich_pharmacology_quantum_adme(self):
        """
        Konsolidierte Inferenz-Phase: PROTEIN -> PHARMA_COMPOUND -> ATOMIC_STRUCTURE.
        1) ChEMBL mechanisms  2) ChEMBL molecule meta
        3) PubChem quantum signatures  4) Graph linking
        """
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True) if v.get("type") == "PROTEIN" and self._is_active(k)]
        # CACHE: avoid re-fetching molecule meta + PubChem for same ChEMBL ID
        _seen_mol: set[str] = set()

        for protein_id, node in protein_nodes:
            target_label = node.get("label")

            chembl_mech_url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism?target_uniprot_accession={protein_id}&format=json"

            try:
                mech_res = await self.client.get(chembl_mech_url)
                if mech_res.status_code != 200:
                    continue
                mechanisms = mech_res.json().get("mechanisms", [])

                for mech in mechanisms:
                    mol_chembl_id = mech.get("molecule_chembl_id")
                    if not mol_chembl_id:
                        continue
                    drug_node_id = f"DRUG_{mol_chembl_id}"

                    # FAST PATH: already-fetched molecule — just add the edge
                    if mol_chembl_id in _seen_mol:
                        if self.g.G.has_node(drug_node_id):
                            self.g.add_edge(
                                src=drug_node_id, trgt=protein_id,
                                attrs={
                                    "rel": "MODULATES_TARGET",
                                    "action": mech.get("action_type"),
                                    "mechanism": mech.get("mechanism_of_action"),
                                    "src_layer": "PHARMA", "trgt_layer": "PROTEIN",
                                    "eco_code": "ECO:0000313",
                                },
                            )
                        continue
                    _seen_mol.add(mol_chembl_id)

                    mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_chembl_id}?format=json"
                    mol_res = await self.client.get(mol_url)
                    if mol_res.status_code != 200:
                        continue
                    mol_data = mol_res.json()
                    struct_data = mol_data.get("molecule_structures") or {}
                    smiles_str = struct_data.get("canonical_smiles")
                    if not smiles_str:
                        continue

                    drug_name = mol_data.get("pref_name") or mol_chembl_id
                    self.g.add_node({
                        "id": drug_node_id,
                        "type": "PHARMA_COMPOUND",
                        "label": drug_name,
                        "molecule_type": mol_data.get("molecule_type"),
                        "max_phase": mol_data.get("max_phase"),
                        "is_approved": mol_data.get("max_phase") == 4,
                        "chembl_id": mol_chembl_id,
                    })

                    self.g.add_edge(
                        src=drug_node_id, trgt=protein_id,
                        attrs={
                            "rel": "MODULATES_TARGET",
                            "action": mech.get("action_type"),
                            "mechanism": mech.get("mechanism_of_action"),
                            "src_layer": "PHARMA", "trgt_layer": "PROTEIN",
                            "eco_code": "ECO:0000313",
                        },
                    )

                    # PubChem quantum-chemical signatures (one-time per molecule)
                    smiles_node_id = f"SMILES_{hash(smiles_str)}"
                    pc_url = (
                        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                        f"{quote(smiles_str, safe='')}"
                        f"/property/DipoleMoment,XLogP3,Complexity,MolecularWeight,InChIKey/JSON"
                    )

                    try:
                        pc_res = await self.fetch_with_retry(pc_url)
                    except Exception:
                        pc_res = None
                    electron_props = {}
                    if pc_res and "PropertyTable" in pc_res:
                        electron_props = pc_res["PropertyTable"]["Properties"][0]

                    self.g.add_node({
                        "id": smiles_node_id,
                        "type": "ATOMIC_STRUCTURE",
                        "label": "Molecular_SMILES",
                        "smiles": smiles_str,
                        "inchikey": electron_props.get("InChIKey"),
                        "dipole_moment": electron_props.get("DipoleMoment"),
                        "lipophilicity_logp": electron_props.get("XLogP3"),
                        "molecular_weight": electron_props.get("MolecularWeight"),
                        "complexity": electron_props.get("Complexity"),
                    })

                    self.g.add_edge(
                        src=drug_node_id, trgt=smiles_node_id,
                        attrs={"rel": "HAS_STRUCTURE", "src_layer": "PHARMA",
                               "trgt_layer": "ATOMIC_STRUCTURE"},
                    )
                    self.g.add_edge(
                        src=smiles_node_id, trgt=protein_id,
                        attrs={"rel": "PHYSICAL_BINDING", "src_layer": "ATOMIC_STRUCTURE",
                               "trgt_layer": "PROTEIN"},
                    )
                    print(f"Enriched: {drug_name} -> Structure & Quantum Ingested")

            except Exception as e:
                print(f"Inference Error for Protein {target_label}: {e}")

    async def _fetch_smiles(self, name):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        res = await self.fetch_with_retry(url)
        return res["PropertyTable"]["Properties"][0]["CanonicalSMILES"] if res else "N/A"

    # --- SÄULE 1: PHARMAKOGENOMIK (ClinPGx / PharmGKB) ───────────────
    async def enrich_pharmacogenomics(self):
        """
        Verknüpft PHARMA_COMPOUND-Nodes mit genetischen Varianten via ClinPGx.
        Pfad: PHARMA_COMPOUND -> CLINICAL_ANNOTATION -> GENETIC_VARIANT -> GENE
        Ermöglicht personalisierte Toxizitäts- und Metabolisierungswarnungen.
        """
        _CLINPGX_BASE = "https://api.clinpgx.org/v1"
        drug_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                      if v.get("type") == "PHARMA_COMPOUND" and self._is_active(k)]

        for node_id, drug in drug_nodes:
            drug_label = drug.get("label")
            if not drug_label:
                continue

            try:
                # A: ClinPGx Chemical-ID via Wirkstoffname
                chem_url = f"{_CLINPGX_BASE}/data/chemical?name={quote(drug_label)}"
                chem_res = await self.client.get(chem_url)
                if chem_res.status_code != 200:
                    continue
                chem_data = chem_res.json().get("data", [])
                if not chem_data:
                    continue

                pgkb_id = chem_data[0].get("id")
                if not pgkb_id:
                    continue

                # B: Klinische Annotationen für diesen Wirkstoff
                # ClinPGx Rate Limit: 2 req/s
                await asyncio.sleep(0.5)
                annot_url = f"{_CLINPGX_BASE}/report/connectedObjects/{pgkb_id}/ClinicalAnnotation"
                annot_res = await self.client.get(annot_url)
                if annot_res.status_code != 200:
                    continue
                annotations = annot_res.json().get("data", [])

                for annot in annotations:
                    annot_id = annot.get("id")
                    if not annot_id:
                        continue

                    # C: CLINICAL_ANNOTATION Node
                    ca_node_id = f"CLIN_ANNOT_{annot_id}"
                    self.g.add_node({
                        "id": ca_node_id,
                        "type": "CLINICAL_ANNOTATION",
                        "label": annot.get("name", annot_id),
                        "evidence_level": annot.get("evidenceLevel"),
                        "phenotype_category": annot.get("phenotypeCategory"),
                        "pgkb_id": annot_id,
                    })

                    self.g.add_edge(
                        src=node_id,
                        trgt=ca_node_id,
                        attrs={
                            "rel": "CLINICAL_SIGNIFICANCE",
                            "src_layer": "PHARMA",
                            "trgt_layer": "GENETICS",
                        },
                    )

                    # D: GENETIC_VARIANT Nodes + Rückverlinkung zu GENE
                    for var in annot.get("relatedVariants", []):
                        var_symbol = var.get("name") or var.get("symbol")
                        if not var_symbol:
                            continue

                        var_node_id = f"VARIANT_{var_symbol}"
                        self.g.add_node({
                            "id": var_node_id,
                            "type": "GENETIC_VARIANT",
                            "label": var_symbol,
                            "location": var.get("location"),
                            "pgkb_id": var.get("id"),
                        })

                        self.g.add_edge(
                            src=ca_node_id,
                            trgt=var_node_id,
                            attrs={"rel": "ASSOCIATED_VARIANT"},
                        )

                    # E: Verlinkung Variante -> bestehendes GENE via relatedGenes
                    for gene_ref in annot.get("relatedGenes", []):
                        gene_symbol = gene_ref.get("symbol") or gene_ref.get("name")
                        if not gene_symbol:
                            continue
                        target_gene_id = f"GENE_{gene_symbol}"
                        # Nur verlinken wenn der GENE-Node existiert
                        if self.g.G.has_node(target_gene_id):
                            for var in annot.get("relatedVariants", []):
                                vs = var.get("name") or var.get("symbol")
                                if vs:
                                    self.g.add_edge(
                                        src=f"VARIANT_{vs}",
                                        trgt=target_gene_id,
                                        attrs={
                                            "rel": "VARIANT_OF",
                                            "src_layer": "GENETICS",
                                            "trgt_layer": "GENE",
                                        },
                                    )

                print(f"PGx Enriched: {drug_label} ({len(annotations)} annotations)")

            except Exception as e:
                print(f"PGx Error for {drug_label}: {e}")

    # --- SÄULE 2: BIOELEKTRISCHE KNOTENEIGENSCHAFTEN (GtoPdb) ──────
    async def enrich_bioelectric_properties(self):
        """
        Integriert biophysikalische Parameter (Ionenselektivität, Leitfähigkeit,
        Spannungsabhängigkeit) für Proteine via GtoPdb.
        Pfad: PROTEIN -> ELECTRICAL_COMPONENT
        """
        _GTOP_BASE = "https://www.guidetopharmacology.org/services"
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN" and self._is_active(k)]

        for node_id, protein in protein_nodes:
            uniprot_acc = protein.get("id")
            if not uniprot_acc:
                continue

            try:
                # A: GtoPdb Target-ID via UniProt Accession
                lookup_url = f"{_GTOP_BASE}/targets?accession={uniprot_acc}&database=UniProt"
                lookup_res = await self.client.get(lookup_url)
                if lookup_res.status_code != 200:
                    continue
                targets = lookup_res.json()
                if not targets:
                    continue

                target = targets[0]
                target_id = target.get("targetId")
                target_class = target.get("type", "UNKNOWN")
                if not target_id:
                    continue

                # B: Drei biophysikalische Endpunkte parallel
                sel_url = f"{_GTOP_BASE}/targets/{target_id}/ionSelectivity"
                cond_url = f"{_GTOP_BASE}/targets/{target_id}/ionConductance"
                volt_url = f"{_GTOP_BASE}/targets/{target_id}/voltageDependence"

                sel_task = self.client.get(sel_url)
                cond_task = self.client.get(cond_url)
                volt_task = self.client.get(volt_url)
                sel_res, cond_res, volt_res = await asyncio.gather(
                    sel_task, cond_task, volt_task, return_exceptions=True,
                )

                # C: Daten extrahieren (leere Responses = kein Ionenkanal)
                ion_selectivity = []
                if not isinstance(sel_res, Exception) and sel_res.status_code == 200:
                    sel_data = sel_res.json()
                    if sel_data:
                        ion_selectivity = [
                            entry.get("ion", entry.get("species", ""))
                            for entry in (sel_data if isinstance(sel_data, list) else [sel_data])
                            if entry
                        ]

                conductance_pS = None
                if not isinstance(cond_res, Exception) and cond_res.status_code == 200:
                    cond_data = cond_res.json()
                    if cond_data:
                        first_cond = cond_data[0] if isinstance(cond_data, list) else cond_data
                        conductance_pS = first_cond.get("conductance") or first_cond.get("value")

                v_half = None
                slope_factor = None
                if not isinstance(volt_res, Exception) and volt_res.status_code == 200:
                    volt_data = volt_res.json()
                    if volt_data:
                        first_volt = volt_data[0] if isinstance(volt_data, list) else volt_data
                        v_half = first_volt.get("vHalf") or first_volt.get("v_half")
                        slope_factor = first_volt.get("slopeFactor") or first_volt.get("slope")

                # D: ELECTRICAL_COMPONENT nur erstellen wenn Daten vorhanden
                if not (ion_selectivity or conductance_pS is not None or v_half is not None):
                    continue

                biophys_id = f"BIOPHYS_{uniprot_acc}"
                self.g.add_node({
                    "id": biophys_id,
                    "type": "ELECTRICAL_COMPONENT",
                    "label": f"Circuit_{target.get('name', uniprot_acc)}",
                    "target_class": target_class,
                    "ion_selectivity": ion_selectivity,
                    "conductance_pS": conductance_pS,
                    "v_half_activation": v_half,
                    "slope_factor": slope_factor,
                    "species": "Human",
                })

                self.g.add_edge(
                    src=node_id,
                    trgt=biophys_id,
                    attrs={
                        "rel": "DESCRIBED_AS_COMPONENT",
                        "src_layer": "PROTEIN",
                        "trgt_layer": "BIOELECTRIC",
                    },
                )

                print(f"Bioelectric Enriched: {protein.get('label')} -> {target_class} "
                      f"(ions={ion_selectivity}, g={conductance_pS}pS, V½={v_half})")

            except Exception as e:
                print(f"Bioelectric Error for {protein.get('label')}: {e}")

    # --- SÄULE 3: MIKROBIOM-METABOLISMUS-ACHSE (VMH) ───────────────
    async def enrich_microbiome_axis(self):
        """
        Modelliert die Transformation von Molekülen durch das Mikrobiom via VMH.
        Pfad: ATOMIC_STRUCTURE -> MICROBIAL_STRAIN -> VMH_METABOLITE -> PROTEIN
        Erklärt indirekte Wirkstoffeffekte durch bakterielle Metabolisierung.
        """
        _VMH_BASE = "https://www.vmh.life/_api"
        mol_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "ATOMIC_STRUCTURE" and self._is_active(k)]

        for node_id, mol in mol_nodes:
            mol_label = mol.get("label", "")
            if not mol_label or mol_label == "Molecular_SMILES":
                # ATOMIC_STRUCTURE Nodes mit generischem Label brauchen InChIKey
                mol_label = mol.get("inchikey") or ""
            if not mol_label:
                continue

            try:
                # A: VMH Metabolit-Suche
                met_url = f"{_VMH_BASE}/metabolites/?search={quote(mol_label)}&format=json&page_size=3"
                met_res = await self.client.get(met_url, timeout=15.0)
                if met_res.status_code != 200:
                    continue
                met_data = met_res.json()
                results = met_data.get("results", [])
                if not results:
                    continue

                vmh_met = results[0]
                vmh_abbr = vmh_met.get("abbreviation")
                if not vmh_abbr:
                    continue

                # B: VMH Metabolit-Node (Brücke zwischen Graph und Mikrobiom)
                vmh_met_id = f"VMH_MET_{vmh_abbr}"
                self.g.add_node({
                    "id": vmh_met_id,
                    "type": "VMH_METABOLITE",
                    "label": vmh_met.get("fullName", vmh_abbr),
                    "vmh_abbreviation": vmh_abbr,
                    "charged_formula": vmh_met.get("chargedFormula"),
                    "inchi_string": vmh_met.get("inchiString"),
                })

                # C: Mikroben die diesen Metaboliten verarbeiten
                microbe_url = f"{_VMH_BASE}/microbes/?metabolite={quote(vmh_abbr)}&format=json&page_size=10"
                microbe_res = await self.client.get(microbe_url, timeout=15.0)
                if microbe_res.status_code != 200:
                    continue
                microbe_data = microbe_res.json().get("results", [])

                for mic in microbe_data:
                    mic_name = mic.get("organism") or mic.get("reconstruction")
                    if not mic_name:
                        continue

                    mic_node_id = f"MICROBE_{mic_name.replace(' ', '_')}"
                    self.g.add_node({
                        "id": mic_node_id,
                        "type": "MICROBIAL_STRAIN",
                        "label": mic_name,
                        "phylum": mic.get("phylum"),
                        "family": mic.get("family"),
                        "metabolic_role": "CONSUMER",
                    })

                    # Edge: Quellmolekül -> Mikrobe
                    self.g.add_edge(
                        src=node_id,
                        trgt=mic_node_id,
                        attrs={
                            "rel": "METABOLIZED_BY",
                            "src_layer": "ATOMIC",
                            "trgt_layer": "MICROBIOME",
                        },
                    )

                # D: Fermentationsprodukte dieses Metaboliten
                ferm_url = f"{_VMH_BASE}/fermcarbon/?metabolite={quote(vmh_abbr)}&format=json&page_size=10"
                ferm_res = await self.client.get(ferm_url, timeout=15.0)
                if ferm_res.status_code == 200:
                    ferm_data = ferm_res.json().get("results", [])
                    for ferm in ferm_data:
                        ferm_model = ferm.get("model")
                        source_type = ferm.get("sourcetype", "")
                        if not ferm_model:
                            continue

                        # Fermentations-Mikrobe als PRODUCER markieren
                        ferm_mic_id = f"MICROBE_{ferm_model.replace(' ', '_')}"
                        self.g.add_node({
                            "id": ferm_mic_id,
                            "type": "MICROBIAL_STRAIN",
                            "label": ferm_model,
                            "metabolic_role": "PRODUCER" if "Fermentation" in source_type else "CONSUMER",
                        })

                        self.g.add_edge(
                            src=ferm_mic_id,
                            trgt=vmh_met_id,
                            attrs={
                                "rel": "PRODUCES_METABOLITE",
                                "source_type": source_type,
                                "src_layer": "MICROBIOME",
                                "trgt_layer": "METABOLITE",
                            },
                        )

                # E: Rückverlinkung VMH_METABOLITE -> PROTEIN via VMH Gene-Mapping
                gene_url = f"{_VMH_BASE}/genes/?reaction={quote(vmh_abbr)}&format=json&page_size=5"
                gene_res = await self.client.get(gene_url, timeout=15.0)
                if gene_res.status_code == 200:
                    gene_hits = gene_res.json().get("results", [])
                    for gh in gene_hits:
                        gene_symbol = gh.get("symbol")
                        if not gene_symbol:
                            continue
                        # Suche passendes PROTEIN über GENE-Node
                        gene_node_id = f"GENE_{gene_symbol}"
                        if self.g.G.has_node(gene_node_id):
                            # TRAVERSE: GENE neighbors to find linked PROTEIN via ENCODED_BY edge
                            for gn_neighbor in self.g.G.neighbors(gene_node_id):
                                if self.g.G.nodes.get(gn_neighbor, {}).get("type") == "PROTEIN":
                                    self.g.add_edge(
                                        src=vmh_met_id,
                                        trgt=gn_neighbor,
                                        attrs={
                                            "rel": "TARGETS_HUMAN",
                                            "src_layer": "METABOLITE",
                                            "trgt_layer": "PROTEIN",
                                        },
                                    )
                                    break

                print(f"Microbiome Enriched: {mol_label} -> VMH:{vmh_abbr} "
                      f"({len(microbe_data)} microbes)")

            except Exception as e:
                print(f"Microbiome Error for {mol_label}: {e}")

    # --- SÄULE 4: ZELLULÄRE INTEGRATION (HPA + Cell Ontology) ──────

    @staticmethod
    def _parse_hpa_cell_enrichment(rtcte_value: str) -> list[str]:
        """
        HPA 'RNA tissue cell type enrichment' -> list of cell-type labels.
        Format: 'Cell type enriched (Hepatocytes);Group enriched (Epithelial cells)'
        """
        if not rtcte_value:
            return []
        _SKIP = {"not detected", "low cell type specificity", "n/a", ""}
        cell_types: list[str] = []
        for segment in rtcte_value.split(";"):
            segment = segment.strip()
            if segment.lower() in _SKIP:
                continue
            paren_start = segment.find("(")
            paren_end = segment.rfind(")")
            if paren_start != -1 and paren_end > paren_start:
                inner = segment[paren_start + 1:paren_end].strip()
                for ct in inner.split(","):
                    ct = ct.strip()
                    if ct:
                        cell_types.append(ct)
        return cell_types

    async def _resolve_cell_ontology(self, cell_label: str) -> dict | None:
        """
        OLS4 CL lookup: cell_label -> CL-ID + description + parent + marker genes.
        Exact match first, fuzzy fallback for plurals/synonyms.
        """
        _OLS4_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
        try:
            # EXACT match against Cell Ontology
            res = await self.client.get(
                _OLS4_SEARCH,
                params={"q": cell_label, "ontology": "cl", "exact": "true", "rows": 1},
                timeout=15.0,
            )
            if res.status_code != 200:
                return None
            docs = res.json().get("response", {}).get("docs", [])

            # FALLBACK: fuzzy search for plural forms / synonyms
            if not docs:
                res = await self.client.get(
                    _OLS4_SEARCH,
                    params={"q": cell_label, "ontology": "cl", "rows": 1},
                    timeout=15.0,
                )
                if res.status_code != 200:
                    return None
                docs = res.json().get("response", {}).get("docs", [])
                if not docs:
                    return None

            hit = docs[0]
            obo_id = hit.get("obo_id", "")
            cl_id = obo_id.replace(":", "_") if obo_id else None
            if not cl_id or not cl_id.startswith("CL_"):
                return None

            desc_raw = hit.get("description", [])
            description = ". ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw or "")

            # MARKER GENES from CL annotation (if available in this release)
            marker_genes = hit.get("annotation", {}).get("has_marker_gene", [])

            return {
                "cl_id": cl_id,
                "label": hit.get("label", cell_label),
                "description": description,
                "marker_genes": marker_genes,
            }
        except Exception as e:
            print(f"OLS4 Error for '{cell_label}': {e}")
            return None

    async def enrich_cell_type_expression(self):
        """
        Zelluläre Integration: GENE -> CELL_TYPE via HPA Expression + OLS4 Metadaten.
        Two-Pass:
          1) HPA rtcte -> provisorische CELL_TYPE Nodes + EXPRESSED_IN_CELL Kanten
          2) OLS4 CL  -> description, CL-ID, parent + HAS_MARKER_GENE Rückverkettung
        """
        _seen: set[str] = set()
        _count = 0

        gene_nodes = [(k, v) for k, v in self.g.G.nodes(data=True) if v.get("type") == "GENE" and self._is_active(k)]

        # ── PASS 1: HPA Expression Data ──────────────────────────────
        for gene_id, gene in gene_nodes:
            if _count >= _MAX_CELL_NODES:
                print(f"CELL_TYPE cap reached ({_MAX_CELL_NODES})")
                break

            gene_name = gene.get("label")
            if not gene_name:
                continue

            hpa_url = (
                f"https://www.proteinatlas.org/api/search_download.php"
                f"?search={quote(gene_name)}&format=json"
                f"&columns=g,eg,up,rtcte&compress=no"
            )
            try:
                res = await self.client.get(hpa_url, timeout=20.0)
                if res.status_code != 200:
                    continue
                entries = res.json()
                if not isinstance(entries, list) or not entries:
                    continue

                # HPA may return multiple genes; prefer exact name match
                matched = next(
                    (e for e in entries if (e.get("Gene") or "").upper() == gene_name.upper()),
                    entries[0],
                )
                rtcte = matched.get("RNA tissue cell type enrichment", "")
                cell_types = self._parse_hpa_cell_enrichment(rtcte)

                for ct_label in cell_types:
                    ct_key = ct_label.lower().strip()
                    cell_node_id = f"CELL_{ct_key.replace(' ', '_').upper()}"

                    if ct_key not in _seen and _count < _MAX_CELL_NODES:
                        self.g.add_node({
                            "id": cell_node_id,
                            "type": "CELL_TYPE",
                            "label": ct_label,
                            "ontology_prefix": "CL",
                            "cl_resolved": False,
                        })
                        _seen.add(ct_key)
                        _count += 1

                    if self.g.G.has_node(cell_node_id):
                        self.g.add_edge(
                            src=gene_id, trgt=cell_node_id,
                            attrs={"rel": "EXPRESSED_IN_CELL", "src_layer": "GENE", "trgt_layer": "CELL"},
                        )

            except Exception as e:
                print(f"HPA Error for {gene_name}: {e}")

        print(f"Pass 1 done: {_count} CELL_TYPE nodes from HPA expression data")

        # ── PASS 2: OLS4 Cell Ontology Metadata ─────────────────────
        unresolved = [(k, v) for k, v in self.g.G.nodes(data=True)
                      if v.get("type") == "CELL_TYPE" and not v.get("cl_resolved")]

        resolved = 0
        for cell_id, cell in unresolved:
            cl_data = await self._resolve_cell_ontology(cell.get("label", ""))
            if not cl_data:
                continue

            cell["cl_id"] = cl_data["cl_id"]
            cell["label"] = cl_data["label"]
            cell["description"] = cl_data["description"]
            cell["cl_resolved"] = True
            resolved += 1

            # HAS_MARKER_GENE: CL annotation -> existing GENE nodes
            for marker in cl_data.get("marker_genes", []):
                target_gene = f"GENE_{marker}"
                if self.g.G.has_node(target_gene):
                    self.g.add_edge(
                        src=cell_id, trgt=target_gene,
                        attrs={"rel": "HAS_MARKER_GENE", "src_layer": "CELL", "trgt_layer": "GENE"},
                    )

        print(f"Pass 2 done: {resolved}/{_count} cells resolved via Cell Ontology (OLS4)")

    async def enrich_tissue_expression_layer(self):
        """
        Tissue-Schicht: HPA Gewebe-nTPM + UBERON (OLS4) + optional CL→Gewebe-Brücke.
        Alle Endpunkte und Helfer sind bewusst in dieser einen Methode gekapselt.
        """
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

                elif child.get("is_cl"):
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
                    lid = f"LAYER_{td.get('uberon_id', tid)}_{layer_idx}"
                    if not self.g.G.has_node(lid):
                        self.g.add_node({
                            "id": lid,
                            "type": "TISSUE_2D_LAYER",
                            "label": f"{td.get('label', '')} — {layer_type} (L{layer_idx})",
                            "layer_idx": layer_idx,
                            "layer_type": layer_type,
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
                lid = f"LAYER_{td.get('uberon_id', tid)}_0"
                if not self.g.G.has_node(lid):
                    self.g.add_node({
                        "id": lid,
                        "type": "TISSUE_2D_LAYER",
                        "label": f"{td.get('label', '')} — generic (L0)",
                        "layer_idx": 0,
                        "layer_type": "generic",
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
                cell_ntpm = 0.0
                for gu, gv, ge in self.g.G.edges(data=True):
                    if ge.get("rel") == "EXPRESSED_IN_TISSUE" and gv == tissue_id:
                        val = ge.get("ntpm", 0.0) or 0.0
                        if val > cell_ntpm:
                            cell_ntpm = val
                density = (cell_ntpm / max_ntpm) if max_ntpm > 0 else 0.1
                density = max(density, 0.1)  # minimum representation

                n_pos = max(1, min(int(density * _MAX_POS_PER_CELL), _MAX_POS_PER_CELL))

                # ── deterministic seed ──
                seed = int(hashlib.md5((cell_id + tissue_id).encode()).hexdigest(), 16) % (2 ** 31)
                rng = random.Random(seed)

                uberon_short = tissue_d.get("uberon_id", tissue_id)
                for i in range(n_pos):
                    x = rng.randint(0, GRID_SIZE - 1)
                    y = rng.randint(0, GRID_SIZE - 1)
                    cp_id = f"CELLPOS_{cell_id}_{uberon_short}_{i}"
                    self.g.add_node({
                        "id": cp_id,
                        "type": "CELL_POSITION",
                        "x": x,
                        "y": y,
                        "layer_idx": _cell_layer_idx(cell_d.get("label", "")),
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

    # ═══════════════════════════════════════════════════════════════════
    # LAYER: SEQUENCE IDENTITY HASHING (SHA-256 Content Addressing)
    # Dedupliziert identische Sequenzen aus verschiedenen Quellen und
    # ermöglicht Fragment-Matching gegen bekannte toxische/allergene Motive.
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _sha256_sequence(seq: str) -> str:
        """Normalisierter SHA-256: uppercase, whitespace-stripped."""
        return hashlib.sha256(seq.strip().upper().encode("utf-8")).hexdigest()

    def compute_sequence_hashes(self):
        """
        Content-Addressing: Jede Aminosäuresequenz bekommt einen SHA-256 Hash-Node.
        Proteine mit identischer Sequenz werden über SEQUENCE_HASH verschmolzen.
        """
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN" and self._is_active(k)]
        _hash_map: dict[str, list[str]] = {}
        hashed = 0

        for node_id, protein in protein_nodes:
            seq = protein.get("sequence")
            if not seq:
                continue

            seq_hash = self._sha256_sequence(seq)
            protein["sequence_hash"] = seq_hash

            if seq_hash not in _hash_map:
                _hash_map[seq_hash] = []
            _hash_map[seq_hash].append(node_id)

        for seq_hash, protein_ids in _hash_map.items():
            hash_node_id = f"SEQHASH_{seq_hash[:16]}"
            self.g.add_node({
                "id": hash_node_id,
                "type": "SEQUENCE_HASH",
                "label": f"SHA256:{seq_hash[:16]}",
                "full_hash": seq_hash,
                "sequence_count": len(protein_ids),
            })
            for pid in protein_ids:
                self.g.add_edge(
                    src=pid, trgt=hash_node_id,
                    attrs={
                        "rel": "HAS_SEQUENCE_IDENTITY",
                        "src_layer": "PROTEIN",
                        "trgt_layer": "SEQUENCE_HASH",
                    },
                )
            hashed += len(protein_ids)

        # DEDUP MARKER: wenn mehrere Proteine denselben Hash teilen
        shared = {h: ids for h, ids in _hash_map.items() if len(ids) > 1}
        for seq_hash, protein_ids in shared.items():
            hash_node_id = f"SEQHASH_{seq_hash[:16]}"
            for i, pid_a in enumerate(protein_ids):
                for pid_b in protein_ids[i + 1:]:
                    self.g.add_edge(
                        src=pid_a, trgt=pid_b,
                        attrs={
                            "rel": "SEQUENCE_IDENTICAL",
                            "via_hash": hash_node_id,
                            "src_layer": "PROTEIN",
                            "trgt_layer": "PROTEIN",
                        },
                    )

        print(f"Sequence Hashing: {hashed} proteins hashed, "
              f"{len(shared)} shared sequences detected")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 11: STRUCTURAL INFERENCE (AlphaFold DB)
    # Verknüpft Proteine mit vorhergesagten 3D-Strukturen.
    # pLDDT-Score dient als Qualitätsmetrik der Vorhersage.
    # ═══════════════════════════════════════════════════════════════════

    async def enrich_structural_layer(self):
        """
        AlphaFold DB Integration: PROTEIN -[HAS_PREDICTED_STRUCTURE]-> 3D_STRUCTURE.
        pLDDT-Score als Zuverlässigkeitsmetrik an der Kante.
        """
        _AF_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN" and self._is_active(k)]
        linked = 0

        for node_id, protein in protein_nodes:
            accession = protein.get("id")
            if not accession:
                continue

            try:
                res = await self.client.get(f"{_AF_BASE}/{accession}", timeout=20.0)
                if res.status_code != 200:
                    continue
                entries = res.json()
                if not entries:
                    continue
                data = entries[0] if isinstance(entries, list) else entries

                model_id = f"STRUCT_{accession}"
                plddt = data.get("globalMetrics", {}).get("globalPlddt") or data.get("uniprotScore")

                self.g.add_node({
                    "id": model_id,
                    "type": "3D_STRUCTURE",
                    "label": f"AlphaFold_{accession}",
                    "pLDDT_avg": plddt,
                    "pdb_url": data.get("pdbUrl"),
                    "cif_url": data.get("cifUrl"),
                    "model_version": data.get("latestVersion"),
                    "gene": data.get("gene"),
                })

                self.g.add_edge(
                    src=node_id, trgt=model_id,
                    attrs={
                        "rel": "HAS_PREDICTED_STRUCTURE",
                        "pLDDT": plddt,
                        "src_layer": "PROTEIN",
                        "trgt_layer": "STRUCTURAL",
                    },
                )
                linked += 1

            except Exception as e:
                print(f"AlphaFold Error for {accession}: {e}")

        print(f"Phase 11: {linked} proteins linked to AlphaFold structures")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 12: DOMAIN DECOMPOSITION (InterPro)
    # Bricht Proteine in funktionale Domänen auf.
    # Ermöglicht funktionale Ähnlichkeitsschlüsse zwischen neuartigen
    # und bekannten Proteinen über gemeinsame Domänen.
    # ═══════════════════════════════════════════════════════════════════

    async def enrich_domain_decomposition(self):
        """
        InterPro Integration: PROTEIN -[CONTAINS_DOMAIN]-> DOMAIN.
        Domänen aus Pfam, SMART, CDD etc. werden als eigenständige Nodes modelliert.
        """
        _INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot"
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN" and self._is_active(k)]
        domain_count = 0

        for node_id, protein in protein_nodes:
            accession = protein.get("id")
            if not accession:
                continue

            try:
                res = await self.client.get(
                    f"{_INTERPRO_BASE}/{accession}",
                    headers={"Accept": "application/json"},
                    timeout=20.0,
                )
                if res.status_code != 200:
                    continue
                payload = res.json()
                results = payload.get("results", [])

                for entry in results:
                    meta = entry.get("metadata", {})
                    ipr_id = meta.get("accession")
                    if not ipr_id:
                        continue

                    domain_node_id = f"DOMAIN_{ipr_id}"
                    # IDEMPOTENT: Domain-Node nur einmal anlegen, mehrfach verlinken
                    if not self.g.G.has_node(domain_node_id):
                        self.g.add_node({
                            "id": domain_node_id,
                            "type": "PROTEIN_DOMAIN",
                            "label": meta.get("name", ipr_id),
                            "interpro_id": ipr_id,
                            "domain_type": meta.get("type"),
                            "source_database": meta.get("source_database"),
                        })

                        # EXPAND: InterPro GO terms -> proper GO_TERM nodes + edges
                        for go_raw in meta.get("go_terms", []):
                            go_ident = go_raw.get("identifier")
                            if not go_ident:
                                continue
                            go_node_id = f"GO_{go_ident.replace(':', '_')}"
                            if not self.g.G.has_node(go_node_id):
                                self.g.add_node({
                                    "id": go_node_id,
                                    "type": "GO_TERM",
                                    "label": go_raw.get("name", go_ident),
                                    "go_id": go_ident,
                                    "aspect": go_raw.get("category", {}).get("name"),
                                })
                            self.g.add_edge(
                                src=domain_node_id, trgt=go_node_id,
                                attrs={
                                    "rel": "ASSOCIATED_GO",
                                    "src_layer": "DOMAIN",
                                    "trgt_layer": "FUNCTIONAL",
                                },
                            )

                    # POSITIONALE INFORMATION: wo sitzt die Domäne in der Sequenz?
                    proteins_block = entry.get("proteins", [])
                    locations = []
                    for prot_entry in proteins_block:
                        for loc_group in prot_entry.get("entry_protein_locations", []):
                            for frag in loc_group.get("fragments", []):
                                locations.append({
                                    "start": frag.get("start"),
                                    "end": frag.get("end"),
                                })

                    self.g.add_edge(
                        src=node_id, trgt=domain_node_id,
                        attrs={
                            "rel": "CONTAINS_DOMAIN",
                            "positions": locations if locations else None,
                            "src_layer": "PROTEIN",
                            "trgt_layer": "DOMAIN",
                        },
                    )
                    domain_count += 1

            except Exception as e:
                print(f"InterPro Error for {accession}: {e}")

        print(f"Phase 12: {domain_count} domain links created")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 13: GO-SEMANTIC-LINKING (Gene Ontology via QuickGO)
    # + COMPARTMENTS LOCALIZATION
    # Erstellt ein semantisches Netz aus Funktionen (Molecular Function,
    # Biological Process) und subzellulärer Lokalisation.
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_go_evidence(raw_code: str) -> tuple[float, str, str]:
        """Resolve QuickGO evidence (short GO code OR ECO URI) -> (reliability, evidence_type, eco_uri)."""
        eco_uri = _GO_EV_TO_ECO.get(raw_code, raw_code)
        reliability, evidence_type = ECO_RELIABILITY.get(eco_uri, ECO_DEFAULT)
        return reliability, evidence_type, eco_uri

    async def enrich_go_semantic_layer(self):
        """
        Gene Ontology via QuickGO: PROTEIN -[ANNOTATED_WITH]-> GO_TERM.
        Evidence codes resolved through short GO codes AND ECO URIs.
        Only annotations with reliability >= 0.5 are kept.
        """
        _QUICKGO_BASE = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
        _MIN_RELIABILITY = 0.5
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN" and self._is_active(k)]
        go_count = 0

        for node_id, protein in protein_nodes:
            accession = protein.get("id")
            if not accession:
                continue

            try:
                res = await self.client.get(
                    _QUICKGO_BASE,
                    params={"geneProductId": accession, "limit": 100, "taxonId": 9606},
                    headers={"Accept": "application/json"},
                    timeout=20.0,
                )
                if res.status_code != 200:
                    continue
                results = res.json().get("results", [])

                for anno in results:
                    go_id = anno.get("goId")
                    if not go_id:
                        continue

                    # RESOLVE: short GO evidence codes (IDA, IPI, …) + full ECO URIs
                    raw_code = anno.get("goEvidence", "")
                    reliability, evidence_type, eco_uri = self._resolve_go_evidence(raw_code)
                    if reliability < _MIN_RELIABILITY:
                        continue

                    go_node_id = f"GO_{go_id.replace(':', '_')}"
                    if not self.g.G.has_node(go_node_id):
                        self.g.add_node({
                            "id": go_node_id,
                            "type": "GO_TERM",
                            "label": anno.get("goName", go_id),
                            "go_id": go_id,
                            "aspect": anno.get("goAspect"),
                        })

                    self.g.add_edge(
                        src=node_id, trgt=go_node_id,
                        attrs={
                            "rel": "ANNOTATED_WITH",
                            "evidence_code": eco_uri,
                            "reliability": reliability,
                            "evidence_type": evidence_type,
                            "qualifier": anno.get("qualifier"),
                            "assigned_by": anno.get("assignedBy"),
                            "extension": anno.get("extensions"),
                            "src_layer": "PROTEIN",
                            "trgt_layer": "FUNCTIONAL",
                        },
                    )
                    go_count += 1

            except Exception as e:
                print(f"QuickGO Error for {accession}: {e}")

        print(f"Phase 13a: {go_count} GO annotations linked (min reliability={_MIN_RELIABILITY})")

    # ── GO TERM METADATA ENRICHMENT ──────────────────────────────────
    async def _enrich_go_term_metadata(self):
        """
        Batch-fetch GO term definitions, synonyms, obsolescence, comments
        from QuickGO /ontology/go/terms/{ids} (up to 25 IDs per call).
        Patches existing GO_TERM nodes in-place.
        """
        _TERM_BASE = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"
        _BATCH_SIZE = 25
        go_nodes = [(n, d) for n, d in self.g.G.nodes(data=True) if d.get("type") == "GO_TERM"]
        enriched = 0

        for i in range(0, len(go_nodes), _BATCH_SIZE):
            chunk = go_nodes[i : i + _BATCH_SIZE]
            ids_csv = ",".join(d["go_id"] for _, d in chunk if d.get("go_id"))
            if not ids_csv:
                continue

            try:
                res = await self.client.get(
                    f"{_TERM_BASE}/{quote(ids_csv, safe='')}",
                    headers={"Accept": "application/json"},
                    timeout=25.0,
                )
                if res.status_code != 200:
                    continue

                for term in res.json().get("results", []):
                    tid = term.get("id")
                    if not tid:
                        continue
                    node_id = f"GO_{tid.replace(':', '_')}"
                    if not self.g.G.has_node(node_id):
                        continue

                    self.g.G.nodes[node_id].update({
                        "definition": (term.get("definition") or {}).get("text"),
                        "synonyms": [s.get("name") for s in term.get("synonyms", []) if s.get("name")],
                        "is_obsolete": term.get("isObsolete", False),
                        "comment": term.get("comment"),
                    })
                    enriched += 1

            except Exception as e:
                print(f"GO Term Metadata Error: {e}")

        print(f"Phase 13a+: {enriched}/{len(go_nodes)} GO_TERM nodes enriched with metadata")

    # ── GO ONTOLOGY HIERARCHY ────────────────────────────────────────
    async def _wire_go_hierarchy(self):
        """
        Build IS_A / PART_OF / REGULATES edges between GO_TERM nodes.
        Uses QuickGO /ontology/go/terms/{ids}/children.

        IMPROVEMENT: when a child node is already in the graph but its intermediate
        parent is NOT, we add the parent as a minimal stub node so the DAG path is
        preserved and can be enriched later by _enrich_go_term_metadata.
        This prevents silent disconnections in sparse GO sub-graphs.
        """
        _CHILDREN_BASE = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"
        _BATCH_SIZE = 25
        _VALID_RELS = {"is_a", "part_of", "regulates", "positively_regulates", "negatively_regulates"}
        go_nodes = [(n, d) for n, d in self.g.G.nodes(data=True) if d.get("type") == "GO_TERM"]
        # MUTABLE set: new stub nodes are added so later batches can reference them
        go_ids_in_graph: set[str] = {n for n, _ in go_nodes}
        hierarchy_count = 0
        stubs_added = 0

        for i in range(0, len(go_nodes), _BATCH_SIZE):
            chunk = go_nodes[i : i + _BATCH_SIZE]
            ids_csv = ",".join(d["go_id"] for _, d in chunk if d.get("go_id"))
            if not ids_csv:
                continue

            try:
                res = await self.client.get(
                    f"{_CHILDREN_BASE}/{quote(ids_csv, safe='')}/children",
                    headers={"Accept": "application/json"},
                    timeout=25.0,
                )
                if res.status_code != 200:
                    continue

                for term in res.json().get("results", []):
                    parent_go = term.get("id")
                    if not parent_go:
                        continue
                    parent_node_id = f"GO_{parent_go.replace(':', '_')}"
                    # Parent is always one of our queried nodes, so it must be in graph
                    if parent_node_id not in go_ids_in_graph:
                        continue

                    for child in term.get("children", []):
                        child_go = child.get("id")
                        relation = child.get("relation", "").lower()
                        if not child_go or relation not in _VALID_RELS:
                            continue

                        child_node_id = f"GO_{child_go.replace(':', '_')}"

                        # STUB PARENT: child is in graph but sibling-path parent is absent
                        # → add a minimal GO_TERM stub so the hierarchy edge is not lost
                        if child_node_id not in go_ids_in_graph:
                            self.g.add_node({
                                "id": child_node_id, "type": "GO_TERM",
                                "label": child_go, "go_id": child_go,
                                "stub": True,  # marks node for future metadata enrichment
                            })
                            go_ids_in_graph.add(child_node_id)
                            stubs_added += 1

                        # DIRECTION: child -[IS_A/PART_OF/…]-> parent (standard GO DAG)
                        self.g.add_edge(
                            src=child_node_id, trgt=parent_node_id,
                            attrs={
                                "rel": relation.upper(),
                                "src_layer": "GO_HIERARCHY",
                                "trgt_layer": "GO_HIERARCHY",
                            },
                        )
                        hierarchy_count += 1

            except Exception as e:
                print(f"GO Hierarchy Error: {e}")

        print(f"Phase 13a++: {hierarchy_count} GO hierarchy edges created ({stubs_added} stub nodes added)")

    # ── GENE -> GO_TERM DERIVED EDGES ────────────────────────────────
    _ASPECT_TO_REL = {
        "molecular_function": "HAS_FUNCTION",
        "biological_process": "INVOLVED_IN_PROCESS",
        "cellular_component": "LOCATED_IN_COMPONENT",
    }

    def _wire_gene_go_edges(self):
        """
        Derive GENE -> GO_TERM edges by traversing
        PROTEIN -[ANNOTATED_WITH]-> GO_TERM and PROTEIN -[ENCODED_BY]-> GENE.
        Uses the strongest reliability per (gene, go_term) pair.
        """
        # COLLECT: protein -> gene mapping
        protein_to_genes: dict[str, list[str]] = {}
        for src, trgt, edata in self.g.G.edges(data=True):
            if edata.get("rel") == "ENCODED_BY":
                protein_to_genes.setdefault(src, []).append(trgt)

        # COLLECT: protein -> go_term annotations with best reliability
        # KEY: (gene_node_id, go_node_id) -> best reliability
        gene_go_best: dict[tuple[str, str], tuple[float, str, str]] = {}

        for src, trgt, edata in self.g.G.edges(data=True):
            if edata.get("rel") != "ANNOTATED_WITH":
                continue
            protein_id = src
            go_node_id = trgt
            reliability = edata.get("reliability", 0)
            evidence_type = edata.get("evidence_type", "")
            go_attrs = self.g.G.nodes.get(go_node_id, {})
            aspect = go_attrs.get("aspect", "")

            for gene_id in protein_to_genes.get(protein_id, []):
                key = (gene_id, go_node_id)
                if key not in gene_go_best or reliability > gene_go_best[key][0]:
                    gene_go_best[key] = (reliability, evidence_type, protein_id)

        derived_count = 0
        for (gene_id, go_node_id), (reliability, evidence_type, protein_id) in gene_go_best.items():
            go_attrs = self.g.G.nodes.get(go_node_id, {})
            aspect = go_attrs.get("aspect", "")
            rel = self._ASPECT_TO_REL.get(aspect, "ANNOTATED_WITH_GENE")

            self.g.add_edge(
                src=gene_id, trgt=go_node_id,
                attrs={
                    "rel": rel,
                    "derived_from": protein_id,
                    "reliability": reliability,
                    "evidence_type": evidence_type,
                    "src_layer": "GENE",
                    "trgt_layer": "FUNCTIONAL",
                },
            )
            derived_count += 1

        print(f"Phase 13a+++: {derived_count} GENE -> GO_TERM derived edges created")

    async def enrich_compartment_localization(self):
        """
        COMPARTMENTS DB (JensenLab): PROTEIN -[LOCALIZED_IN]-> COMPARTMENT.
        Subzelluläre Lokalisation mit Konfidenz-Score.
        """
        _COMP_BASE = "https://compartments.jensenlab.org/Entity"
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN" and self._is_active(k)]
        loc_count = 0
        _MIN_CONFIDENCE = 2.0

        for node_id, protein in protein_nodes:
            accession = protein.get("id")
            gene_label = protein.get("label", "")
            if not accession:
                continue

            try:
                res = await self.client.get(
                    _COMP_BASE,
                    params={"query": gene_label, "type": "9606", "format": "json"},
                    timeout=15.0,
                )
                if res.status_code != 200:
                    continue
                entries = res.json()
                if not isinstance(entries, list):
                    continue

                for entry in entries:
                    compartment = entry.get("compartment", {})
                    comp_id_raw = compartment.get("id") or entry.get("go_id")
                    comp_name = compartment.get("name") or entry.get("name")
                    confidence = float(entry.get("confidence", 0))

                    if not comp_id_raw or not comp_name:
                        continue
                    if confidence < _MIN_CONFIDENCE:
                        continue

                    comp_node_id = f"COMP_{comp_id_raw.replace(':', '_')}"
                    if not self.g.G.has_node(comp_node_id):
                        self.g.add_node({
                            "id": comp_node_id,
                            "type": "COMPARTMENT",
                            "label": comp_name,
                            "go_id": comp_id_raw,
                        })

                        # CROSS-LINK: if go_id matches an existing GO_TERM node, wire them
                        if comp_id_raw and comp_id_raw.startswith("GO:"):
                            mapped_go_node = f"GO_{comp_id_raw.replace(':', '_')}"
                            if self.g.G.has_node(mapped_go_node):
                                self.g.add_edge(
                                    src=comp_node_id, trgt=mapped_go_node,
                                    attrs={
                                        "rel": "MAPPED_TO_GO",
                                        "src_layer": "LOCALIZATION",
                                        "trgt_layer": "FUNCTIONAL",
                                    },
                                )

                    self.g.add_edge(
                        src=node_id, trgt=comp_node_id,
                        attrs={
                            "rel": "LOCALIZED_IN",
                            "confidence": confidence,
                            "source": entry.get("source"),
                            "src_layer": "PROTEIN",
                            "trgt_layer": "LOCALIZATION",
                        },
                    )
                    loc_count += 1

            except Exception as e:
                print(f"COMPARTMENTS Error for {gene_label}: {e}")

        print(f"Phase 13b: {loc_count} localization links created (min confidence={_MIN_CONFIDENCE})")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 13c: GO-CAM CAUSAL ACTIVITY MODELS (BioLink API)
    # Fetches gene-function associations and builds GOCAM_ACTIVITY nodes
    # with causal/structural edges into the GO semantic layer.
    # API: http://api.geneontology.org/api/bioentity/gene/UniProtKB:{acc}/function
    # ═══════════════════════════════════════════════════════════════════

    _BIOLINK_BASE = "http://api.geneontology.org/api"
    _GOCAM_ROWS = 50

    async def enrich_gocam_activities(self):
        """
        GO-CAM via BioLink API: GOCAM_ACTIVITY nodes + causal edges.
        For each GENE node, resolves its PROTEIN neighbor via ENCODED_BY edge,
        then queries functional associations and wires them through self.g (GUtils).
        """
        gene_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                      if v.get("type") == "GENE" and self._is_active(k)]
        activity_count = 0
        edge_count = 0
        _seen: set[str] = set()

        for gene_id, gene_data in gene_nodes:
            # RESOLVE: protein accession via ENCODED_BY edge (PROTEIN -> GENE)
            accession = None
            for neighbor in self.g.G.neighbors(gene_id):
                if self.g.G.nodes.get(neighbor, {}).get("type") == "PROTEIN":
                    accession = neighbor
                    break
            if not accession:
                continue

            url = f"{self._BIOLINK_BASE}/bioentity/gene/UniProtKB:{accession}/function"
            try:
                res = await self.client.get(
                    url,
                    params={"rows": self._GOCAM_ROWS},
                    headers={"Accept": "application/json"},
                    timeout=25.0,
                )
                if res.status_code != 200:
                    continue

                for assoc in res.json().get("associations", []):
                    obj = assoc.get("object", {})
                    go_id = obj.get("id")
                    go_label = obj.get("label", "")
                    go_category = obj.get("category", [])
                    if not go_id:
                        continue

                    rel_label = (assoc.get("relation") or {}).get("label", "associated_with")

                    # UNIQUE: one activity per (accession, GO term, relation)
                    act_id = f"GOCAM_{accession}_{go_id.replace(':', '_')}_{rel_label}"
                    if act_id in _seen:
                        continue
                    _seen.add(act_id)

                    # EVIDENCE + PROVENANCE
                    evidence_types = [
                        ev.get("label") for ev in assoc.get("evidence_types", [])
                        if ev.get("label")
                    ]
                    provided_by = [
                        s for s in assoc.get("provided_by", [])
                        if isinstance(s, str)
                    ]

                    # ENSURE: GO_TERM node exists
                    go_node_id = f"GO_{go_id.replace(':', '_')}"
                    if not self.g.G.has_node(go_node_id):
                        self.g.add_node({
                            "id": go_node_id,
                            "type": "GO_TERM",
                            "label": go_label,
                            "go_id": go_id,
                            "aspect": self._infer_go_aspect(go_category),
                        })

                    # CREATE: GOCAM_ACTIVITY node
                    self.g.add_node({
                        "id": act_id,
                        "type": "GOCAM_ACTIVITY",
                        "label": f"{gene_data.get('label', '')} {rel_label} {go_label}",
                        "activity_relation": rel_label,
                        "evidence_types": evidence_types,
                        "provided_by": provided_by,
                    })
                    activity_count += 1

                    # EDGE: GOCAM_ACTIVITY -[ENABLED_BY]-> GENE
                    self.g.add_edge(
                        src=act_id, trgt=gene_id,
                        attrs={"rel": "ENABLED_BY", "src_layer": "GOCAM", "trgt_layer": "GENE"},
                    )
                    edge_count += 1

                    # EDGE: GOCAM_ACTIVITY -[aspect-rel]-> GO_TERM
                    go_rel = self._gocam_edge_rel(rel_label, go_category)
                    self.g.add_edge(
                        src=act_id, trgt=go_node_id,
                        attrs={"rel": go_rel, "src_layer": "GOCAM", "trgt_layer": "FUNCTIONAL"},
                    )
                    edge_count += 1

            except Exception as e:
                print(f"GO-CAM Error for {accession}: {e}")

        print(f"Phase 13c: {activity_count} GOCAM_ACTIVITY nodes, {edge_count} causal edges created")

    @staticmethod
    def _infer_go_aspect(categories: list) -> str | None:
        """Map BioLink category list to GO aspect string."""
        for cat in categories:
            low = cat.lower() if isinstance(cat, str) else ""
            if "molecular" in low or "activity" in low:
                return "molecular_function"
            if "biological" in low or "process" in low:
                return "biological_process"
            if "cellular" in low or "component" in low:
                return "cellular_component"
        return None

    @staticmethod
    def _gocam_edge_rel(rel_label: str, categories: list) -> str:
        """Choose GOCAM edge rel based on GO aspect and association relation."""
        for cat in categories:
            low = cat.lower() if isinstance(cat, str) else ""
            if "biological" in low or "process" in low:
                return "PART_OF"
            if "cellular" in low or "component" in low:
                return "OCCURS_IN"
        if "enables" in rel_label.lower():
            return "ENABLES"
        if "contributes" in rel_label.lower():
            return "CONTRIBUTES_TO"
        return "ASSOCIATED_WITH"

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 14: ALLERGEN DETECTION (UniProt KW-0020 + Description Scan)
    # Identifiziert allergene Proteine im Graph und erstellt ALLERGEN-
    # Nodes als Brücke zu immunologischen Pfaden.
    # ═══════════════════════════════════════════════════════════════════

    async def detect_allergen_proteins(self):
        """
        Scannt PROTEIN-Nodes auf allergenes Potenzial.
        Primär: UniProt Keyword KW-0020 ("Allergen") via API.
        Fallback: Description-basierte Detektion für bereits geladene Proteine.
        """
        # A: UniProt-Suche nach annotierten humanen Allergenen (KW-0020)
        url = (
            "https://rest.uniprot.org/uniprotkb/search"
            "?query=keyword:KW-0020+AND+organism_id:9606"
            "&fields=accession,id,protein_name,gene_names,keyword"
            "&format=json&size=500"
        )
        allergen_accessions: set[str] = set()
        try:
            res = await self.client.get(url, timeout=30.0)
            if res.status_code == 200:
                for entry in res.json().get("results", []):
                    acc = entry.get("primaryAccession")
                    if acc:
                        allergen_accessions.add(acc)
        except Exception as e:
            print(f"UniProt Allergen KW-0020 Query Error: {e}")

        # B: Match gegen bestehende PROTEIN-Nodes im Graph
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN" and self._is_active(k)]
        detected = 0

        for node_id, protein in protein_nodes:
            accession = protein.get("id", "")
            desc = (protein.get("description") or "").lower()

            # DETECTION: UniProt-Annotation ODER Beschreibung enthält Allergen-Signale
            is_allergen = (
                accession in allergen_accessions
                or "allergen" in desc
                or "ige-binding" in desc
                or "ige binding" in desc
            )
            if not is_allergen:
                continue

            allergen_id = f"ALLERGEN_{accession}"
            self.g.add_node({
                "id": allergen_id,
                "type": "ALLERGEN",
                "label": protein.get("label", accession),
                "description": protein.get("description"),
                "detection_method": "UNIPROT_KW0020" if accession in allergen_accessions else "DESCRIPTION_SCAN",
            })
            self.g.add_edge(
                src=accession, trgt=allergen_id,
                attrs={
                    "rel": "IS_ALLERGEN",
                    "src_layer": "PROTEIN",
                    "trgt_layer": "ALLERGEN",
                },
            )
            detected += 1

        print(f"Phase 14: {detected} allergen proteins detected "
              f"({len(allergen_accessions)} UniProt KW-0020 matches)")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 15: ALLERGEN MOLECULAR IMPACT (CTD + Open Targets)
    # A) CTD: Welche Gene werden durch das Allergen in der Expression
    #    verändert? -> Zytokin-Haushalt (IL-4, IL-5, IL-13 etc.)
    # B) Open Targets GraphQL: Immunologische Krankheitsassoziationen
    #    -> ALLERGEN -[TRIGGERS]-> IMMUNE_RESPONSE
    # ═══════════════════════════════════════════════════════════════════

    _OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    _OT_DISEASE_QUERY = """
    query($ensgId: String!) {
      target(ensemblId: $ensgId) {
        approvedSymbol
        associatedDiseases(page: {size: 50, index: 0}) {
          rows {
            disease { id name }
            score
          }
        }
      }
    }
    """

    async def enrich_allergen_molecular_impact(self):
        """
        Molecular Mapping: CTD Chemical-Gene Interactions + Open Targets Disease Links.
        Erzeugt ALTERS_EXPRESSION und TRIGGERS Kanten für den Allergie-Subgraph.
        """
        allergen_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                          if v.get("type") == "ALLERGEN" and self._is_active(k)]

        if not allergen_nodes:
            print("Phase 15: No allergen nodes found, skipping")
            return

        ctd_links = 0
        immune_links = 0

        for allergen_id, allergen in allergen_nodes:
            allergen_label = allergen.get("label", "")

            # RESOLVE source protein via IS_ALLERGEN edge (PROTEIN -> ALLERGEN)
            source_protein = ""
            for neighbor in self.g.G.neighbors(allergen_id):
                if self.g.G.nodes.get(neighbor, {}).get("type") == "PROTEIN":
                    source_protein = neighbor
                    break

            # ── A: CTD Chemical-Gene Interactions ──────────────────────
            # ZIEL: welche Gene werden durch dieses Allergen hochreguliert?
            ctd_url = (
                f"http://ctdbase.org/tools/batchQuery.go"
                f"?inputType=chem&inputTerms={quote(allergen_label)}"
                f"&report=genes_curated&format=json"
            )
            try:
                ctd_res = await self.client.get(ctd_url, timeout=30.0)
                if ctd_res.status_code == 200:
                    content_type = ctd_res.headers.get("content-type", "")
                    if "json" in content_type:
                        raw = ctd_res.json()
                        interactions = raw if isinstance(raw, list) else []

                        for inter in interactions:
                            target_gene = inter.get("GeneSymbol")
                            action = inter.get("InteractionActions", "")
                            organism = inter.get("Organism", "")

                            if "Homo sapiens" not in organism:
                                continue

                            gene_node_id = f"GENE_{target_gene}"
                            if not self.g.G.has_node(gene_node_id):
                                continue

                            impact = "INFLAMMATORY_CASCADE" if "increases" in action.lower() else "REGULATORY"

                            self.g.add_edge(
                                src=allergen_id,
                                trgt=gene_node_id,
                                attrs={
                                    "rel": "ALTERS_EXPRESSION",
                                    "mechanism": action,
                                    "impact": impact,
                                    "src_layer": "ALLERGEN",
                                    "trgt_layer": "GENE",
                                },
                            )
                            ctd_links += 1
            except Exception as e:
                print(f"CTD Error for {allergen_label}: {e}")

            # ── B: Open Targets Disease Associations (GraphQL) ────────
            # Benötigt Ensembl-ID via verlinktem GENE-Node
            ensembl_id = None
            if source_protein and self.g.G.has_node(source_protein):
                for neighbor in self.g.G.neighbors(source_protein):
                    n_data = self.g.G.nodes.get(neighbor, {})
                    if n_data.get("type") == "GENE" and n_data.get("ensembl_id"):
                        ensembl_id = n_data["ensembl_id"]
                        break

            if not ensembl_id:
                continue

            try:
                ot_res = await self.client.post(
                    self._OT_URL,
                    json={"query": self._OT_DISEASE_QUERY, "variables": {"ensgId": ensembl_id}},
                    timeout=20.0,
                )
                if ot_res.status_code != 200:
                    continue

                target_data = ot_res.json().get("data", {}).get("target", {})
                rows = (target_data.get("associatedDiseases") or {}).get("rows", [])

                for row in rows:
                    disease = row.get("disease", {})
                    disease_name = disease.get("name", "")
                    disease_id = disease.get("id", "")
                    score = row.get("score", 0)

                    # FILTER: immunologisch relevante Assoziationen mit Score > 0.3
                    disease_lower = disease_name.lower()
                    is_immune = any(t in disease_lower for t in (
                        "allerg", "hypersensitiv", "asthma", "dermatitis",
                        "rhinitis", "anaphyla", "urticaria", "eczema",
                        "atopic", "immun", "inflammat", "histamin", "mast cell",
                    ))
                    if not is_immune or score < 0.3:
                        continue

                    response_id = f"IMMUNE_{disease_id.replace(':', '_')}"
                    if not self.g.G.has_node(response_id):
                        self.g.add_node({
                            "id": response_id,
                            "type": "IMMUNE_RESPONSE",
                            "label": disease_name,
                            "disease_id": disease_id,
                            "association_score": score,
                        })

                    self.g.add_edge(
                        src=allergen_id, trgt=response_id,
                        attrs={
                            "rel": "TRIGGERS",
                            "score": score,
                            "src_layer": "ALLERGEN",
                            "trgt_layer": "IMMUNE",
                        },
                    )
                    immune_links += 1

            except Exception as e:
                print(f"Open Targets Error for {allergen_label}: {e}")

        print(f"Phase 15: {ctd_links} gene expression links (CTD), "
              f"{immune_links} immune response links (Open Targets)")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 16: ALLERGEN-FOOD CROSS-LINKING + KREUZALLERGIE
    # Verbindet FOOD_SOURCE -> ALLERGEN wenn ein Lebensmittel ein Protein
    # enthält, das als Allergen markiert ist. Domänen-Überlappung zwischen
    # Allergenen erzeugt CROSS_REACTIVITY Kanten (Kreuzallergie-Prädiktion).
    # ═══════════════════════════════════════════════════════════════════

    def crosslink_allergen_food_sources(self):
        """
        Cross-Link: FOOD_SOURCE -> ALLERGEN + CROSS_REACTIVITY zwischen Allergenen.
        Ermöglicht Vorhersage, welche Lebensmittel ähnliche Entzündungskaskaden auslösen.
        """
        # LOOKUP: source_protein_accession -> allergen_node_id (via IS_ALLERGEN edges)
        allergen_by_protein: dict[str, str] = {}
        for nid, ndata in self.g.G.nodes(data=True):
            if ndata.get("type") != "ALLERGEN" or not self._is_active(nid):
                continue
            for neighbor in self.g.G.neighbors(nid):
                if self.g.G.nodes.get(neighbor, {}).get("type") == "PROTEIN":
                    allergen_by_protein[neighbor] = nid
                    break

        if not allergen_by_protein:
            print("Phase 16: No allergen nodes for cross-linking")
            return

        # ── A: FOOD_SOURCE -> ALLERGEN via CONTAINS_NUTRIENT Kanten ──
        linked = 0
        for u, v, data in self.g.G.edges(data=True):
            if data.get("rel") != "CONTAINS_NUTRIENT":
                continue
            if self.g.G.nodes.get(u, {}).get("type") != "FOOD_SOURCE":
                continue

            allergen_id = allergen_by_protein.get(v)
            if not allergen_id:
                continue

            self.g.add_edge(
                src=u, trgt=allergen_id,
                attrs={
                    "rel": "CONTAINS_ALLERGEN",
                    "severity": "CRITICAL",
                    "src_layer": "FOOD",
                    "trgt_layer": "ALLERGEN",
                },
            )
            linked += 1
            food_label = self.g.G.nodes.get(u, {}).get("label", u)
            allergen_label = self.g.G.nodes.get(allergen_id, {}).get("label", allergen_id)
            print(f"Critical Path: {food_label} -> ALLERGEN {allergen_label}")

        # ── B: KREUZALLERGIE via gemeinsame Protein-Domänen ──────────
        # Wenn zwei Allergene dieselbe Domäne teilen -> Kreuzreaktivitätsrisiko
        allergen_domains: dict[str, set[str]] = {}
        for protein_acc, a_id in allergen_by_protein.items():
            domains: set[str] = set()
            if self.g.G.has_node(protein_acc):
                for _, neighbor, edata in self.g.G.edges(protein_acc, data=True):
                    if edata.get("rel") == "CONTAINS_DOMAIN":
                        domains.add(neighbor)
            if domains:
                allergen_domains[a_id] = domains

        cross_links = 0
        allergen_list = list(allergen_domains.keys())
        for i, a_id in enumerate(allergen_list):
            for b_id in allergen_list[i + 1:]:
                shared = allergen_domains[a_id] & allergen_domains[b_id]
                if shared:
                    self.g.add_edge(
                        src=a_id, trgt=b_id,
                        attrs={
                            "rel": "CROSS_REACTIVITY",
                            "shared_domains": list(shared),
                            "domain_overlap": len(shared),
                            "src_layer": "ALLERGEN",
                            "trgt_layer": "ALLERGEN",
                        },
                    )
                    cross_links += 1

        print(f"Phase 16: {linked} food-allergen links, {cross_links} cross-reactivity pairs")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 17: CELLULAR COMPONENTS + CODING / NON-CODING GENE MAPPING
    # A) UniProt cc_subcellular_location -> CELLULAR_COMPONENT nodes
    #    linked to PROTEIN and coding GENE nodes.
    # B) Ensembl overlap -> NON_CODING_GENE nodes (lncRNA, miRNA, …)
    #    linked to same CELLULAR_COMPONENT + nearby coding GENE.
    # ═══════════════════════════════════════════════════════════════════

    async def enrich_cellular_components(self):
        """
        Two-pass cellular-component integration:
          A) PROTEIN -> CELLULAR_COMPONENT via UniProt cc_subcellular_location
             + GENE (coding) -> CELLULAR_COMPONENT back-link
          B) Ensembl overlap -> NON_CODING_GENE -> CELLULAR_COMPONENT + GENE
        """
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN" and self._is_active(k)]
        comp_count = 0
        coding_links = 0

        # ── PASS A: UniProt subcellular locations ─────────────────────
        for node_id, protein in protein_nodes:
            accession = protein.get("id")
            if not accession:
                continue

            url = (
                f"https://rest.uniprot.org/uniprotkb/{accession}"
                f"?fields=cc_subcellular_location&format=json"
            )
            try:
                res = await self.fetch_with_retry(url)
                comments = res.get("comments", [])

                # COLLECT: components this protein resides in (for Pass B back-ref)
                protein_comp_ids: list[str] = []

                for comment in comments:
                    if comment.get("commentType") != "SUBCELLULAR LOCATION":
                        continue

                    for sl in comment.get("subcellularLocations", []):
                        loc = sl.get("location", {})
                        sl_id = loc.get("id")
                        sl_label = loc.get("value")
                        if not sl_id or not sl_label:
                            continue

                        comp_node_id = f"CELLCOMP_{sl_id}"

                        # IDEMPOTENT: create component node once
                        if not self.g.G.has_node(comp_node_id):
                            topo = sl.get("topology", {})
                            self.g.add_node({
                                "id": comp_node_id,
                                "type": "CELLULAR_COMPONENT",
                                "label": sl_label,
                                "sl_id": sl_id,
                                "topology": topo.get("value"),
                            })
                            comp_count += 1

                        # ECO: best evidence score from this location's evidences
                        best_rel = 0.0
                        best_eco = None
                        for ev in loc.get("evidences", []):
                            eco_code = ev.get("evidenceCode")
                            rel, _ = ECO_RELIABILITY.get(eco_code, ECO_DEFAULT)
                            if rel > best_rel:
                                best_rel, best_eco = rel, eco_code

                        self.g.add_edge(
                            src=node_id, trgt=comp_node_id,
                            attrs={
                                "rel": "RESIDES_IN",
                                "eco_code": best_eco,
                                "reliability": best_rel,
                                "src_layer": "PROTEIN",
                                "trgt_layer": "CELLULAR_COMPONENT",
                            },
                        )
                        protein_comp_ids.append(comp_node_id)

                # BACK-LINK: coding GENE -> CELLULAR_COMPONENT
                for neighbor in self.g.G.neighbors(node_id):
                    n_data = self.g.G.nodes.get(neighbor, {})
                    if n_data.get("type") != "GENE":
                        continue
                    for cid in protein_comp_ids:
                        self.g.add_edge(
                            src=neighbor, trgt=cid,
                            attrs={
                                "rel": "CODING_GENE_IN_COMPONENT",
                                "src_layer": "GENE",
                                "trgt_layer": "CELLULAR_COMPONENT",
                            },
                        )
                        coding_links += 1

            except Exception as e:
                print(f"CellComp Error for {accession}: {e}")

        print(f"Pass A done: {comp_count} CELLULAR_COMPONENT nodes, "
              f"{coding_links} coding-gene links")

        # ── PASS B: Non-coding genes via Ensembl overlap ──────────────
        gene_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                      if v.get("type") == "GENE"
                      and self._is_active(k)
                      and v.get("chromosome")
                      and v.get("gene_start") is not None
                      and v.get("gene_end") is not None]

        _seen_nc: set[str] = set()
        nc_count = 0
        nc_comp_links = 0

        for gene_id, gene in gene_nodes:
            if nc_count >= _MAX_NC_GENE_NODES:
                print(f"NON_CODING_GENE cap reached ({_MAX_NC_GENE_NODES})")
                break

            chrom = gene["chromosome"]
            start = max(1, gene["gene_start"] - _OVERLAP_FLANK_BP)
            end = gene["gene_end"] + _OVERLAP_FLANK_BP
            region = f"{chrom}:{start}-{end}"

            # BIOTYPE FILTER: only non-coding classes
            biotype_params = ";".join(f"biotype={bt}" for bt in _NC_BIOTYPES)
            ensembl_url = (
                f"https://rest.ensembl.org/overlap/region/homo_sapiens/{region}"
                f"?feature=gene;{biotype_params};content-type=application/json"
            )

            try:
                overlap_res = await self.client.get(ensembl_url, timeout=20.0)
                if overlap_res.status_code != 200:
                    continue
                nc_genes = overlap_res.json()
                if not isinstance(nc_genes, list):
                    continue

                for nc in nc_genes:
                    nc_ensembl = nc.get("id") or nc.get("gene_id")
                    if not nc_ensembl or nc_ensembl in _seen_nc:
                        continue
                    if nc_count >= _MAX_NC_GENE_NODES:
                        break

                    nc_node_id = f"NCGENE_{nc_ensembl}"
                    nc_label = (nc.get("external_name")
                                or nc.get("Name")
                                or nc_ensembl)
                    nc_biotype = nc.get("biotype", "unknown")

                    self.g.add_node({
                        "id": nc_node_id,
                        "type": "NON_CODING_GENE",
                        "label": nc_label,
                        "biotype": nc_biotype,
                        "ensembl_id": nc_ensembl,
                        "chromosome": chrom,
                        "start": nc.get("start"),
                        "end": nc.get("end"),
                    })
                    _seen_nc.add(nc_ensembl)
                    nc_count += 1

                    # EDGE: NON_CODING_GENE -> nearby coding GENE
                    self.g.add_edge(
                        src=nc_node_id, trgt=gene_id,
                        attrs={
                            "rel": "OVERLAPS_CODING_GENE",
                            "src_layer": "NON_CODING",
                            "trgt_layer": "GENE",
                        },
                    )

                    # INFERRED LOCALIZATION: share coding gene's components
                    for _, neighbor, edata in self.g.G.edges(gene_id, data=True):
                        if edata.get("rel") == "CODING_GENE_IN_COMPONENT":
                            self.g.add_edge(
                                src=nc_node_id, trgt=neighbor,
                                attrs={
                                    "rel": "NC_GENE_IN_COMPONENT",
                                    "inferred_from": gene_id,
                                    "src_layer": "NON_CODING",
                                    "trgt_layer": "CELLULAR_COMPONENT",
                                },
                            )
                            nc_comp_links += 1

            except Exception as e:
                print(f"Ensembl Overlap Error for {gene.get('label')}: {e}")

        print(f"Pass B done: {nc_count} NON_CODING_GENE nodes, "
              f"{nc_comp_links} component links (inferred)")

    # --- GRAPH EMBEDDING & TEMPSTORE ─────────────────────────────────

    # EMBED_MODEL: Gemini text-embedding-004 (768-dim, best retrieval quality)
    _EMBED_MODEL = "models/text-embedding-004"
    _EMBED_BATCH = 96  # API max per request ~100, keep headroom

    @staticmethod
    def _node_to_text(node_id: str, attrs: dict) -> str:
        """Deterministic text repr of a node for embedding."""
        parts = [f"[{attrs.get('type', 'NODE')}]", f"id={node_id}"]
        for key in ("label", "description", "smiles", "evidence_type",
                     "ion_selectivity", "metabolic_role", "cl_id",
                     "pLDDT_avg", "interpro_id", "domain_type",
                     "go_id", "aspect", "definition", "sequence_hash", "full_hash",
                     "detection_method",
                     "disease_id", "association_score",
                     "sl_id", "topology", "biotype",
                     "activity_relation"):
            val = attrs.get(key)
            if val:
                parts.append(f"{key}={val}")
        # LIST FIELDS: synonyms (GO_TERM), evidence_types (GOCAM_ACTIVITY)
        for list_key in ("synonyms", "evidence_types"):
            vals = attrs.get(list_key)
            if vals and isinstance(vals, list):
                parts.append(f"{list_key}={';'.join(str(v) for v in vals[:5])}")
        return " | ".join(parts)

    @staticmethod
    def _edge_to_text(src: str, trgt: str, attrs: dict) -> str:
        """Deterministic text repr of an edge for embedding."""
        rel = attrs.get("rel", "RELATED_TO")
        parts = [f"[EDGE:{rel}]", f"{src} -> {trgt}"]
        for key in ("src_layer", "trgt_layer", "action", "mechanism",
                     "causality", "source_type",
                     "impact", "severity", "score", "domain_overlap",
                     "qualifier", "assigned_by", "derived_from",
                     "reliability", "evidence_type"):
            val = attrs.get(key)
            if val:
                parts.append(f"{key}={val}")
        return " | ".join(parts)

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Gemini batch embedding with chunked requests to stay under API limits."""
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._EMBED_BATCH):
            chunk = texts[i : i + self._EMBED_BATCH]
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self._EMBED_MODEL,
                content=chunk,
                task_type="RETRIEVAL_DOCUMENT",
            )
            all_vectors.extend(result["embedding"])
        return all_vectors

    async def embed_graph_to_tempstore(self) -> str:
        """
        EMBED every node & edge -> write plain graph + embedded graph to tempdir.
        Returns the tempdir path containing:
          graph.json            – raw NetworkX node-link export
          graph.embedded.json   – same structure with '_embedding' on each element
        """
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        G: nx.MultiGraph = self.g.G

        # ── 1. Serialise raw graph ──────────────────────────────────────
        raw_data = nx.node_link_data(G)

        # ── 2. Collect texts for nodes ──────────────────────────────────
        node_ids: list[str] = []
        node_texts: list[str] = []
        for nid, attrs in G.nodes(data=True):
            node_ids.append(nid)
            node_texts.append(self._node_to_text(nid, attrs))

        # ── 3. Collect texts for edges ──────────────────────────────────
        edge_keys: list[tuple] = []
        edge_texts: list[str] = []
        for src, trgt, key, attrs in G.edges(data=True, keys=True):
            edge_keys.append((src, trgt, key))
            edge_texts.append(self._edge_to_text(src, trgt, attrs))

        # ── 4. Batch-embed everything in one pass ───────────────────────
        all_texts = node_texts + edge_texts
        if not all_texts:
            print("EMBED: graph empty, nothing to embed")
            return ""

        print(f"EMBED: encoding {len(node_texts)} nodes + {len(edge_texts)} edges …")
        all_vectors = await self._batch_embed(all_texts)

        node_vectors = all_vectors[: len(node_texts)]
        edge_vectors = all_vectors[len(node_texts) :]

        # ── 5. Build embedded copy ──────────────────────────────────────
        embedded_data = nx.node_link_data(G)

        # ATTACH node embeddings
        nid_to_vec = dict(zip(node_ids, node_vectors))
        for node_entry in embedded_data["nodes"]:
            vec = nid_to_vec.get(node_entry["id"])
            if vec:
                node_entry["_embedding"] = vec

        # ATTACH edge embeddings
        ekey_to_vec = dict(zip(edge_keys, edge_vectors))
        for link_entry in embedded_data["links"]:
            k = (link_entry.get("source"), link_entry.get("target"), link_entry.get("key"))
            vec = ekey_to_vec.get(k)
            if vec:
                link_entry["_embedding"] = vec

        # ── 6. Write to tempstore ───────────────────────────────────────
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        store_dir = tempfile.mkdtemp(prefix=f"acid_graph_{ts}_")

        raw_path = os.path.join(store_dir, "graph.json")
        emb_path = os.path.join(store_dir, "graph.embedded.json")

        # INTEGRITY: sha256 of raw for downstream validation
        raw_bytes = json.dumps(raw_data, ensure_ascii=False, default=str).encode()
        with open(raw_path, "wb") as f:
            f.write(raw_bytes)

        emb_bytes = json.dumps(embedded_data, ensure_ascii=False, default=str).encode()
        with open(emb_path, "wb") as f:
            f.write(emb_bytes)

        # MANIFEST with checksums
        manifest = {
            "created_utc": ts,
            "node_count": len(node_ids),
            "edge_count": len(edge_keys),
            "embedding_model": self._EMBED_MODEL,
            "embedding_dim": len(node_vectors[0]) if node_vectors else 0,
            "sha256_raw": hashlib.sha256(raw_bytes).hexdigest(),
            "sha256_embedded": hashlib.sha256(emb_bytes).hexdigest(),
        }
        with open(os.path.join(store_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"EMBED: stored -> {store_dir}")
        print(f"  graph.json          ({len(raw_bytes):,} bytes)")
        print(f"  graph.embedded.json ({len(emb_bytes):,} bytes)")
        return store_dir

    # --- PHASE 18: ELECTRON DENSITY MATRIX (RDKit + PySCF) ─────────
    # SMILES → 3D-Atome → Einteilchen-Dichtematrix (RDM1)
    _PARENT_LAYER_TYPES = frozenset({
        "PHARMA_COMPOUND", "PROTEIN", "MINERAL", "MOLECULE_CHAIN", "GENE",
    })
    # PHYSIK-KONSTANTEN für Anregungsenergie → Laser-Parameter
    _HA_TO_EV = 27.211386
    _EV_TO_NM = 1239.84193
    _EV_TO_HZ = 2.417989242e14
    _NIR_LOW = 650    # nm – untere Grenze therapeutisches Fenster
    _NIR_HIGH = 900   # nm – obere Grenze therapeutisches Fenster
    _MIN_OSC_STRENGTH = 0.001  # Schwelle für messbare Absorption

    def compute_electron_density_matrices(self):
        """
        Übersetzt alle Molekül-Nodes (SMILES) in Atome, berechnet die
        Einteilchen-Dichtematrix via DFT(B3LYP)/def2-SVP + ddCOSMO(Wasser),
        und bestimmt per TD-DFT die Anregungsenergien (Laser-Frequenzen).
        Pro messbarer Anregung wird ein EXCITATION_FREQUENCY Node erstellt.
        Ergebnisse + Frequenzen werden an alle Eltern-Layer geheftet.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from pyscf import gto, dft, tddft, solvent
            import numpy as np
        except ImportError as exc:
            print(f"PHASE 18 SKIP – fehlende Abhängigkeit: {exc}")
            return

        # ── ALLE NODES MIT GÜLTIGEM SMILES SAMMELN ──────────────────
        smiles_nodes = [
            (nid, data) for nid, data in self.g.G.nodes(data=True)
            if data.get("smiles") and data["smiles"] not in (None, "N/A") and self._is_active(nid)
        ]
        if not smiles_nodes:
            print("  Keine Nodes mit SMILES gefunden – überspringe.")
            return

        computed, skipped = 0, 0

        for node_id, node in smiles_nodes:
            smiles = node["smiles"]

            # ── SCHRITT 1: RDKit – SMILES → 3D-Koordinaten ──────────
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  RDKit parse-Fehler: {smiles[:60]} – skip")
                skipped += 1
                continue

            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
                print(f"  3D-Embedding fehlgeschlagen: {smiles[:60]} – skip")
                skipped += 1
                continue
            AllChem.MMFFOptimizeMolecule(mol)

            conf = mol.GetConformer()
            atoms_pyscf = []
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                atoms_pyscf.append([atom.GetSymbol(), [pos.x, pos.y, pos.z]])

            # ── SCHRITT 2: DFT(B3LYP) + ddCOSMO(Wasser) für in-vivo Relevanz
            try:
                pyscf_mol = gto.Mole(verbose=0)
                pyscf_mol.atom = atoms_pyscf
                pyscf_mol.basis = "def2-SVP"
                pyscf_mol.build()

                mf = dft.RKS(pyscf_mol)
                mf.xc = "B3LYP"
                mf = solvent.ddCOSMO(mf)
                mf.with_solvent.eps = 78.3553  # Wasser-Dielektrikum (in vivo)
                mf.kernel()

                rdm1 = mf.make_rdm1()
            except Exception as exc:
                print(f"  PySCF-Fehler für {node_id}: {exc} – skip")
                skipped += 1
                continue

            # ── SCHRITT 3: Ergebnisse auf Node + Eltern-Layer ────────
            n_basis = rdm1.shape[0]
            # OBERES DREIECK (symmetrisch) als flache Liste → JSON-tauglich
            upper_tri = rdm1[np.triu_indices(n_basis)].tolist()

            node["atom_decomposition"] = atoms_pyscf
            node["electron_density_matrix"] = upper_tri
            node["density_matrix_shape"] = [n_basis, n_basis]
            node["density_matrix_basis"] = "def2-SVP"
            node["total_electrons"] = float(np.trace(rdm1))
            node["total_energy_hartree"] = float(mf.e_tot)
            node["scf_converged"] = bool(mf.converged)

            # PROPAGATION: edge from parent-layer neighbors → this structure node
            for neighbor in self.g.G.neighbors(node_id):
                if self.g.G.nodes[neighbor].get("type") in self._PARENT_LAYER_TYPES:
                    self.g.add_edge(
                        src=neighbor, trgt=node_id,
                        attrs={
                            "rel": "HAS_DENSITY_RESULT",
                            "total_electrons": node["total_electrons"],
                            "total_energy_hartree": node["total_energy_hartree"],
                            "basis": "def2-SVP",
                            "scf_converged": node["scf_converged"],
                            "src_layer": self.g.G.nodes[neighbor].get("type", "PARENT"),
                            "trgt_layer": node.get("type", "ATOMIC_STRUCTURE"),
                        },
                    )

            # ── SCHRITT 4: TD-DFT → Anregungsenergien + Laser-Frequenzen ─
            src_layer = node.get("type", "ATOMIC_STRUCTURE")
            try:
                td = mf.TDDFT()
                td.nstates = 10
                td.kernel()

                exc_energies_ev = td.e * self._HA_TO_EV
                osc_strengths = td.oscillator_strength(gauge='length')

                for state_idx, (e_ev, f_osc) in enumerate(
                    zip(exc_energies_ev, osc_strengths), start=1
                ):
                    if f_osc < self._MIN_OSC_STRENGTH:
                        continue

                    wl_nm = self._EV_TO_NM / e_ev
                    freq_hz = e_ev * self._EV_TO_HZ

                    # EXCITATION_FREQUENCY Node
                    freq_id = f"FREQ_{node_id}_S{state_idx}"
                    self.g.add_node({
                        "id": freq_id,
                        "type": "EXCITATION_FREQUENCY",
                        "label": f"S0->S{state_idx} {wl_nm:.1f}nm",
                        "excitation_energy_ev": round(float(e_ev), 6),
                        "wavelength_nm": round(float(wl_nm), 2),
                        "frequency_hz": float(freq_hz),
                        "oscillator_strength": round(float(f_osc), 6),
                        "in_nir_window": bool(self._NIR_LOW <= wl_nm <= self._NIR_HIGH),
                        "solvent": "water",
                        "basis": "def2-SVP",
                        "xc_functional": "B3LYP",
                    })

                    # Edge: Quell-SMILES-Node → EXCITATION_FREQUENCY
                    self.g.add_edge(
                        src=node_id, trgt=freq_id,
                        attrs={
                            "rel": "HAS_EXCITATION",
                            "src_layer": src_layer,
                            "trgt_layer": "PHOTOPHYSICS",
                        },
                    )

                    # PROPAGATION: edge from parent-layer neighbors → freq node
                    for neighbor in self.g.G.neighbors(node_id):
                        if self.g.G.nodes[neighbor].get("type") in self._PARENT_LAYER_TYPES:
                            self.g.add_edge(
                                src=neighbor, trgt=freq_id,
                                attrs={
                                    "rel": "HAS_FREQ_RESULT",
                                    "wavelength_nm": round(float(wl_nm), 2),
                                    "oscillator_strength": round(float(f_osc), 6),
                                    "in_nir_window": bool(self._NIR_LOW <= wl_nm <= self._NIR_HIGH),
                                    "src_layer": self.g.G.nodes[neighbor].get("type", "PARENT"),
                                    "trgt_layer": "EXCITATION_FREQUENCY",
                                },
                            )

                    nir_tag = " [NIR]" if self._NIR_LOW <= wl_nm <= self._NIR_HIGH else ""
                    print(f"    FREQ S{state_idx}: {wl_nm:.1f} nm  f={f_osc:.4f}{nir_tag}")

            except Exception as exc:
                print(f"  TD-DFT-Fehler für {node_id}: {exc} – Frequenzen übersprungen")

            computed += 1
            print(f"  RDM1+TD OK: {node_id}  ({n_basis}x{n_basis}, E={mf.e_tot:.6f} Ha)")

        print(f"  Fertig – {computed} berechnet, {skipped} übersprungen.")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 19: BIOELECTRIC → DISEASE SIGNAL PIPELINE
    # Brücke von Biologie → messbares Signal → Krankheit.
    # 7 Passes: Disease Ontology, Bioelectric State, EM Signature,
    #           Multi-Scale Aggregation, Scan Signal, Inverse Inference.
    # ═══════════════════════════════════════════════════════════════════

    # ── OpenTargets: full disease query (broader than allergen-only) ──
    _OT_FULL_DISEASE_QUERY = """
    query($ensgId: String!) {
      target(ensemblId: $ensgId) {
        approvedSymbol
        associatedDiseases(page: {size: 200, index: 0}) {
          rows {
            disease { id name therapeuticAreas { id label } }
            score
            datatypeScores { id score }
          }
        }
      }
    }
    """

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
                            if chem_type in ("MINERAL", "PHARMA_COMPOUND", "ATOMIC_STRUCTURE"):
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

    async def enrich_scan_2d_ingestion(self, scan_path: str, modality_hint: str | None = None):
        """PHASE 20: Load a 2D medical image into the graph as a RAW_SCAN node.
        Modality is auto-detected from file metadata unless modality_hint overrides."""
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

    async def enrich_scan_segmentation(self):
        """PHASE 21: Segment all RAW_SCAN nodes into SPATIAL_REGION nodes.
        Uses Otsu thresholding + connected-component labelling via SimpleITK.
        Each region gets pixel statistics; anatomy label resolved via OLS4 UBERON."""
        import SimpleITK as sitk

        # OLS4 UBERON endpoint (same as Phase 19)
        _OLS4_UBERON = "https://www.ebi.ac.uk/ols4/api/search"
        _uberon_cache: dict[str, dict | None] = {}

        async def _resolve_uberon(term: str) -> dict | None:
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
                }
                _uberon_cache[key] = result
                return result
            except Exception:
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

            # CONVERT numpy → SimpleITK image
            sitk_img = sitk.GetImageFromArray(pixels.astype(np.float32))

            # OTSU THRESHOLD → binary mask
            otsu_filter = sitk.OtsuThresholdImageFilter()
            otsu_filter.SetInsideValue(0)
            otsu_filter.SetOutsideValue(1)
            binary = otsu_filter.Execute(sitk_img)

            # CONNECTED COMPONENTS → labelled regions
            cc_filter = sitk.ConnectedComponentImageFilter()
            labelled = cc_filter.Execute(binary)
            n_labels = cc_filter.GetObjectCount()

            stats = sitk.LabelStatisticsImageFilter()
            stats.Execute(sitk_img, labelled)

            region_count = 0
            for label_idx in range(1, n_labels + 1):
                if _total_regions >= _MAX_SPATIAL_REGIONS:
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

                # CENTROID from bounding box
                if len(bbox) >= 4:
                    cx = bbox[0] + bbox[2] / 2.0
                    cy = bbox[1] + bbox[3] / 2.0
                else:
                    cx, cy = 0.0, 0.0

                region_id = f"SPATIAL_REGION_{scan_id}_{label_idx}"
                self.g.add_node({
                    "id": region_id,
                    "type": "SPATIAL_REGION",
                    "label": f"Region_{label_idx}_{scan.get('label', scan_id)}",
                    "scan_source": scan_id,
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

            # UBERON LABEL RESOLUTION per region — intensity profile → anatomy query
            region_nodes = [
                (k, v) for k, v in self.g.G.nodes(data=True)
                if v.get("type") == "SPATIAL_REGION" and v.get("scan_source") == scan_id
            ]

            # INTENSITY RANKING: brighter regions → tissue-like, darker → bone/air
            sorted_regions = sorted(region_nodes, key=lambda x: x[1].get("pixel_mean", 0), reverse=True)

            # ANATOMY TERMS derived from modality + intensity tier
            _MODALITY_ANATOMY_TIERS: dict[str, list[str]] = {
                "MRI":        ["white matter", "gray matter", "cerebrospinal fluid"],
                "MRI_T1":     ["white matter", "gray matter", "cerebrospinal fluid"],
                "MRI_T2":     ["gray matter", "white matter", "cerebrospinal fluid"],
                "CT":         ["soft tissue", "bone", "air"],
                "PET":        ["metabolically active tissue", "soft tissue", "background"],
                "ULTRASOUND": ["soft tissue", "fluid", "bone"],
                "FMRI":       ["cortex", "subcortical structure", "white matter"],
            }
            tiers = _MODALITY_ANATOMY_TIERS.get(modality, ["tissue", "connective tissue", "background"])

            for idx, (reg_id, reg) in enumerate(sorted_regions):
                # MAP intensity rank to anatomy tier
                tier_idx = min(idx, len(tiers) - 1)
                anatomy_term = tiers[tier_idx]

                uberon = await _resolve_uberon(anatomy_term)
                if uberon and uberon.get("uberon_id"):
                    self.g.G.nodes[reg_id]["uberon_id"] = uberon["uberon_id"]
                    self.g.G.nodes[reg_id]["anatomy_label"] = uberon["label"]

            print(f"  Scan {scan_id}: {region_count} SPATIAL_REGION nodes")

        print(f"  Phase 21 complete: {_total_regions} total regions")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 22a — UBERON BRIDGE (SPATIAL_REGION → TISSUE / ORGAN)
    # ══════════════════════════════════════════════════════════════════

    async def enrich_scan_uberon_bridge(self):
        """PHASE 22a: Connect SPATIAL_REGION nodes to existing TISSUE / ORGAN nodes
        via UBERON ID matching and Ubergraph part_of traversal."""
        _UBERGRAPH_SPARQL = "https://ubergraph.apps.renci.org/sparql"

        async def _ubergraph_parent_organ(uberon_id: str) -> dict | None:
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
                organ_uberon = organ_uri.split("/")[-1].replace("_", ":")
                return {"uberon_id": organ_uberon, "label": organ_label}
            except Exception:
                return None

        # INDEX: existing TISSUE + ORGAN nodes by uberon_id for fast lookup
        tissue_by_uberon: dict[str, str] = {}
        organ_by_uberon: dict[str, str] = {}
        for nid, nd in self.g.G.nodes(data=True):
            ntype = nd.get("type")
            uid = nd.get("uberon_id")
            if not uid:
                continue
            if ntype == "TISSUE":
                tissue_by_uberon[uid] = nid
            elif ntype == "ORGAN":
                organ_by_uberon[uid] = nid

        spatial_nodes = [
            (k, v) for k, v in self.g.G.nodes(data=True)
            if v.get("type") == "SPATIAL_REGION" and v.get("uberon_id")
        ]

        _tissue_links = 0
        _organ_links = 0
        for reg_id, reg in spatial_nodes:
            uid = reg["uberon_id"]

            # DIRECT MATCH: SPATIAL_REGION → TISSUE
            if uid in tissue_by_uberon:
                self.g.add_edge(
                    src=reg_id, trgt=tissue_by_uberon[uid],
                    attrs={"rel": "MAPS_TO_TISSUE", "src_layer": "SPATIAL_REGION", "trgt_layer": "TISSUE"},
                )
                _tissue_links += 1

            # DIRECT MATCH: SPATIAL_REGION → ORGAN
            if uid in organ_by_uberon:
                self.g.add_edge(
                    src=reg_id, trgt=organ_by_uberon[uid],
                    attrs={"rel": "MAPS_TO_ORGAN", "src_layer": "SPATIAL_REGION", "trgt_layer": "ORGAN"},
                )
                _organ_links += 1
                continue

            # FALLBACK: Ubergraph part_of → parent ORGAN
            parent = await _ubergraph_parent_organ(uid)
            if parent and parent.get("uberon_id"):
                parent_uid = parent["uberon_id"]
                if parent_uid in organ_by_uberon:
                    self.g.add_edge(
                        src=reg_id, trgt=organ_by_uberon[parent_uid],
                        attrs={"rel": "MAPS_TO_ORGAN", "src_layer": "SPATIAL_REGION", "trgt_layer": "ORGAN"},
                    )
                    _organ_links += 1

        print(f"  Phase 22a: {_tissue_links} tissue links, {_organ_links} organ links")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 22b — MODALITY-SPECIFIC FEATURE EXTRACTION
    # ══════════════════════════════════════════════════════════════════

    async def enrich_scan_feature_extraction(self):
        """PHASE 22b: Extract a 5D feature vector per SPATIAL_REGION from pixel stats.
        Creates extended SCAN_SIGNAL nodes with modality_feature_vec.
        Vector layout: [intensity, contrast, modality_proxy, variance, edge_density]."""

        spatial_nodes = [
            (k, v) for k, v in self.g.G.nodes(data=True)
            if v.get("type") == "SPATIAL_REGION"
        ]

        # COMPUTE global intensity range across all regions for normalisation
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

            # DIM 0: normalised intensity (0..1)
            intensity = (px_mean - global_min) / intensity_range

            # DIM 1: contrast = local range / global range
            local_range = px_max - px_min
            contrast = local_range / intensity_range if intensity_range > 0 else 0.0

            # DIM 2: modality-specific proxy value
            if modality in ("MRI_T1", "MRI"):
                # T1 PROXY: bright = white matter (short T1)
                mod_proxy = intensity
            elif modality == "MRI_T2":
                # T2 PROXY: bright = fluid (long T2)
                mod_proxy = 1.0 - intensity
            elif modality == "CT":
                # HU PROXY: scale px_mean into ~[-1000, +3000] HU range
                mod_proxy = (px_mean + 1000.0) / 4000.0
            elif modality == "PET":
                # SUV PROXY: normalised uptake
                mod_proxy = intensity * 1.5
            elif modality == "FMRI":
                # BOLD PROXY: variance-driven
                mod_proxy = min(px_std / (px_mean + 1e-6), 1.0)
            else:
                mod_proxy = intensity

            # DIM 3: variance (normalised)
            variance = min(px_std / (intensity_range + 1e-6), 1.0)

            # DIM 4: edge density ~ (max-min)/mean as a gradient proxy
            edge_density = local_range / (px_mean + 1e-6)
            edge_density = min(edge_density, 1.0)

            feature_vec = [
                round(intensity, 6),
                round(contrast, 6),
                round(mod_proxy, 6),
                round(variance, 6),
                round(edge_density, 6),
            ]

            # CREATE SCAN_SIGNAL node (extended schema)
            scan_signal_id = f"SCAN_SIGNAL_{reg_id}"
            self.g.add_node({
                "id": scan_signal_id,
                "type": "SCAN_SIGNAL",
                "label": f"ScanSig_{reg.get('label', reg_id)}",
                "sensor_type": modality,
                "modality": modality,
                "spatial_region_id": reg_id,
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

    # ══════════════════════════════════════════════════════════════════
    # PHASE 22c — PATHOLOGY FINDING INFERENCE
    # (HPO via OLS4 + cosine similarity → DISEASE)
    # ══════════════════════════════════════════════════════════════════

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
            avg_vec = [sum(v[d] for v in linked_vecs) / n for d in range(_SCAN_FEAT_DIM)]
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
                if len(dis_vec) != _SCAN_FEAT_DIM:
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

    async def _ingest_organ_layer(self, organs: list[str]) -> None:
        """STAGE A — Organ-first top-down ingestion.

        Prompt: Make Sure to First fetch Just Organ Data from Uniprot & co based on the input.

        1. Resolves each organ term → UBERON ontology ID via OLS4 (exact match first, fuzzy fallback).
        2. Creates ORGAN node in the graph (type=ORGAN, uberon_id, description).
        3. Fetches disease/harmful ontology (HPO + MONDO) linked to each organ via OLS4;
           creates DISEASE nodes and ORGAN_ASSOCIATED_DISEASE edges.
        4. Protein seed fetch (parallel): UniProt (organism_id:9606) AND (tissue:{organ}).
        5. Stores (uberon_id, label) in self._organ_uberon_seeds for tissue layer Block A."""
        if not organs:
            print("  Stage A: no organs supplied — skipping")
            return

        if self.active_subgraph is None:
            self.active_subgraph = set()

        _OLS4_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
        organ_nodes_created = 0
        disease_edges = 0

        for organ in organs:
            # ── 1. Resolve UBERON ID — exact match first, fuzzy fallback ──
            uberon_meta: dict | None = None
            try:
                for exact in ("true", "false"):
                    res = await self.client.get(
                        _OLS4_SEARCH,
                        params={"q": organ, "ontology": "uberon", "exact": exact, "rows": 1},
                        timeout=15.0,
                    )
                    if res.status_code == 200:
                        docs = res.json().get("response", {}).get("docs", [])
                        if docs:
                            hit = docs[0]
                            obo_id = hit.get("obo_id", "") or ""
                            sf = obo_id.replace(":", "_") if obo_id else hit.get("short_form", "")
                            if sf and sf.startswith("UBERON_"):
                                desc_raw = hit.get("description", [])
                                uberon_meta = {
                                    "uberon_id": sf,
                                    "label": hit.get("label", organ),
                                    "description": ". ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw or ""),
                                }
                                break
            except Exception as e:
                print(f"  Stage A OLS4 UBERON error for '{organ}': {e}")

            organ_uberon = uberon_meta["uberon_id"] if uberon_meta else None
            organ_label  = uberon_meta["label"]     if uberon_meta else organ
            organ_nid    = f"ORGAN_{organ_uberon or organ.replace(' ', '_')}"

            # ── 2. Create ORGAN node (anchor for the entire organ hierarchy) ──
            if not self.g.G.has_node(organ_nid):
                self.g.add_node({
                    "id": organ_nid,
                    "type": "ORGAN",
                    "label": organ_label,
                    "input_term": organ,
                    "uberon_id": organ_uberon or "",
                    "description": (uberon_meta or {}).get("description", ""),
                    "source": "OLS4_UBERON" if organ_uberon else "input",
                })
                organ_nodes_created += 1

            if organ_uberon:
                self._organ_uberon_seeds.append((organ_uberon, organ_label))

            # ── 3. Disease/harmful ontology for this organ (HPO + MONDO via OLS4) ──
            for ontology in ("hp", "mondo"):
                try:
                    res = await self.client.get(
                        _OLS4_SEARCH,
                        params={"q": organ, "ontology": ontology, "rows": 5, "type": "class"},
                        timeout=10.0,
                    )
                    if res.status_code != 200:
                        continue
                    for doc in res.json().get("response", {}).get("docs", []):
                        obo_id = doc.get("obo_id", "")
                        if not obo_id:
                            continue
                        d_nid = f"DISEASE_{obo_id.replace(':', '_')}"
                        if not self.g.G.has_node(d_nid):
                            dr = doc.get("description", [])
                            self.g.add_node({
                                "id": d_nid,
                                "type": "DISEASE",
                                "label": doc.get("label", obo_id),
                                "description": ". ".join(dr) if isinstance(dr, list) else str(dr or ""),
                                "obo_id": obo_id,
                                "ontology": ontology.upper(),
                                "organ_term": organ,
                            })
                        if not self.g.G.has_edge(organ_nid, d_nid):
                            self.g.add_edge(
                                src=organ_nid, trgt=d_nid,
                                attrs={
                                    "rel": "ORGAN_ASSOCIATED_DISEASE",
                                    "src_layer": "ORGAN",
                                    "trgt_layer": "DISEASE",
                                    "ontology": ontology.upper(),
                                    "source": "OLS4",
                                },
                            )
                            disease_edges += 1
                except Exception as e:
                    print(f"  Stage A OLS4 disease error (organ='{organ}', ont={ontology}): {e}")

        # ── 4. Protein seed fetch — parallel over all organs ──
        tasks = [
            self.fetch_proteins_by_query(f"(organism_id:9606) AND (tissue:{organ})")
            for organ in organs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, list):
                self.active_subgraph.update(res)

        print(f"  Stage A: {organ_nodes_created} ORGAN nodes, "
              f"{disease_edges} ORGAN_ASSOCIATED_DISEASE edges, "
              f"{len(self.active_subgraph)} protein seeds from {len(organs)} organ(s)")

    async def _ingest_function_layer(
        self, functions: list[str], db: str
    ) -> None:
        """STAGE B — Function-driven ingestion.
        If db=='uniprot': QuickGO molecular_function + UniProt keyword.
        If db=='pubchem': PubChem compound → protein xrefs → UniProt ingest.
        Appends new protein IDs to self.active_subgraph."""
        if not functions:
            print("  Stage B: no function annotations supplied — skipping")
            return

        if self.active_subgraph is None:
            self.active_subgraph = set()

        tasks: list = []
        if db == "pubchem":
            # PUBCHEM PATH: compound name → CID → protein xrefs
            for fn in functions:
                tasks.append(self._fetch_pubchem_proteins(fn))
        else:
            # UNIPROT PATH (default): QuickGO + keyword fallback
            for fn in functions:
                tasks.append(self._resolve_function_seeds(fn))
                tasks.append(
                    self.fetch_proteins_by_query(
                        f"(organism_id:9606) AND (keyword:{fn})"
                    )
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        _new = 0
        for res in results:
            if isinstance(res, list):
                self.active_subgraph.update(res)
                _new += len(res)

        print(f"  Stage B ({db}): +{_new} protein seeds from "
              f"{len(functions)} function term(s)")

    async def _fetch_pubchem_proteins(self, term: str) -> list[str]:
        """PUBCHEM SEED PATH: compound name → CID → ProteinGI xrefs → UniProt ingest.
        Returns list of ingested protein accession IDs."""
        accessions: list[str] = []
        try:
            # STEP 1: resolve compound name → CID
            cid_url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{quote(term)}/JSON"
            )
            res = await self.client.get(cid_url, timeout=15.0)
            if res.status_code != 200:
                return accessions
            compounds = res.json().get("PC_Compounds", [])
            if not compounds:
                return accessions
            cid = compounds[0].get("id", {}).get("id", {}).get("cid")
            if not cid:
                return accessions

            # STEP 2: CID → protein xrefs
            xref_url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                f"{cid}/xrefs/ProteinGI/JSON"
            )
            xref_res = await self.client.get(xref_url, timeout=15.0)
            if xref_res.status_code != 200:
                return accessions
            info = xref_res.json().get("InformationList", {}).get("Information", [])
            gi_ids = []
            for block in info:
                gi_ids.extend(block.get("ProteinGI", []))
            if not gi_ids:
                return accessions

            # STEP 3: GI → UniProt accessions (batch via ID Mapping API)
            mapping_url = "https://rest.uniprot.org/idmapping/run"
            body = {"from": "GI_number", "to": "UniProtKB", "ids": ",".join(str(g) for g in gi_ids[:200])}
            map_res = await self.client.post(mapping_url, data=body, timeout=20.0)
            if map_res.status_code not in (200, 303):
                return accessions
            job_id = map_res.json().get("jobId")
            if not job_id:
                return accessions

            # POLL MAPPING RESULT (UniProt async job)
            poll_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
            for _ in range(15):
                await asyncio.sleep(2)
                poll = await self.client.get(poll_url, timeout=10.0)
                if poll.status_code == 200 and poll.json().get("results") is not None:
                    break

            result_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
            result_res = await self.client.get(result_url, timeout=15.0)
            if result_res.status_code != 200:
                return accessions
            for entry in result_res.json().get("results", []):
                acc = entry.get("to", {}).get("primaryAccession")
                if acc:
                    accessions.append(acc)

            # INGEST discovered accessions into graph
            if accessions:
                q = " OR ".join(f"accession:{a}" for a in accessions[:300])
                return await self.fetch_proteins_by_query(f"({q})")

        except Exception as e:
            print(f"  PubChem protein resolve error ({term}): {e}")
        return accessions

    async def _ingest_harmful_layer(self, outsrc_criteria: list[str]) -> None:
        """STAGE C — Harmful / disease pre-seeding.
        Per outsrc term: OLS4 HPO search + UniProt keyword to pre-populate
        DISEASE nodes and disease-relevant proteins before enrichment runs."""
        if not outsrc_criteria:
            print("  Stage C: no outsrc criteria supplied — skipping")
            return

        if self.active_subgraph is None:
            self.active_subgraph = set()

        _OLS4_HPO = "https://www.ebi.ac.uk/ols4/api/search"
        _disease_seeded = 0

        for term in outsrc_criteria:
            # HPO / MONDO disease node pre-seeding via OLS4
            try:
                for ontology in ("hp", "mondo"):
                    res = await self.client.get(
                        _OLS4_HPO,
                        params={"q": term, "ontology": ontology, "rows": 5, "type": "class"},
                        timeout=10.0,
                    )
                    if res.status_code != 200:
                        continue
                    docs = res.json().get("response", {}).get("docs", [])
                    for doc in docs:
                        obo_id = doc.get("obo_id", "")
                        if not obo_id:
                            continue
                        disease_node_id = f"DISEASE_{obo_id.replace(':', '_')}"
                        if not self.g.G.has_node(disease_node_id):
                            self.g.add_node({
                                "id": disease_node_id,
                                "type": "DISEASE",
                                "label": doc.get("label", term),
                                "description": (doc.get("description", [""])[0]
                                                if doc.get("description") else ""),
                                "obo_id": obo_id,
                                "outsrc_term": term,
                            })
                            _disease_seeded += 1
            except Exception as e:
                print(f"  Stage C OLS4 error ({term}): {e}")

            # UniProt keyword → disease-relevant proteins
            try:
                prot_ids = await self.fetch_proteins_by_query(
                    f"(organism_id:9606) AND (keyword:{term})"
                )
                if prot_ids:
                    self.active_subgraph.update(prot_ids)
            except Exception as e:
                print(f"  Stage C UniProt keyword error ({term}): {e}")

        print(f"  Stage C: {_disease_seeded} DISEASE nodes pre-seeded from "
              f"{len(outsrc_criteria)} outsrc term(s)")

    def _node_in_organ_tissue_cascade(self, data: dict) -> bool:
        """True if node belongs to organ→tissue→molecular→atomic/electron seam."""
        if not isinstance(data, dict):
            return False
        if data.get("electron_density_matrix") is not None:
            return True
        if data.get("atom_decomposition"):
            return True
        return data.get("type") in self._ORGAN_TISSUE_CASCADE_TYPES

    def build_tissue_hierarchy_map(self, cfg: dict | None = None) -> nx.Graph:
        """
        Prompt: return a seamless map from organ layer to atomic structure / electron matrix.

        Undirected BFS from ORGAN (or ANATOMY_PART fallback) roots matching cfg['organs'];
        keeps only nodes in _ORGAN_TISSUE_CASCADE_TYPES plus any node carrying
        electron_density_matrix / atom_decomposition (Phase 18 output).
        """
        G = self.g.G
        want = set()
        if cfg:
            want = {str(o).strip().lower() for o in (cfg.get("organs") or []) if str(o).strip()}

        roots: set[str] = set()
        for nid, d in G.nodes(data=True):
            if d.get("type") != "ORGAN":
                continue
            it = (d.get("input_term") or "").strip().lower()
            if want and it in want:
                roots.add(nid)
        if want:
            for nid, d in G.nodes(data=True):
                if d.get("type") != "ORGAN":
                    continue
                if (d.get("label") or "").strip().lower() in want:
                    roots.add(nid)
        if not roots:
            roots = {n for n, d in G.nodes(data=True) if d.get("type") == "ORGAN"}
        if not roots and want:
            for nid, d in G.nodes(data=True):
                if d.get("type") != "ANATOMY_PART":
                    continue
                lab = (d.get("label") or "").strip().lower()
                if any(w == lab or w in lab or lab in w for w in want):
                    roots.add(nid)

        if not roots:
            print("  Tissue hierarchy map: no organ roots — returning empty graph")
            return nx.Graph()

        visited: set[str] = set()
        q: deque[str] = deque()
        for r in roots:
            if r in G:
                visited.add(r)
                q.append(r)

        while q:
            u = q.popleft()
            for v in G.neighbors(u):
                if v in visited:
                    continue
                vd = G.nodes[v]
                if self._node_in_organ_tissue_cascade(vd):
                    visited.add(v)
                    q.append(v)

        H = G.subgraph(visited).copy()
        print(f"  Tissue hierarchy map: {H.number_of_nodes()}N / {H.number_of_edges()}E "
              f"(from {len(roots)} organ/anatomy root(s))")
        return H

    # --- CONSOLIDATED WORKFLOW ---
    async def finalize_biological_graph(
        self,
        cfg: dict,
        scan_path: str | None = None,
        modality_hint: str | None = None,
    ):
        """
        CONFIG-DRIVEN HIERARCHICAL EXTRACTION WORKFLOW.

        Receives:
            cfg: dict[db:str, organs:list, function_annotation:list, outsrc_criteria:list]
                - db: "uniprot" | "pubchem" — starting point for seed resolution
                - organs: organ/tissue terms driving the organ→tissue→cell→protein hierarchy
                - function_annotation: functional terms driving GO/pathway resolution
                - outsrc_criteria: harmful / disease exclusion criteria

        Hierarchical extraction order (top-down):
            Stage A — Organ Layer:  organ → tissue → cells → proteins & genes
            Stage B — Function Layer: function → chemicals, proteins & genes
            Stage C — Harmful Layer:  outsrc_criteria → disease / harmful pre-seeding
            Then enrichment phases 2–22c build the full interconnected graph:
                functions → chems → atomic → electron_matrix → disease and harmful results

        Returns
        -------
        dict with keys:
            gutils — firegraph GUtils wrapper (same as self.g)
            graph — full networkx.Graph (self.g.G)
            tissue_hierarchy_map — networkx.Graph: seamless organ→tissue→…→electron slice
        """
        _required = {"db", "organs", "function_annotation", "outsrc_criteria"}
        _missing = _required - set(cfg)
        if _missing:
            raise ValueError(f"cfg missing required keys: {_missing}")
        self.workflow_cfg = cfg

        # DELTA HELPERS: track nodes/edges added per phase for progress visibility
        def _snap() -> tuple[int, int]:
            return self.g.G.number_of_nodes(), self.g.G.number_of_edges()

        def _delta(before: tuple[int, int]) -> None:
            n0, e0 = before
            n1, e1 = self.g.G.number_of_nodes(), self.g.G.number_of_edges()
            dn, de = n1 - n0, e1 - e0
            print(f"  ↳ +{dn}N / +{de}E  →  total {n1}N / {e1}E")

        try:
            # ═══ STAGE A: ORGAN-DRIVEN SEED INGESTION ═══════════════════
            print(f"--- STAGE A: Organ-Driven Seed Ingestion  [{len(cfg['organs'])} term(s)] ---")
            _s = _snap(); await self._ingest_organ_layer(cfg["organs"]); _delta(_s)

            # ═══ STAGE B: FUNCTION-DRIVEN SEED INGESTION ════════════════
            print(f"--- STAGE B: Function-Driven Seed Ingestion  [{len(cfg['function_annotation'])} term(s)] ---")
            _s = _snap(); await self._ingest_function_layer(cfg["function_annotation"], cfg["db"]); _delta(_s)

            # ═══ STAGE C: HARMFUL / OUTSRC PRE-SEEDING ══════════════════
            print(f"--- STAGE C: Harmful / Outsrc Pre-Seeding  [{len(cfg['outsrc_criteria'])} term(s)] ---")
            _s = _snap(); await self._ingest_harmful_layer(cfg["outsrc_criteria"]); _delta(_s)

            n_seed, e_seed = _snap()
            print(f"  Seed complete — {n_seed}N / {e_seed}E  "
                  f"(active_subgraph: {len(self.active_subgraph) if self.active_subgraph else 'ALL'})")

            # ═══ ENRICHMENT PHASES (respect active_subgraph) ════════════
            print("--- PHASE 2: Deep Fetching UniProt Details (PPI / Disease / Reactome) ---")
            _s = _snap(); await self.enrich_gene_nodes_deep(); _delta(_s)

            print("--- PHASE 3: Live Pharmacology (ChEMBL + BfArM) ---")
            _s = _snap(); await self.enrich_pharmacology_quantum_adme(); _delta(_s)

            print("--- PHASE 4: Atomic & Molecular Mapping (SMILES) ---")
            _s = _snap(); await self.enrich_molecular_structures(); _delta(_s)

            print("--- PHASE 5: Nutritional Origin (Open Food Facts DE) ---")
            _s = _snap(); await self.enrich_food_sources(); _delta(_s)

            print("--- PHASE 6: Genomic & Functional Enrichment (Ensembl + Reactome) ---")
            _s = _snap()
            await asyncio.gather(self.enrich_genomic_data(), self.enrich_functional_dynamics())
            _delta(_s)

            print("--- PHASE 6+: Reactome MOL↔PATHWAY Bridge ---")
            _s = _snap(); self._bridge_reactome_nodes(); _delta(_s)

            print("--- PHASE 7: Pharmacogenomics (ClinPGx) ---")
            _s = _snap(); await self.enrich_pharmacogenomics(); _delta(_s)

            print("--- PHASE 8: Bioelectric Properties (GtoPdb) ---")
            _s = _snap(); await self.enrich_bioelectric_properties(); _delta(_s)

            print("--- PHASE 9: Microbiome Metabolism (VMH) ---")
            _s = _snap(); await self.enrich_microbiome_axis(); _delta(_s)

            print("--- PHASE 10: Cellular Integration (HPA + Cell Ontology) ---")
            _s = _snap(); await self.enrich_cell_type_expression(); _delta(_s)

            print("--- PHASE 10b: Tissue Integration (HPA + Uberon + CL bridge) ---")
            _s = _snap(); await self.enrich_tissue_expression_layer(); _delta(_s)

            print("--- PHASE 10.5: Sequence Identity Hashing (SHA-256) ---")
            _s = _snap(); self.compute_sequence_hashes(); _delta(_s)

            print("--- PHASE 11: Structural Inference (AlphaFold DB) ---")
            _s = _snap(); await self.enrich_structural_layer(); _delta(_s)

            print("--- PHASE 12: Domain Decomposition (InterPro) ---")
            _s = _snap(); await self.enrich_domain_decomposition(); _delta(_s)

            print("--- PHASE 13a: GO-Semantic-Linking (QuickGO) ---")
            _s = _snap(); await self.enrich_go_semantic_layer(); _delta(_s)

            print("--- PHASE 13a+: GO Term Metadata Enrichment ---")
            _s = _snap(); await self._enrich_go_term_metadata(); _delta(_s)

            print("--- PHASE 13a++: GO Ontology Hierarchy (with stub parents) ---")
            _s = _snap(); await self._wire_go_hierarchy(); _delta(_s)

            print("--- PHASE 13a+++: GENE → GO_TERM Derived Edges ---")
            _s = _snap(); self._wire_gene_go_edges(); _delta(_s)

            print("--- PHASE 13b: Subcellular Localization (COMPARTMENTS) ---")
            _s = _snap(); await self.enrich_compartment_localization(); _delta(_s)

            print("--- PHASE 13c: GO-CAM Causal Activity Models ---")
            _s = _snap(); await self.enrich_gocam_activities(); _delta(_s)

            print("--- PHASE 14: Allergen Detection (UniProt KW-0020) ---")
            _s = _snap(); await self.detect_allergen_proteins(); _delta(_s)

            print("--- PHASE 15: Allergen Molecular Impact (CTD + Open Targets) ---")
            _s = _snap(); await self.enrich_allergen_molecular_impact(); _delta(_s)

            print("--- PHASE 16: Allergen-Food Cross-Linking & Kreuzallergie ---")
            _s = _snap(); self.crosslink_allergen_food_sources(); _delta(_s)

            print("--- PHASE 17: Cellular Components + Coding/Non-Coding Gene Mapping ---")
            _s = _snap(); await self.enrich_cellular_components(); _delta(_s)

            print("--- PHASE 18: Electron Density Matrix (RDKit + PySCF) ---")
            _s = _snap(); self.compute_electron_density_matrices(); _delta(_s)

            print("--- PHASE 19: Bioelectric → Disease Signal Pipeline ---")
            _s = _snap(); await self.enrich_bioelectric_disease_signal_pipeline(); _delta(_s)

            if scan_path:
                print("--- PHASE 20: 2D Scan Ingestion ---")
                _s = _snap(); await self.enrich_scan_2d_ingestion(scan_path, modality_hint); _delta(_s)

                print("--- PHASE 21: Spatial Segmentation (SimpleITK) ---")
                _s = _snap(); await self.enrich_scan_segmentation(); _delta(_s)

                print("--- PHASE 22a: UBERON Bridge (SPATIAL_REGION → TISSUE/ORGAN) ---")
                _s = _snap(); await self.enrich_scan_uberon_bridge(); _delta(_s)

                print("--- PHASE 22b: Modality Feature Extraction ---")
                _s = _snap(); await self.enrich_scan_feature_extraction(); _delta(_s)

                print("--- PHASE 22c: Pathology Finding Inference (HPO + Disease) ---")
                _s = _snap(); await self.enrich_pathology_findings(); _delta(_s)

            self.g.print_status_G()
            tissue_hierarchy_map = self.build_tissue_hierarchy_map(cfg)
            return {
                "gutils": self.g,
                "graph": self.g.G,
                "tissue_hierarchy_map": tissue_hierarchy_map,
                "cfg": cfg,
            }

        finally:
            await self.close()





    def visualize_graph(self, dest_path: str | None = None) -> str | None:
        """
        Render the biological knowledge graph as an interactive HTML file.

        Calls firegraph's create_g_visual with ds=False (plain nx graph) so that
        every node type and edge in self.g.G is drawn with its biological style.
        A compact legend overlay is injected automatically.

        Parameters
        ----------
        dest_path : write HTML here if given; otherwise return the HTML string.

        Returns
        -------
        str | None  – HTML content when dest_path is None, else None.
        """
        from firegraph.graph.visual import create_g_visual
        return create_g_visual(self.g.G, dest_path=dest_path, ds=False, add_legend=True)


if __name__ == "__main__":
    import argparse as _argparse
    import os as _os
    import sys as _sys

    # Force UTF-8 output so Unicode arrows in print() don't crash on Windows cp1252
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(_sys.stderr, "reconfigure"):
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    _p = _argparse.ArgumentParser(description="UniprotKB cfg-driven graph build")
    _p.add_argument("--db", default=_os.environ.get("UNIKB_DB", "uniprot"),
                     choices=["uniprot", "pubchem"],
                     help="Data source for seed resolution (env: UNIKB_DB)")
    _p.add_argument("--organs", nargs="*",
                     default=(_os.environ.get("UNIKB_ORGANS", "").split(",")
                              if _os.environ.get("UNIKB_ORGANS") else []),
                     help="Organ terms, space-separated (env: UNIKB_ORGANS, comma-separated)")
    _p.add_argument("--functions", nargs="*",
                     default=(_os.environ.get("UNIKB_FUNCTIONS", "").split(",")
                              if _os.environ.get("UNIKB_FUNCTIONS") else []),
                     help="Function annotation terms (env: UNIKB_FUNCTIONS, comma-separated)")
    _p.add_argument("--outsrc", nargs="*",
                     default=(_os.environ.get("UNIKB_OUTSRC", "").split(",")
                              if _os.environ.get("UNIKB_OUTSRC") else []),
                     help="Harmful / disease exclusion criteria (env: UNIKB_OUTSRC, comma-separated)")
    _p.add_argument("--scan-path", default=None, help="Optional 2D scan file")
    _p.add_argument("--modality-hint", default=None, help="Override scan modality auto-detection")
    _args = _p.parse_args()

    _cfg = {
        "db": _args.db,
        "organs": [o.strip() for o in _args.organs if o.strip()],
        "function_annotation": [f.strip() for f in _args.functions if f.strip()],
        "outsrc_criteria": [o.strip() for o in _args.outsrc if o.strip()],
    }

    g = GUtils()
    kb = UniprotKB(g)

    _result = asyncio.run(kb.finalize_biological_graph(
        _cfg, scan_path=_args.scan_path, modality_hint=_args.modality_hint,
    ))

    _out = "output/graph.html"
    _os.makedirs("output", exist_ok=True)
    kb.visualize_graph(dest_path=_out)
    print(f"Graph HTML written to {_out}")

    if _result and _result.get("tissue_hierarchy_map") is not None:
        import networkx as _nx
        _tm_path = "output/tissue_hierarchy_map.json"
        with open(_tm_path, "w", encoding="utf-8") as _jf:
            json.dump(_nx.node_link_data(_result["tissue_hierarchy_map"]), _jf,
                      ensure_ascii=False, default=str)
        _tm = _result["tissue_hierarchy_map"]
        print(f"Tissue hierarchy map (organ→electron) JSON → {_tm_path}  "
              f"({_tm.number_of_nodes()}N / {_tm.number_of_edges()}E)")