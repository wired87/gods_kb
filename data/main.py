
from __future__ import annotations

import asyncio
import hashlib
import json
import os

from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote

import warnings

from data._ingest_organ_layer import get_proteins_for_tissue
from data.get_synthetic_proteins import get_synthetic_proteins_for_human

# CHAR: EOL notice is noisy; ``filterwarnings`` message match is brittle — scope ignore to import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)

import httpx
from firegraph.graph import GUtils

from ds import (
    PHYSICAL_CATEGORY_ALIASES,
    cleanup_key_entries,
)
from data._bridge_reactome_nodes import _bridge_reactome_nodes
from data._enrich_go_term_metadata import _enrich_go_term_metadata
from data._wire_gene_go_edges import _wire_gene_go_edges
from data._wire_go_hierarchy import _wire_go_hierarchy
from data.build_tissue_hierarchy_map import build_tissue_hierarchy_map
from data.compute_electron_density_matrices import compute_electron_density_matrices
from data.compute_sequence_hashes import compute_sequence_hashes
from data.crosslink_allergen_food_sources import crosslink_allergen_food_sources
from data.detect_allergen_proteins import detect_allergen_proteins
from data.enrich_allergen_molecular_impact import enrich_allergen_molecular_impact
from data.enrich_bioelectric_disease_signal_pipeline import enrich_bioelectric_disease_signal_pipeline
from data.enrich_bioelectric_properties import enrich_bioelectric_properties
from data.enrich_cell_type_expression import enrich_cell_type_expression
from data.enrich_cellular_components import enrich_cellular_components
from data.enrich_compartment_localization import enrich_compartment_localization
from data.enrich_domain_decomposition import enrich_domain_decomposition
from data.enrich_food_sources import enrich_food_sources
from data.enrich_functional_dynamics import enrich_functional_dynamics
from data.enrich_genomic_data import enrich_genomic_data
from data.enrich_gene_nodes_deep import enrich_gene_nodes_deep
from data.enrich_go_semantic_layer import enrich_go_semantic_layer
from data.enrich_gocam_activities import enrich_gocam_activities
from data.enrich_microbiome_axis import enrich_microbiome_axis
from data.enrich_molecular_structures import enrich_molecular_structures
from data.enrich_pathology_findings import enrich_pathology_findings
from data.enrich_pharmacogenomics import enrich_pharmacogenomics
from data.enrich_pharmacology_quantum_adme import enrich_pharmacology_quantum_adme
from data.enrich_scan_2d_ingestion import enrich_scan_2d_ingestion
from data.enrich_scan_feature_extraction import enrich_scan_feature_extraction
from data.enrich_scan_segmentation import enrich_scan_segmentation
from data.enrich_scan_uberon_bridge import enrich_scan_uberon_bridge
from data.enrich_structural_layer import enrich_structural_layer
from data.enrich_tissue_expression_layer import enrich_tissue_expression_layer
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

# ── MODALITY FEATURE VECTOR DIMENSION: [intensity, contrast, t1/hu/suv_proxy, variance, edge_density] ──
_SCAN_FEAT_DIM = 5

# ── MAX SPATIAL REGIONS per scan (memory guard) ──────────────────
_MAX_SPATIAL_REGIONS = 200


def graph_node_to_embed_text(nid: str, attrs: dict) -> str:
    """
    CHAR: one embedding-ready string per graph node for ``ctlr`` (sentence-transformers).
    Module-level so ``ctlr`` can import without relying on ``UniprotKB._node_to_text`` on the class
    (avoids ``type object 'UniprotKB' has no attribute '_node_to_text'`` on partial/stale loads).
    """
    bits: list[str] = []
    t = attrs.get("type")
    if t:
        bits.append(str(t))
    for v in attrs.values():
        if v is None:
            continue
        s = str(v).strip()
        if s:
            bits.append(s)
    if len(bits) <= 1:
        bits.append(str(nid))
    seen: set[str] = set()
    out: list[str] = []
    for b in bits:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return " | ".join(out)


class UniprotKB:
    _ST_SENTENCE_MODEL = os.environ.get(
        "ST_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    _EMBED_DIM = int(os.environ.get("ST_EMBED_DIM", "365"))
    _EMBED_BATCH = int(os.environ.get("ST_EMBED_BATCH", "32"))

    # ── filter_physical_compound vocabulary (``ds.PHYSICAL_CATEGORY_ALIASES``) ──
    # CHAR: canonical slots are what ``resolve_physical_filter_slots`` / _physical_enrich_blocked use internally.
    PHYSICAL_COMPOUND_FILTER_SLOTS: tuple[str, ...] = tuple(
        sorted(frozenset(PHYSICAL_CATEGORY_ALIASES.values()))
    )

    # CHAR: every accepted alias key (after lower/underscore normalization in the parser); sorted for stable UX/API.
    PHYSICAL_COMPOUND_FILTER_ACCEPTED_TOKENS: tuple[str, ...] = tuple(
        sorted(PHYSICAL_CATEGORY_ALIASES.keys())
    )
    # CHAR: full alias → slot map (module dict; do not mutate at runtime).
    PHYSICAL_COMPOUND_FILTER_ALIASES: dict[str, str] = PHYSICAL_CATEGORY_ALIASES
    _ALLOWED_EXPAND_RELS = frozenset({
        "ENCODED_BY", "MODULATES_TARGET", "HAS_STRUCTURE",
        "PARTICIPATES_IN", "REQUIRES_MINERAL", "CONTAINS_NUTRIENT",
        "INTERACTS_WITH", "ASSOCIATED_WITH_DISEASE",
        "ORGAN_ASSOCIATED_VITAMIN", "ORGAN_ASSOCIATED_FATTY_ACID",
        "ORGAN_ASSOCIATED_COFACTOR", "ORGAN_ASSOCIATED_MINERAL",
        "DESCRIPTION_XREF", "ANNOTATED_WITH", "LOCALIZED_IN", "MAPPED_TO_GO",
        "EXPRESSED_IN_TISSUE", "EXPRESSED_IN_CELL", "SAME_AS",
        "ORGAN_ASSOCIATED_COMPOUND", "PHAROS_CONTEXT_TARGET",
        "ORGAN_KEGG_PATHWAY_BRIDGE", "ORGAN_CTD_CHEMICAL_CONTEXT",
        "CONTAINS_ALLERGEN", "CROSS_REACTIVITY",
    })

    # CHAR: node types kept when extracting organ→tissue→molecular→atomic/electron view
    _ORGAN_TISSUE_CASCADE_TYPES = frozenset({
        "ORGAN", "ANATOMY_PART", "TISSUE", "TISSUE_2D_LAYER", "CELL_POSITION", "CELL_TYPE",
        "DISEASE", "GENE", "PROTEIN", "NON_CODING_GENE",
        "SEQUENCE_HASH", "3D_STRUCTURE", "PROTEIN_DOMAIN",
        "PHARMA_COMPOUND", "MINERAL", "MOLECULE_CHAIN", "VMH_METABOLITE", "FOOD_SOURCE",
        "VITAMIN", "FATTY_ACID", "COFACTOR",
        "ATOMIC_STRUCTURE", "EXCITATION_FREQUENCY", "MICROBIAL_STRAIN",
        "COMPARTMENT", "CELLULAR_COMPONENT", "GO_TERM", "REACTOME_PATHWAY",
        "ECO_EVIDENCE", "ALLERGEN", "IMMUNE_RESPONSE",
    })

    # CHAR: default off — no ``GENE`` / ``CELL_TYPE`` nodes in ``main`` unless enabled.
    workflow_create_gene_nodes: bool = False
    workflow_create_cell_type_nodes: bool = False

    # CHAR: same implementation as module ``graph_node_to_embed_text`` (kept for callers using the class).
    _node_to_text = staticmethod(graph_node_to_embed_text)

    def __init__(self, g):
        self.g = g
        self.client = httpx.AsyncClient(headers=_BROWSER_HEADERS, timeout=60.0)
        # None → all nodes active (legacy behaviour); set → context-driven subgraph
        self.active_subgraph: set[str] | None = None
        # WORKFLOW CONFIG — set by main(...)
        self.workflow_cfg: dict | None = None
        # ORGAN seeds resolved in Stage A; consumed by enrich_tissue_expression_layer Block A
        self._organ_uberon_seeds: list[tuple[str, str]] = []
        # None → all physical enrichment phases run; frozenset → only listed slots fetch new entities
        self._physical_allow: frozenset[str] | None = None

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

    def _node_in_organ_tissue_cascade(self, vd: dict) -> bool:
        """
        BFS gate for ``build_tissue_hierarchy_map``: cascade-typed nodes plus Phase-18
        electron / atomic matrix payloads (not listed as a single ``type`` enum).
        """
        if (vd.get("type") or "") in self._ORGAN_TISSUE_CASCADE_TYPES:
            return True
        if vd.get("electron_density_matrix") is not None:
            return True
        if vd.get("atom_decomposition") is not None:
            return True
        return False


    def _physical_enrich_blocked(self, *slots: str) -> bool:
        """
        True → skip a fetch/enrich phase. Empty *slots → never block (ontology / disease-only paths).
        When self._physical_allow is None, nothing is blocked.
        """
        if not slots:
            return False
        allow = self._physical_allow
        if allow is None:
            return False
        return not any(s in allow for s in slots)

    def apply_eco_weighting(self, entry: dict, protein_node: dict) -> None:
        """CHAR: best-effort ECO tier from UniProt comment/feature evidences → node attrs (pre-add_node)."""
        print("apply_eco_weighting...")
        best_score, best_label = ECO_DEFAULT

        def _take_code(code_raw: str | None) -> None:
            nonlocal best_score, best_label
            code = (code_raw or "").strip()
            if code.startswith("ECO:"):
                sc, lab = ECO_RELIABILITY.get(code, ECO_DEFAULT)
            elif code in _GO_EV_TO_ECO:
                sc, lab = ECO_RELIABILITY.get(_GO_EV_TO_ECO[code], ECO_DEFAULT)
            else:
                return
            if sc > best_score:
                best_score, best_label = sc, lab
        print("pentry comment", entry.get("comments"))
        for block in entry.get("comments") or []:
            for ev in block.get("evidences") or []:
                _take_code(ev.get("evidenceCode"))

        print("pentry features", entry.get("features"))
        for feat in entry.get("features") or []:
            for ev in feat.get("evidences") or []:
                _take_code(ev.get("evidenceCode"))

        protein_node["eco_reliability"] = best_score
        protein_node["eco_evidence_tier"] = best_label
        print("apply_eco_weighting... done")

    # --- CORE INGESTION ---
    async def get_all_proteins(self):
        """Initialer Fetch des menschlichen Proteoms."""
        url = "https://rest.uniprot.org/uniprotkb/search?query=proteome:UP000005640"
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            print("get_all_proteins data", data)
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
                #self._gather_genes(entry, protein_id, graph_nodes, graph_edges)

                self.apply_eco_weighting(entry, protein_node)

            for node in graph_nodes: self.g.add_node(node)
            for edge in graph_edges: self.g.add_edge(**edge)
        except Exception as e:
            print(f"Error in Protein Ingestion: {e}")


    # ══════════════════════════════════════════════════════════════════
    # CONTEXT-DRIVEN GRAPH PIPELINE  (seed → expand → enrich)
    # ══════════════════════════════════════════════════════════════════

    def get_uniprot_url_single_gene(self, accession: str) -> str:
        """REST URL for one UniProtKB entry (JSON). Phase 2 ``enrich_gene_nodes_deep`` deep fetch."""
        acc = (accession or "").strip()
        return f"https://rest.uniprot.org/uniprotkb/{quote(acc)}?format=json"

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
                _pn = {
                    "id": protein_id,
                    "type": "PROTEIN",
                    "label": entry.get("uniProtkbId"),
                    "description": entry.get("proteinDescription", {})
                                        .get("recommendedName", {})
                                        .get("fullName", {})
                                        .get("value"),
                    "taxonId": entry.get("organism", {}).get("taxonId"),
                }
                graph_nodes.append(_pn)
                #self._gather_genes(entry, protein_id, graph_nodes, graph_edges)
                self.apply_eco_weighting(entry, _pn)
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

        from data.enrich_bioelectric_properties import enrich_bioelectric_properties
        from data.enrich_compartment_localization import enrich_compartment_localization
        from data.enrich_food_sources import enrich_food_sources
        from data.enrich_functional_dynamics import enrich_functional_dynamics
        from data.enrich_genomic_data import enrich_genomic_data
        from data.enrich_gene_nodes_deep import enrich_gene_nodes_deep
        from data.enrich_go_semantic_layer import enrich_go_semantic_layer
        from data.enrich_molecular_structures import enrich_molecular_structures
        from data.enrich_pharmacology_quantum_adme import enrich_pharmacology_quantum_adme
        from data.enrich_structural_layer import enrich_structural_layer

        print("  Phase 1: Gene deep enrichment")
        await enrich_gene_nodes_deep(self)

        print("  Phase 2: Functional dynamics (Reactome)")
        await enrich_functional_dynamics(self)

        print("  Phase 3: Pharmacology (ChEMBL + PubChem)")
        await enrich_pharmacology_quantum_adme(self)

        print("  Phase 4: Molecular structures (PubChem/ChEBI)")
        await enrich_molecular_structures(self)

        print("  Phase 5: Genomic + Food + Bioelectric")
        await asyncio.gather(
            enrich_genomic_data(self),
            enrich_food_sources(self),
            enrich_bioelectric_properties(self),
        )

        print("  Phase 6: Structural + GO + Compartments")
        await asyncio.gather(
            enrich_structural_layer(self),
            enrich_go_semantic_layer(self),
            enrich_compartment_localization(self),
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

    async def main(
        self,
        organs: list[str], # brain
        filter_physical_compound: list[str],# chems, protein etc
    ):
        print("start kb main...")

        try:
            ### PROTEINS
            get_proteins_for_tissue(
                self.g,
                organs,
            )

            # SYNTHETIC PROTEINS
            get_synthetic_proteins_for_human(self.g)

            #enrich_bioelectric_properties(self), "protein",

            #compute_electron_density_matrices()

            self.g.print_status_G()
        except Exception as e:
            print("Err kb main:", e)

        finally:
            await self.close()

    async def run_disease_linking_workflow(
            self,
            scan_path=None,
    ):
        # PHASE 10c: Reference DBs (CTD for Gene-Disease-Organ links)
        from data.enrich_ctd_organ_associations import enrich_ctd_organ_associations
        await enrich_ctd_organ_associations(self)

        # PHASE 15: Open Targets (Molecular Impact/Disease Targets)
        await enrich_allergen_molecular_impact(self)

        # PHASE 19: Bioelectric Disease Pipeline
        await enrich_bioelectric_disease_signal_pipeline(self)

        # PHASE 22c: Pathology Finding Inference (only if scan exists)
        if scan_path:
            await enrich_pathology_findings(self)


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





"""
tissue_hierarchy_map = build_tissue_hierarchy_map(self, cfg)
            _allow = self._physical_allow
            _manifest = graph_workflow_api_manifest()
            _secure = {
                "physical_filter_raw_tokens": list(_pf_raw),
                "physical_filter_canonical_slots_resolved": _pf_resolved,
                "physical_filter_unknown_tokens": list(_pf_unknown),
                "physical_gating_active_slots": sorted(_allow) if _allow is not None else None,
                "embedding_model_env": "ST_EMBED_MODEL",
                "embedding_dim_env": "ST_EMBED_DIM",
                "optional_scan_path": bool(scan_path),
                "data_modules_wired": _manifest["main"]["phase_modules"],
            }
            print("--- PHASE 19: Bioelectric → Disease Signal Pipeline ---")
            _s = _snap()
            await enrich_bioelectric_disease_signal_pipeline(self)

            _scan_slots = ("organ", "tissue", "cell", "protein")
            scan_path=None
            if scan_path:
                await _phase(
                    "PHASE 20: 2D Scan Ingestion",
                    lambda: enrich_scan_2d_ingestion(self, scan_path, modality_hint), *_scan_slots,
                )
                await _phase(
                    "PHASE 21: Spatial Segmentation (SimpleITK)",
                    lambda: enrich_scan_segmentation(self), *_scan_slots,
                )
                await _phase(
                    "PHASE 22a: UBERON Bridge (SPATIAL_REGION → TISSUE/ORGAN)",
                    lambda: enrich_scan_uberon_bridge(self), *_scan_slots,
                )
                await _phase(
                    "PHASE 22b: Modality Feature Extraction",
                    lambda: enrich_scan_feature_extraction(self), *_scan_slots,
                )
                await _phase(
                    "PHASE 22c: Pathology Finding Inference (HPO + Disease)",
                    lambda: enrich_pathology_findings(self), *_scan_slots,
                )
            else:
                phase_trace.append({
                    "phase": "PHASE 20-22c: 2D scan subgraph",
                    "need_slots": list(_scan_slots),
                    "status": "skipped",
                    "reason": "no_scan_path",
                    "delta_nodes": 0,
                    "delta_edges": 0,
                })

def _gather_genes(self, entry, protein_id, nodes, edges):
    if not getattr(self, "workflow_create_gene_nodes", False):
        return
    for gene in entry.get("genes", []):
        gene_val = gene.get("geneName", {}).get("value")
        if not gene_val: continue
        gene_node_id = f"GENE_{gene_val}"
        nodes.append({"id": gene_node_id, "type": "GENE", "label": gene_val})
        edges.append({"src": protein_id, "trgt": gene_node_id,
                      "attrs": {"rel": "ENCODED_BY", "src_layer": "PROTEIN", "trgt_layer": "GENE"}})

def _trace_append(
    label: str,
    before: tuple[int, int],
    *,
    need_slots: tuple[str, ...] = (),
    status: str = "completed",
    reason: str | None = None,
    **extra: Any,
) -> None:
    n0, e0 = before
    n1, e1 = self.g.G.number_of_nodes(), self.g.G.number_of_edges()
    row: dict[str, Any] = {
        "phase": label,
        "need_slots": list(need_slots),
        "status": status,
        "delta_nodes": n1 - n0,
        "delta_edges": e1 - e0,
    }
    if reason:
        row["reason"] = reason
    row.update(extra)
    phase_trace.append(row)
    if status == "completed":
        _delta(before)

async def _phase(label: str, run: Callable[[], Awaitable[Any]], *need_slots: str) -> None:
    # CHAR: defer ``run()`` until after the physical-filter guard — otherwise coroutines are
    # constructed then discarded and Python warns "was never awaited".
    if self._physical_enrich_blocked(*need_slots):
        phase_trace.append({
            "phase": label,
            "need_slots": list(need_slots),
            "status": "skipped",
            "reason": "physical_filter",
            "delta_nodes": 0,
            "delta_edges": 0,
        })
        print(f"--- {label} (skipped — physical filter) ---")
        return
    _b = _snap()
    await run()
    _trace_append(label, _b, need_slots=need_slots)


            enrich_pharmacology_quantum_adme()

            await _phase(
                "PHASE 4: Atomic & Molecular Mapping (SMILES)",
                lambda: enrich_molecular_structures(self), "molecule", "atom",
            )
            await _phase(
                "PHASE 5: Nutritional Origin (Open Food Facts DE)",
                lambda: enrich_food_sources(self), "food", "chemical", "molecule",
            )



            _phase_sync("PHASE 6+: Reactome MOL↔PATHWAY Bridge", lambda: _bridge_reactome_nodes(self), "protein")

            await _phase(
                "PHASE 7: Pharmacogenomics (ClinPGx)",
                lambda: enrich_pharmacogenomics(self), "gene",
            )
            
            
await _phase(
                "PHASE 9: Microbiome Metabolism (VMH)",
                lambda: enrich_microbiome_axis(self), "molecule", "chemical",
            )
            if getattr(self, "workflow_create_cell_type_nodes", False):
                await _phase(
                    "PHASE 10: Cellular Integration (HPA + Cell Ontology)",
                    lambda: enrich_cell_type_expression(self), "cell",
                )
            else:
                phase_trace.append({
                    "phase": "PHASE 10: Cellular Integration (HPA + Cell Ontology)",
                    "need_slots": ["cell"],
                    "status": "skipped",
                    "reason": "workflow_create_cell_type_nodes_disabled",
                    "delta_nodes": 0,
                    "delta_edges": 0,
                })
                print(
                    "--- PHASE 10: Cellular Integration (HPA + Cell Ontology) "
                    "(skipped — workflow_create_cell_type_nodes=False) ---",
                )
            await _phase(
                "PHASE 10b: Tissue Integration (HPA + Uberon + CL bridge)",
                lambda: enrich_tissue_expression_layer(self), "tissue", "organ",
            )

            async def _phase10c_reference_databases() -> None:
                from data.enrich_ctd_organ_associations import enrich_ctd_organ_associations
                from data.enrich_kegg_organ_pathways import enrich_kegg_organ_pathways
                from data.enrich_pharos_target_central import enrich_pharos_target_central
                from data.enrich_pubchem_entrez_organ import enrich_pubchem_entrez_organ

                await asyncio.gather(
                    enrich_pubchem_entrez_organ(self),
                    enrich_ctd_organ_associations(self),
                    enrich_pharos_target_central(self),
                    enrich_kegg_organ_pathways(self),
                )

            await _phase(
                "PHASE 10c: Reference DBs (PubChem/Entrez, CTD, Pharos, KEGG)",
                lambda: _phase10c_reference_databases(),
                "organ", "gene", "chemical", "molecule",
            )

            async def _phase10d_pubchem_micronutrients() -> None:
                from data.fetch_pubchem_cofactors_for_organs import fetch_pubchem_cofactors_for_organs
                from data.fetch_pubchem_fatty_acids_for_organs import fetch_pubchem_fatty_acids_for_organs
                from data.fetch_pubchem_minerals_for_organs import fetch_pubchem_minerals_for_organs
                from data.fetch_pubchem_vitamins_for_organs import fetch_pubchem_vitamins_for_organs

                await asyncio.gather(
                    fetch_pubchem_vitamins_for_organs(self),
                    fetch_pubchem_fatty_acids_for_organs(self),
                    fetch_pubchem_cofactors_for_organs(self),
                    fetch_pubchem_minerals_for_organs(self),
                )

            await _phase(
                "PHASE 10d: PubChem micronutrients (vitamins / fatty acids / cofactors / minerals)",
                lambda: _phase10d_pubchem_micronutrients(),
                "organ", "vitamin", "fatty_acid", "cofactor", "mineral", "chemical", "molecule",
            )

            _phase_sync(
                "PHASE 10.5: Sequence Identity Hashing (SHA-256)",
                lambda: compute_sequence_hashes(self), "protein",
            )
            await _phase(
                "PHASE 11: Structural Inference (AlphaFold DB)",
                lambda: enrich_structural_layer(self), "protein",
            )
            await _phase(
                "PHASE 12: Domain Decomposition (InterPro)",
                lambda: enrich_domain_decomposition(self), "protein",
            )

            # CHAR: GO ontology backbone — always on so meanings stay graph-wide
            print("--- PHASE 13a: GO-Semantic-Linking (QuickGO) ---")
            _s = _snap()
            await enrich_go_semantic_layer(self)
            print("--- PHASE 13a+: GO Term Metadata Enrichment ---")
            _s = _snap()
            await _enrich_go_term_metadata(self)
            print("--- PHASE 13a++: GO Ontology Hierarchy (with stub parents) ---")
            _s = _snap()
            await _wire_go_hierarchy(self)

            _phase_sync(
                "PHASE 13a+++: GENE → GO_TERM Derived Edges",
                lambda: _wire_gene_go_edges(self), "gene",
            )

            await _phase(
                "PHASE 13b: Subcellular Localization (COMPARTMENTS)",
                lambda: enrich_compartment_localization(self), "cellular_component",
            )
            await _phase(
                "PHASE 13c: GO-CAM Causal Activity Models",
                lambda: enrich_gocam_activities(self), "protein",
            )
            await _phase(
                "PHASE 14: Allergen Detection (UniProt KW-0020)",
                lambda: detect_allergen_proteins(self), "protein",
            )
            await _phase(
                "PHASE 15: Allergen Molecular Impact (CTD + Open Targets)",
                lambda: enrich_allergen_molecular_impact(self), "protein", "chemical",
            )
            _phase_sync(
                "PHASE 16: Allergen-Food Cross-Linking & Kreuzallergie",
                lambda: crosslink_allergen_food_sources(self), "protein", "chemical", "food",
            )
            await _phase(
                "PHASE 17: Cellular Components + Coding/Non-Coding Gene Mapping",
                lambda: enrich_cellular_components(self), "cellular_component", "gene",
            )
            
"""