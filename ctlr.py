"""
Controller — tissue-scoped graph filter via embeddings and state propagation.

Prompt: adapt the embedding creation process to a local process using the sentence-transformer
lib (365 dims) -> remove cloud based embedding creation.

User prompt (implementation spec):
    Controller
    Create ctlr.py
    ->
    Include class (GOAL: Use the graph representation of the specified tissues to filter based on given outsrc_specs:list[Str], functional_annotations:list[Str]

    include methods:
    - embed_content:
    enbed each entire nodes values in the graph (save under Embedding attr in the Same Node), do the Same for each Item of outsrc_specs, functional_annotations
    - outsrc_nodes:
    The Goal Here IS to use each Item of outsrc_specs(to receive nodes based on their Ontology that result in that outsrc criteria
    Each Node that Matches an outsrc criteria (followig advanced search algorithms like cosing similarity (e.d. maybe more - case specific - you have the choice after carefully validation) set node.state = 1

    - match_functional_nodes:
    use each embed Item of functional_annotation to Perform similarity Search (and other potential improved search algorithms) for each Node of the Graph ->
    results (e.g. reactome , ontology, genes, ptoreins, chemicals and any other)
    must
    be marked with node.state=0 if node.state != 1 or sttae param in onde does not exists.
    For each identified node where state = 0:
    if ontology (disease, funcitonal etc): get all phsical neighbor components (e.g. protein, gene etc.) -> mark them also with state = 0 (if state != 0 or does not exists) -> walk graph to top level of all physical components -> mark them AND their ontology neighbor nodes  in additioin to all of the hierarchial r´compoundes (e.g tissue -> all cells -> all proteins, etc) alwith state=0 (if state != 0 or does not exists);
    else (physical component): mark them AND their ontology neighbor nodes  in additioin to all of the hierarchial r´compoundes (e.g tissue -> all cells -> all proteins, etc) alwith state=0 (if state != 0 or does not exists);

    - extend_functional_nodes:
    create worked_nodes:set
    for each node where state = 0 and id not in worked_nodes:
    include its component name into the static phrase "influence or positive impact on {items identifier/name}" and each within a list[str] -> store each node where state = 0 in worked_nodes -> run the match_functional_nodes call with that list
    Repeat this proces to identify all supporters and positive influencers of the functinnnal_annotations


    - filter_components:
    delte all nodes where state != 0 or does not exists to retrun a graph

    main:
    wrap all of the described functinal wokflows  one by one here and retur the filtered graph from it.
"""
from __future__ import annotations

import asyncio
import math
import os
import threading
from typing import Any

import networkx as nx

try:
    import numpy as np

    _HAS_NP = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NP = False

from data.main import graph_node_to_embed_text
from uniprot_kb import UniprotKB

# CHAR: local ST model + batch/dim — shared with ``UniprotKB`` constants
_ST_MODEL_NAME = UniprotKB._ST_SENTENCE_MODEL
_EMBED_DIM = UniprotKB._EMBED_DIM
_EMBED_BATCH = UniprotKB._EMBED_BATCH

_st_model = None  # lazy ``SentenceTransformer`` (heavy import / weights)
_st_lock = threading.Lock()


def _get_sentence_model():
    """CHAR: load once; ``SentenceTransformer`` is not cheap to construct."""
    global _st_model
    if _st_model is None:
        with _st_lock:
            if _st_model is None:
                from sentence_transformers import SentenceTransformer

                _st_model = SentenceTransformer(_ST_MODEL_NAME)
    return _st_model


def _truncate_renorm_rows(rows: Any, dim: int) -> list[list[float]]:
    """
    CHAR: fixed ``dim`` (365) for cosine pipeline — take head from model output,
    zero-pad if a tiny model ever returned fewer, then L2-normalize per row.
    """
    if not len(rows):
        return []
    out: list[list[float]] = []
    if _HAS_NP:
        for row in rows:
            r = np.asarray(row, dtype=np.float64).ravel()[:dim]
            if r.size < dim:
                r = np.pad(r, (0, dim - int(r.size)))
            n = float(np.linalg.norm(r))
            out.append((r / max(n, 1e-12)).tolist())
        return out
    for row in rows:
        seq = [float(x) for x in row][:dim]
        if len(seq) < dim:
            seq.extend([0.0] * (dim - len(seq)))
        n = math.sqrt(sum(x * x for x in seq))
        n = max(n, 1e-12)
        out.append([x / n for x in seq])
    return out


def _encode_texts_local(texts: list[str]) -> list[list[float]]:
    """CHAR: synchronous encode — call via ``asyncio.to_thread`` from async ctlr methods."""
    if not texts:
        return []
    model = _get_sentence_model()
    raw = model.encode(
        texts,
        batch_size=_EMBED_BATCH,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return _truncate_renorm_rows(raw, _EMBED_DIM)


# CHAR: same cascade universe as tissue hierarchy map (organ → … → molecular)
_CASCADE_TYPES = UniprotKB._ORGAN_TISSUE_CASCADE_TYPES

# CHAR: ontology / annotation / anatomy scaffolding — neighbor rules + hierarchical closure
_ONTOLOGY_LIKE_TYPES = frozenset({
    "DISEASE", "GO_TERM", "REACTOME_PATHWAY", "COMPARTMENT", "CELLULAR_COMPONENT",
    "IMMUNE_RESPONSE", "ORGAN", "ANATOMY_PART", "TISSUE", "TISSUE_2D_LAYER",
    "CELL_POSITION", "CELL_TYPE", "CELL_STATE", "TISSUE_STATE", "ORGAN_STATE",
    "BIOELECTRIC_STATE", "ELECTRICAL_COMPONENT", "ECO_EVIDENCE", "PATHOLOGY_FINDING",
    "SCAN_SIGNAL", "SPATIAL_REGION", "RAW_SCAN",
})

# CHAR: molecular / physical entities — linked from ontology hits
_PHYSICAL_TYPES = frozenset({
    "GENE", "PROTEIN", "PHARMA_COMPOUND", "MINERAL", "MOLECULE_CHAIN", "VMH_METABOLITE",
    "NON_CODING_GENE", "3D_STRUCTURE", "PROTEIN_DOMAIN", "ATOMIC_STRUCTURE", "SEQUENCE_HASH",
    "EXCITATION_FREQUENCY", "MICROBIAL_STRAIN", "FOOD_SOURCE", "ALLERGEN",
})

# CHAR: outsrc similarity targets — ontology-heavy, not bare GENE/PROTEIN
_OUTSRC_CANDIDATE_TYPES = frozenset({
    "DISEASE", "GO_TERM", "REACTOME_PATHWAY", "COMPARTMENT", "CELLULAR_COMPONENT",
    "IMMUNE_RESPONSE", "PATHOLOGY_FINDING",
})


def _l2_normalize_rows(mat: list[list[float]]) -> list[list[float]]:
    """CHAR: row-wise L2 normalize for cosine similarity as dot product."""
    if _HAS_NP and mat:
        a = np.asarray(mat, dtype=np.float64)
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (a / norms).tolist()
    out: list[list[float]] = []
    for row in mat:
        s = math.sqrt(sum(x * x for x in row))
        s = max(s, 1e-12)
        out.append([x / s for x in row])
    return out


def _cosine_top_pairs(
    query_rows: list[list[float]],
    doc_rows: list[list[float]],
    doc_ids: list[str],
    threshold: float,
    top_k: int,
) -> set[str]:
    """
    CHAR: union of {docs with sim >= threshold} and per-query top-k by cosine.
    query_rows / doc_rows must be L2-normalized.
    """
    hits: set[str] = set()
    if not query_rows or not doc_rows or not doc_ids:
        return hits
    if _HAS_NP:
        q = np.asarray(query_rows, dtype=np.float64)
        d = np.asarray(doc_rows, dtype=np.float64)
        sims = q @ d.T  # (n_queries, n_docs)
        for qi in range(sims.shape[0]):
            row = sims[qi]
            above = np.where(row >= threshold)[0]
            for j in above:
                hits.add(doc_ids[int(j)])
            if top_k > 0:
                idx = np.argpartition(-row, min(top_k, len(row)) - 1)[:top_k]
                for j in idx:
                    hits.add(doc_ids[int(j)])
        return hits
    # Pure-Python fallback
    for qi, qv in enumerate(query_rows):
        scores: list[tuple[float, int]] = []
        for j, dv in enumerate(doc_rows):
            s = sum(a * b for a, b in zip(qv, dv))
            scores.append((s, j))
            if s >= threshold:
                hits.add(doc_ids[j])
        if top_k > 0:
            scores.sort(key=lambda x: -x[0])
            for _, j in scores[:top_k]:
                hits.add(doc_ids[j])
    return hits


class TissueGraphController:
    """
    CHAR: embed tissue subgraph nodes, mark outsrc (state=1), functional closure (state=0), then subgraph-keep 0-only.
    """

    def __init__(
        self,
        graph: nx.Graph | nx.MultiGraph,
        outsrc_specs: list[str],
        functional_annotations: list[str],
        *,
        outsrc_cosine_threshold: float = 0.72,
        functional_cosine_threshold: float = 0.65,
        functional_top_k: int = 12,
        max_extend_rounds: int = 4,
        api_key: str | None = None,
    ) -> None:
        # Prompt-aligned params live on the instance for downstream orchestration.
        self.graph = graph
        self.outsrc_specs = [s.strip() for s in (outsrc_specs or []) if str(s).strip()]
        self.functional_annotations = [s.strip() for s in (functional_annotations or []) if str(s).strip()]
        self.outsrc_cosine_threshold = outsrc_cosine_threshold
        self.functional_cosine_threshold = functional_cosine_threshold
        self.functional_top_k = functional_top_k
        self.max_extend_rounds = max(0, int(max_extend_rounds))
        # Kept for API compatibility with ``run_controller(..., api_key=)``; embeddings are local ST only.
        self._api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        self._outsrc_embeddings: list[list[float]] = []
        self._functional_embeddings: list[list[float]] = []

    async def _batch_embed(self, texts: list[str], task_type: str = "") -> list[list[float]]:
        """CHAR: local sentence-transformers encode; ``task_type`` unused (Gemini RETRIEVAL_* removed)."""
        if not texts:
            return []
        return await asyncio.to_thread(_encode_texts_local, texts)

    async def embed_content(self) -> None:
        """CHAR: node Document embeddings → attrs['Embedding']; query embeddings for specs and annotations."""
        node_ids: list[str] = []
        node_texts: list[str] = []
        for nid, attrs in self.graph.nodes(data=True):
            node_ids.append(str(nid))
            node_texts.append(graph_node_to_embed_text(str(nid), dict(attrs)))

        doc_vecs = await self._batch_embed(node_texts, task_type="RETRIEVAL_DOCUMENT")
        for nid, vec in zip(node_ids, doc_vecs):
            self.graph.nodes[nid]["Embedding"] = vec

        if self.outsrc_specs:
            self._outsrc_embeddings = _l2_normalize_rows(
                await self._batch_embed(self.outsrc_specs, task_type="RETRIEVAL_QUERY")
            )
        else:
            self._outsrc_embeddings = []

        if self.functional_annotations:
            self._functional_embeddings = _l2_normalize_rows(
                await self._batch_embed(self.functional_annotations, task_type="RETRIEVAL_QUERY")
            )
        else:
            self._functional_embeddings = []

    def outsrc_nodes(self) -> None:
        """CHAR: cosine match outsrc queries to ontology-class nodes → state=1 (barrier)."""
        if not self._outsrc_embeddings:
            return
        cand_ids: list[str] = []
        cand_vecs: list[list[float]] = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("state") == 1:
                continue
            if data.get("type") not in _OUTSRC_CANDIDATE_TYPES:
                continue
            emb = data.get("Embedding")
            if not emb:
                continue
            cand_ids.append(str(nid))
            cand_vecs.append(list(emb))
        if not cand_ids:
            return
        cand_vecs = _l2_normalize_rows(cand_vecs)
        hit_ids = _cosine_top_pairs(
            self._outsrc_embeddings,
            cand_vecs,
            cand_ids,
            self.outsrc_cosine_threshold,
            top_k=max(3, self.functional_top_k),
        )
        for nid in hit_ids:
            self.graph.nodes[nid]["state"] = 1

    def _try_mark_zero(self, nid: str) -> bool:
        """CHAR: assign state=0 unless outsrc barrier (state=1). Returns True if state is0 after."""
        n = self.graph.nodes[nid]
        if n.get("state") == 1:
            return False
        n["state"] = 0
        return True

    def _propagate_neighbors(self) -> None:
        """
        CHAR: fixpoint — ontology/physical neighbor marking, then cascade-type closure; never cross state=1.
        """
        changed = True
        while changed:
            changed = False
            for nid, data in list(self.graph.nodes(data=True)):
                if data.get("state") != 0:
                    continue
                nt = data.get("type")
                for v in self.graph.neighbors(nid):
                    vd = self.graph.nodes[v]
                    if vd.get("state") == 1:
                        continue
                    vt = vd.get("type")
                    if nt in _ONTOLOGY_LIKE_TYPES and vt in _PHYSICAL_TYPES:
                        if vd.get("state") != 0:
                            vd["state"] = 0
                            changed = True
                    elif nt in _PHYSICAL_TYPES and vt in _ONTOLOGY_LIKE_TYPES:
                        if vd.get("state") != 0:
                            vd["state"] = 0
                            changed = True
            for nid, data in list(self.graph.nodes(data=True)):
                if data.get("state") != 0:
                    continue
                for v in self.graph.neighbors(nid):
                    vd = self.graph.nodes[v]
                    if vd.get("state") == 1:
                        continue
                    if vd.get("type") not in _CASCADE_TYPES:
                        continue
                    if vd.get("state") != 0:
                        vd["state"] = 0
                        changed = True

    async def match_functional_nodes(self, functional_queries: list[str] | None = None) -> None:
        """
        CHAR: similarity seed state=0, then neighbor + hierarchical propagation.
        When functional_queries is set, only those strings are embedded (extension rounds).
        """
        if functional_queries is not None:
            q_texts = [s.strip() for s in functional_queries if str(s).strip()]
            q_vecs = _l2_normalize_rows(
                await self._batch_embed(q_texts, task_type="RETRIEVAL_QUERY")
            ) if q_texts else []
        else:
            q_vecs = self._functional_embeddings

        if not q_vecs:
            self._propagate_neighbors()
            return

        doc_ids: list[str] = []
        doc_vecs: list[list[float]] = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("state") == 1:
                continue
            emb = data.get("Embedding")
            if not emb:
                continue
            doc_ids.append(str(nid))
            doc_vecs.append(list(emb))

        if not doc_ids:
            self._propagate_neighbors()
            return

        doc_vecs = _l2_normalize_rows(doc_vecs)
        hit_ids = _cosine_top_pairs(
            q_vecs,
            doc_vecs,
            doc_ids,
            self.functional_cosine_threshold,
            self.functional_top_k,
        )
        for nid in hit_ids:
            self._try_mark_zero(nid)

        self._propagate_neighbors()

    def _display_name_for(self, nid: str) -> str:
        """CHAR: human-readable token for embedding phrase."""
        d = self.graph.nodes[nid]
        for key in ("label", "go_id", "disease_id"):
            v = d.get(key)
            if v:
                return str(v)
        return str(nid)

    async def extend_functional_nodes(self) -> None:
        """CHAR: phrase expansion → extra match_functional_nodes rounds; worked_nodes avoids repeats."""
        worked_nodes: set[str] = set()
        for _ in range(self.max_extend_rounds):
            batch_nids = [
                str(nid)
                for nid, d in self.graph.nodes(data=True)
                if d.get("state") == 0 and str(nid) not in worked_nodes
            ]
            if not batch_nids:
                break
            phrases: list[str] = []
            for nid in batch_nids:
                worked_nodes.add(nid)
                phrases.append(
                    f"influence or positive impact on {self._display_name_for(nid)}"
                )
            await self.match_functional_nodes(functional_queries=phrases)

    def filter_components(self) -> nx.Graph | nx.MultiGraph:
        """CHAR: keep only nodes with state == 0 (strict functional closure)."""
        keep = {nid for nid, d in self.graph.nodes(data=True) if d.get("state") == 0}
        return self.graph.subgraph(keep).copy()


async def run_controller(
    graph: nx.Graph | nx.MultiGraph,
    outsrc_specs: list[str],
    functional_annotations: list[str],
    **kwargs: Any,
) -> nx.Graph | nx.MultiGraph:
    """
    CHAR: ordered pipeline — embed → outsrc → match → extend → filter.
    """
    ctl = TissueGraphController(graph, outsrc_specs, functional_annotations, **kwargs)
    await ctl.embed_content()
    ctl.outsrc_nodes()
    await ctl.match_functional_nodes()
    await ctl.extend_functional_nodes()
    return ctl.filter_components()


def main(
    graph: nx.Graph | nx.MultiGraph | None = None,
    outsrc_specs: list[str] | None = None,
    functional_annotations: list[str] | None = None,
    **kwargs: Any,
) -> nx.Graph | nx.MultiGraph:
    """
    CHAR: synchronous wrapper; supply graph and spec lists from your workflow (no baked-in example data).
    """
    if graph is None or outsrc_specs is None or functional_annotations is None:
        raise TypeError(
            "ctlr.main() requires graph, outsrc_specs, and functional_annotations."
        )
    return asyncio.run(
        run_controller(graph, outsrc_specs, functional_annotations, **kwargs)
    )
