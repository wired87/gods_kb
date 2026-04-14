"""
Description parsing → cross-reference edges compatible with ``firegraph.graph.GUtils``.

Prompt (user): sub ids in description etc must be recognized and linked to edges following
the standard format (specified in GUtils: ``rel``, ``src_layer``, ``trgt_layer`` on edges).

Prompt (user): descriptions must be embed (save under embedding key) — local sentence-transformers,
same dim policy as ``UniprotKB`` / ``ctlr``.

CHAR: conservative regexes to limit false-positive numeric matches; stubs use ``graph_identity`` rules.
"""
from __future__ import annotations

import asyncio
import math
import re
import threading
from typing import Any

from data.graph_identity import go_term_node_id

# ── embedding (lazy ST — mirrors ``ctlr`` without importing it) ─────────────
_st_model = None
_st_lock = threading.Lock()


def _truncate_renorm_row(row: list[float], dim: int) -> list[float]:
    out = [float(x) for x in row][:dim]
    if len(out) < dim:
        out.extend([0.0] * (dim - len(out)))
    n = math.sqrt(sum(x * x for x in out))
    n = max(n, 1e-12)
    return [x / n for x in out]


def _get_st_model():
    global _st_model
    if _st_model is None:
        with _st_lock:
            if _st_model is None:
                from sentence_transformers import SentenceTransformer
                from uniprot_kb import UniprotKB

                _st_model = SentenceTransformer(UniprotKB._ST_SENTENCE_MODEL)
    return _st_model


async def embed_text_for_node_key(text: str) -> list[float]:
    """CHAR: async wrapper → thread ST encode; key ``embedding`` on graph nodes."""
    from uniprot_kb import UniprotKB

    s = (text or "").strip()
    if not s:
        return [0.0] * UniprotKB._EMBED_DIM
    dim = UniprotKB._EMBED_DIM

    def _run() -> list[float]:
        m = _get_st_model()
        raw = m.encode(
            [s],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return _truncate_renorm_row(raw[0].tolist(), dim)

    return await asyncio.to_thread(_run)


async def apply_embedding_and_description_xrefs(
    gutils: Any,
    nid: str,
    *,
    description: str,
    ntype: str,
    label: str | None = None,
) -> None:
    """
    CHAR: set ``embedding`` from embedding-ready text; wire description CURIE tokens to targets.
    """
    from data.main import graph_node_to_embed_text

    vd = dict(gutils.G.nodes.get(nid, {}))
    vd.setdefault("type", ntype)
    if label:
        vd.setdefault("label", label)
    vd["description"] = description or ""
    text = graph_node_to_embed_text(nid, vd)
    vec = await embed_text_for_node_key(text)
    # CHAR: user-facing ``embedding`` + ``Embedding`` for ``ctlr.TissueGraphController`` parity.
    gutils.update_node({"id": nid, "embedding": vec, "Embedding": vec})
    wire_description_xrefs(gutils, nid, description or "", ntype)


# ── CURIE / accession patterns (conservative) ───────────────────────────────
_GO_RE = re.compile(r"\bGO[:\s]*(\d{7})\b", re.I)
_HP_RE = re.compile(r"\bHP[:\s]*(\d{7})\b", re.I)
_MONDO_RE = re.compile(r"\bMONDO[:\s]*(\d{7})\b", re.I)
_KEGG_PATH_RE = re.compile(r"\b(hsa\d{5})\b", re.I)
_CID_RE = re.compile(r"\b(?:CID|PubChem\s*CID)[:\s#]*(\d{6,})\b", re.I)
_UNIPROT_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)
_GENE_PREFIX_RE = re.compile(r"\bGENE[:\s]+([A-Z][A-Z0-9]{0,15})\b")


def wire_description_xrefs(gutils: Any, src_nid: str, description: str, src_layer: str) -> int:
    """
    CHAR: scan ``description``; add stub targets where missing; ``DESCRIPTION_XREF`` edges.
    Returns edge count created for observability.
    """
    if not description or not src_layer:
        return 0
    g = gutils.G
    added = 0
    src_layer_u = src_layer.upper()

    def _edge(trgt: str, trgt_layer: str, curie: str) -> None:
        nonlocal added
        if not g.has_edge(src_nid, trgt):
            gutils.add_edge(
                src=src_nid,
                trgt=trgt,
                attrs={
                    "rel": "DESCRIPTION_XREF",
                    "src_layer": src_layer_u,
                    "trgt_layer": trgt_layer,
                    "xref_curie": curie,
                },
            )
            added += 1

    for m in _GO_RE.finditer(description):
        curie = f"GO:{m.group(1)}"
        tid = go_term_node_id(curie)
        if not g.has_node(tid):
            gutils.add_node(
                {
                    "id": tid,
                    "type": "GO_TERM",
                    "label": curie,
                    "go_id": curie,
                    "stub": True,
                }
            )
        _edge(tid, "GO_TERM", curie)

    for m in _HP_RE.finditer(description):
        curie = f"HP:{m.group(1)}"
        tid = f"DISEASE_{curie.replace(':', '_')}"
        if not g.has_node(tid):
            gutils.add_node(
                {
                    "id": tid,
                    "type": "DISEASE",
                    "label": curie,
                    "obo_id": curie,
                    "ontology": "HP",
                    "stub": True,
                }
            )
        _edge(tid, "DISEASE", curie)

    for m in _MONDO_RE.finditer(description):
        curie = f"MONDO:{m.group(1)}"
        tid = f"DISEASE_{curie.replace(':', '_')}"
        if not g.has_node(tid):
            gutils.add_node(
                {
                    "id": tid,
                    "type": "DISEASE",
                    "label": curie,
                    "obo_id": curie,
                    "ontology": "MONDO",
                    "stub": True,
                }
            )
        _edge(tid, "DISEASE", curie)

    for m in _KEGG_PATH_RE.finditer(description):
        local = m.group(1).lower()
        curie = f"KEGG:{local}"
        tid = f"KEGG_PATHWAY_{local}"
        if not g.has_node(tid):
            gutils.add_node(
                {
                    "id": tid,
                    "type": "KEGG_PATHWAY",
                    "label": local,
                    "kegg_id": local,
                    "stub": True,
                }
            )
        _edge(tid, "KEGG_PATHWAY", curie)

    for m in _CID_RE.finditer(description):
        cid = m.group(1)
        curie = f"PUBCHEM.COMPOUND:{cid}"
        tid = f"PUBCHEM_CID_{cid}"
        if not g.has_node(tid):
            gutils.add_node(
                {
                    "id": tid,
                    "type": "PUBCHEM_COMPOUND",
                    "label": f"CID {cid}",
                    "pubchem_cid": cid,
                    "stub": True,
                }
            )
        _edge(tid, "PUBCHEM_COMPOUND", curie)

    for m in _UNIPROT_RE.finditer(description):
        acc = m.group(1)
        curie = f"UniProtKB:{acc}"
        if g.has_node(acc):
            _edge(acc, "PROTEIN", curie)

    for m in _GENE_PREFIX_RE.finditer(description):
        sym = m.group(1)
        gid = f"GENE_{sym}"
        if g.has_node(gid):
            _edge(gid, "GENE", f"HGNC:{sym}")

    return added
