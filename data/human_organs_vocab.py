"""
Canonical human organ / system vocabulary for external-database enrichment steps.

Prompt (user): use local defined list of all humans organs or fetch from specific api —
this module supplies the local list; workflow organ seeds still come from ``finalize_biological_graph``.

CHAR: lowercase tokens for case-insensitive matching against ORGAN ``input_term`` / ``label``.
No sample graph data — vocabulary only.
"""
from __future__ import annotations

# CHAR: broad FMA/UBERON-aligned everyday organ & organ-system names (human context).
HUMAN_ORGAN_CANONICAL_TERMS: tuple[str, ...] = (
    "adipose tissue",
    "adrenal gland",
    "appendix",
    "bladder",
    "blood",
    "bone",
    "bone marrow",
    "brain",
    "breast",
    "cervix",
    "colon",
    "duodenum",
    "ear",
    "esophagus",
    "eye",
    "gall bladder",
    "heart",
    "hypothalamus",
    "kidney",
    "large intestine",
    "larynx",
    "liver",
    "lung",
    "lymph node",
    "muscle",
    "nasal cavity",
    "oral cavity",
    "ovary",
    "pancreas",
    "pharynx",
    "pituitary gland",
    "placenta",
    "prostate",
    "rectum",
    "skin",
    "small intestine",
    "spinal cord",
    "spleen",
    "stomach",
    "testis",
    "thymus",
    "thyroid gland",
    "trachea",
    "ureter",
    "urethra",
    "uterus",
)

_HUMAN_ORGAN_INDEX: frozenset[str] = frozenset(HUMAN_ORGAN_CANONICAL_TERMS)


def is_human_organ_vocab_term(term: str) -> bool:
    """True when normalized ``term`` is one of the canonical organ phrases."""
    t = (term or "").strip().lower()
    return t in _HUMAN_ORGAN_INDEX


def normalize_organ_query(term: str) -> str:
    """Stable display/query token from user or graph organ label."""
    return " ".join((term or "").strip().lower().split())
