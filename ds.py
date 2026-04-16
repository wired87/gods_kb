"""
ds — project-wide static datasets and shared physical-filter vocabulary.

Prompt: avoid circular imports by storing hardcoded variables (of the entire project)
within this module at the project root.

Prompt: analyze ``filter_physical_compound`` (``str`` or token list), classify tokens
against ``PHYSICAL_CATEGORY_ALIASES`` keys, and expose canonical slot names for
``query_pipe`` and ``UniprotKB`` gating.

CHAR: keep this module free of imports from ``data.*`` or ``query_pipe`` so ``ds`` stays
the acyclic leaf every consumer can import safely.
"""
from __future__ import annotations

import re
ORGANS = [
    "Blood",
    "Brain",
    "Heart",
    "Liver",
    "Kidney",
    "Lung",
    "Muscle",
    "Skeletal muscle",
    "Pancreas",
    "Spleen",
    "Placenta",
    "Testis",
    "Ovary",
    "Uterus",
    "Colon",
    "Small intestine",
    "Stomach",
    "Skin",
    "Adipose tissue",
    "Leukocyte"
]
# ── PHYSICAL COMPONENT FILTER (user aliases → coarse fetch slots) ─────────
# CHAR: same buckets as ``UniprotKB`` physical-layer gating; ontology + disease seeding
# stay on regardless (see ``_physical_enrich_blocked``).
PHYSICAL_CATEGORY_ALIASES: dict[str, str] = {
    "gene": "gene",
    "genes": "gene",
    "protein": "protein",
    "proteins": "protein",
    "organ": "organ",
    "organs": "organ",
    "tissue": "tissue",
    "tissues": "tissue",
    "cell": "cell",
    "cells": "cell",
    "cell_type": "cell",
    "cellular_component": "cellular_component",
    "cellular_components": "cellular_component",
    "cellularcomponent": "cellular_component",
    "compartment": "cellular_component",
    "compartments": "cellular_component",
    "chemical": "chemical",
    "chemicals": "chemical",
    "compound": "chemical",
    "compounds": "chemical",
    "molecule": "molecule",
    "molecules": "molecule",
    "atom": "atom",
    "atoms": "atom",
    "atomic": "atom",
    "food": "food",
    "foods": "food",
    "vitamin": "vitamin",
    "vitamins": "vitamin",
    "fatty_acid": "fatty_acid",
    "fatty_acids": "fatty_acid",
    "fat_acid": "fatty_acid",
    "cofactor": "cofactor",
    "cofactors": "cofactor",
    "co_factor": "cofactor",
    "mineral": "mineral",
    "minerals": "mineral",
}


def cleanup_key_entries(value: str | list[str] | None) -> list[str]:
    """
    Normalize ``filter_physical_compound`` input into raw string tokens (not yet alias-resolved).

    Accepts a comma/semicolon/pipe-separated string or a list of fragments; list elements
    may themselves contain separators.
    """
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        parts = re.split(r"[,;|]+", cleaned)
        return [p.strip() for p in parts if p.strip()]
    out: list[str] = []
    for x in value:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        if re.search(r"[,;|]", s):
            out.extend(cleanup_key_entries(s))
        else:
            out.append(s)
    return out






