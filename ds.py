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
from typing import Iterable

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


def coerce_physical_filter_tokens(value: str | list[str] | None) -> list[str]:
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
            out.extend(coerce_physical_filter_tokens(s))
        else:
            out.append(s)
    return out


def classify_physical_filter_tokens(raw_tokens: Iterable[str]) -> tuple[frozenset[str] | None, list[str]]:
    """
    Map raw tokens to canonical ``PHYSICAL_CATEGORY_ALIASES`` values.

    Returns ``(allowed_slots, unknown_raw)`` where ``allowed_slots`` is ``None`` when
    no valid token was recognized (caller treats as full workflow, same as empty filter).
    """
    slots: set[str] = set()
    unknown: list[str] = []
    for raw in raw_tokens:
        key = " ".join(str(raw).strip().lower().replace("-", " ").split())
        if not key:
            continue
        key_us = key.replace(" ", "_")
        canon = PHYSICAL_CATEGORY_ALIASES.get(key_us)
        if canon:
            slots.add(canon)
        else:
            unknown.append(str(raw))
    if not slots:
        return None, unknown
    return frozenset(slots), unknown


def resolve_physical_filter_slots(
    filter_physical_compound: str | list[str] | None,
    *,
    warn_unknown: bool = True,
) -> frozenset[str] | None:
    """
    Return allowed internal slots, or ``None`` when input is empty or wholly unrecognized
    (full workflow — same semantics as the former ``data.main`` helper).
    """
    tokens = coerce_physical_filter_tokens(filter_physical_compound)
    if not tokens:
        return None
    allowed, unknown = classify_physical_filter_tokens(tokens)
    if warn_unknown:
        for u in unknown:
            print(f"  WARN: unknown filter_physical_compound token {u!r} — ignored")
    return allowed


def resolve_physical_filter_canonical_list(
    filter_physical_compound: str | list[str] | None,
    *,
    warn_unknown: bool = True,
) -> list[str]:
    """Stable sorted list of canonical physical slots; empty list ⇒ no gating / inherit full workflow upstream."""
    fs = resolve_physical_filter_slots(
        filter_physical_compound,
        warn_unknown=warn_unknown,
    )
    return sorted(fs) if fs else []
