"""
Unit tests for ds — physical filter coercion and classification.

Prompt (user): Make this project a production ready system. Include test files if
needed inside a tests directory.
"""
from __future__ import annotations

import pytest

from ds import (
    PHYSICAL_CATEGORY_ALIASES,
    classify_physical_filter_tokens,
    coerce_physical_filter_tokens,
    resolve_physical_filter_canonical_list,
    resolve_physical_filter_slots,
)


def test_coerce_none_and_empty_string() -> None:
    assert coerce_physical_filter_tokens(None) == []
    assert coerce_physical_filter_tokens("") == []
    assert coerce_physical_filter_tokens("   ") == []


def test_coerce_string_separators() -> None:
    assert coerce_physical_filter_tokens("gene, protein") == ["gene", "protein"]
    assert coerce_physical_filter_tokens("organ;tissue") == ["organ", "tissue"]
    assert coerce_physical_filter_tokens("cell | molecule") == ["cell", "molecule"]


def test_coerce_list_nested_separators() -> None:
    assert coerce_physical_filter_tokens(["a,b", "c"]) == ["a", "b", "c"]


def test_classify_known_aliases() -> None:
    allowed, unknown = classify_physical_filter_tokens(["genes", "Proteins", "cell_type"])
    assert allowed == frozenset({"gene", "protein", "cell"})
    assert unknown == []


def test_classify_unknown_returns_empty_slots() -> None:
    allowed, unknown = classify_physical_filter_tokens(["not_a_real_category"])
    assert allowed is None
    assert unknown == ["not_a_real_category"]


def test_resolve_slots_empty_means_none() -> None:
    assert resolve_physical_filter_slots(None, warn_unknown=False) is None
    assert resolve_physical_filter_slots("", warn_unknown=False) is None


def test_resolve_canonical_list_sorted() -> None:
    out = resolve_physical_filter_canonical_list("tissue, gene", warn_unknown=False)
    assert out == ["gene", "tissue"]


def test_physical_alias_map_covers_string_keys() -> None:
    # CHAR: regression guard — every alias key must be normalizable to lookup form.
    for k in PHYSICAL_CATEGORY_ALIASES:
        assert k == k.strip().lower()
        assert " " not in k
