"""
shop — Stripe Checkout from JSON: raw commerce spec or ctlr tissue-graph export.

Prompt (server integration): shopping route receives a .json file; docstring hint:
create your generated sequence and order it to our home — powered by Stripe.

Prompt: adapt shopping to accept ctlr workflow output (node-link JSON) and return
checkout plus a sales-facing brief for external users.

Prompt: within shop.py main process include an absorption-parameter that takes either
injection or oral.

CHAR: pricing comes from an optional ``commerce`` / ``checkout`` object and/or
environment variables — graph files carry biology only.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal

AbsorptionRoute = Literal["injection", "oral"]


def _normalize_absorption(raw: str) -> AbsorptionRoute:
    """CHAR: one canonical string for Stripe metadata — reject anything else early."""
    s = str(raw).strip().lower()
    if s == "injection":
        return "injection"
    if s == "oral":
        return "oral"
    raise ValueError('absorption must be "injection" or "oral".')


def _norm_links_key(spec: dict[str, Any]) -> list[Any]:
    """CHAR: node_link_data uses ``links``; some exports use ``edges``."""
    raw = spec.get("links")
    if isinstance(raw, list):
        return raw
    raw = spec.get("edges")
    return raw if isinstance(raw, list) else []


def _extract_ctlr_graph_payload(spec: dict[str, Any]) -> dict[str, Any] | None:
    """
    CHAR: recognize NetworkX node-link JSON (ctlr ``filter_components`` export) or
    a pipeline dict that nests ``filtered_tissue_hierarchy_map``.
    """
    if not isinstance(spec, dict):
        return None
    nested = spec.get("filtered_tissue_hierarchy_map")
    if isinstance(nested, dict) and isinstance(nested.get("nodes"), list):
        if isinstance(_norm_links_key(nested), list):
            return nested
    if isinstance(spec.get("nodes"), list) and isinstance(_norm_links_key(spec), list):
        return spec
    return None


def _display_name_from_node(node: dict[str, Any]) -> str:
    """CHAR: align with ctlr human labels — no graph internals exposed beyond names."""
    attrs = {k: v for k, v in node.items() if k != "id"}
    for key in (
        "label",
        "name",
        "gene_symbol",
        "protein_name",
        "go_id",
        "disease_id",
        "uberon_id",
    ):
        v = attrs.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    nid = node.get("id")
    return str(nid) if nid is not None else "component"


def _build_sales_brief(graph_payload: dict[str, Any]) -> dict[str, Any]:
    """CHAR: compact, customer-safe copy — counts + highlight labels only."""
    nodes = graph_payload.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
    links = _norm_links_key(graph_payload)
    by_type: Counter[str] = Counter()
    highlights: list[str] = []
    seen: set[str] = set()
    priority_types = (
        "PROTEIN",
        "GENE",
        "TISSUE",
        "CELL_TYPE",
        "ORGAN",
        "MOLECULE_CHAIN",
        "PHARMA_COMPOUND",
    )

    for raw in nodes:
        if not isinstance(raw, dict):
            continue
        attrs = {k: v for k, v in raw.items() if k != "id"}
        nt = str(attrs.get("type", "UNKNOWN")).strip() or "UNKNOWN"
        by_type[nt] += 1

    for ptype in priority_types:
        for raw in nodes:
            if not isinstance(raw, dict):
                continue
            attrs = {k: v for k, v in raw.items() if k != "id"}
            if str(attrs.get("type", "")).strip() != ptype:
                continue
            label = _display_name_from_node(raw)
            if label not in seen:
                seen.add(label)
                highlights.append(label)
            if len(highlights) >= 12:
                break
        if len(highlights) >= 12:
            break

    if len(highlights) < 8:
        for raw in nodes:
            if not isinstance(raw, dict):
                continue
            label = _display_name_from_node(raw)
            if label not in seen:
                seen.add(label)
                highlights.append(label)
            if len(highlights) >= 12:
                break

    n_nodes = len(nodes)
    n_links = len(links)
    title = f"Targeted biological design — {n_nodes} curated components"
    return {
        "title": title,
        "scale": {"components": n_nodes, "relationships": n_links},
        "component_mix": dict(sorted(by_type.items(), key=lambda x: -x[1])[:10]),
        "highlights": highlights[:12],
    }


def _merge_commerce(
    spec: dict[str, Any],
    graph_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """CHAR: commerce dict wins; graph may carry embedded ``commerce`` / ``checkout``."""
    block: dict[str, Any] = {}
    for key in ("commerce", "checkout"):
        sub = spec.get(key)
        if isinstance(sub, dict):
            block.update(sub)
    return block


def _resolve_urls(spec: dict[str, Any], commerce: dict[str, Any]) -> tuple[str, str]:
    success = (
        str(
            commerce.get("success_url")
            or spec.get("success_url")
            or os.environ.get("STRIPE_SUCCESS_URL")
            or ""
        )
    ).strip()
    cancel = (
        str(
            commerce.get("cancel_url")
            or spec.get("cancel_url")
            or os.environ.get("STRIPE_CANCEL_URL")
            or ""
        )
    ).strip()
    return success, cancel


def _build_line_items(
    commerce: dict[str, Any],
    spec: dict[str, Any],
    sales: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """CHAR: explicit line_items / price id / env fallback / unit_amount + product title from sales."""
    price_id = (
        commerce.get("stripe_price_id")
        or commerce.get("price_id")
        or spec.get("stripe_price_id")
        or spec.get("price_id")
        or os.environ.get("STRIPE_PRICE_ID")
        or os.environ.get("STRIPE_DEFAULT_PRICE_ID")
    )
    if price_id:
        qty = int(commerce.get("quantity", spec.get("quantity", 1)))
        return [{"price": str(price_id), "quantity": max(1, qty)}]

    raw = commerce.get("line_items") or spec.get("line_items")
    if isinstance(raw, list) and raw:
        line_items: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("Each line_items entry must be an object.")
            name = str(item.get("name", "")).strip()
            unit_amount = item.get("unit_amount")
            quantity = int(item.get("quantity", 1))
            currency = str(item.get("currency", "usd")).lower()
            if not name or unit_amount is None:
                raise ValueError("line_items entries require name and unit_amount.")
            line_items.append(
                {
                    "quantity": max(1, quantity),
                    "price_data": {
                        "currency": currency,
                        "unit_amount": int(unit_amount),
                        "product_data": {"name": name},
                    },
                }
            )
        return line_items

    amt_raw = commerce.get("unit_amount") or os.environ.get("STRIPE_CHECKOUT_UNIT_AMOUNT")
    if amt_raw is not None:
        currency = str(
            commerce.get("currency") or os.environ.get("STRIPE_CHECKOUT_CURRENCY") or "usd"
        ).lower()
        name = str(
            commerce.get("product_name")
            or os.environ.get("STRIPE_CHECKOUT_PRODUCT_NAME")
            or (sales.get("title") if sales else "")
            or "Custom order"
        ).strip()
        desc_bits = (sales or {}).get("highlights") or []
        desc = ", ".join(str(x) for x in desc_bits[:6]) if desc_bits else ""
        pd: dict[str, Any] = {"name": name}
        if desc:
            pd["description"] = desc[:499]
        return [
            {
                "quantity": max(1, int(commerce.get("quantity", spec.get("quantity", 1)))),
                "price_data": {
                    "currency": currency,
                    "unit_amount": int(amt_raw),
                    "product_data": pd,
                },
            }
        ]

    raise ValueError(
        "Set stripe_price_id (or STRIPE_PRICE_ID), line_items, or unit_amount / "
        "STRIPE_CHECKOUT_UNIT_AMOUNT with a ctlr graph export."
    )


def create_checkout_from_order_json(
    order_json_path: str,
    *,
    absorption: str,
) -> dict[str, Any]:
    """
    Create a Stripe Checkout Session from ``order_json_path`` (must end in .json).

    Prompt: within shop.py main process include an absorption-parameter that takes either
    injection or oral.

    **A — ctlr / pipeline graph export** (NetworkX node-link: ``nodes`` + ``links`` or
    ``edges``, optionally nested under ``filtered_tissue_hierarchy_map``): builds a
    customer-facing ``sales`` brief (counts, highlights). Pricing uses optional
    embedded ``commerce`` / ``checkout`` or environment: ``STRIPE_PRICE_ID``,
    ``STRIPE_CHECKOUT_UNIT_AMOUNT``, ``STRIPE_CHECKOUT_CURRENCY``,
    ``STRIPE_CHECKOUT_PRODUCT_NAME``, plus ``STRIPE_SUCCESS_URL`` /
    ``STRIPE_CANCEL_URL`` (or URLs inside ``commerce``).

    **B — legacy commerce-only JSON** — ``stripe_price_id`` / ``line_items`` /
    ``success_url`` / ``cancel_url`` at top level (unchanged).

    Returns ``checkout_url``, ``session_id``, and ``sales`` when the file is a graph
    export (otherwise ``sales`` is null). ``absorption`` is stored on the Checkout
    Session metadata as ``absorption`` (``injection`` or ``oral``).
    """
    import stripe

    absorption_route = _normalize_absorption(absorption)

    key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    if not key:
        raise EnvironmentError("STRIPE_SECRET_KEY is not set.")

    path = Path(order_json_path)
    if path.suffix.lower() != ".json":
        raise ValueError("order_json_path must point to a .json file.")
    if not path.is_file():
        raise FileNotFoundError(f"Order file not found: {path}")

    with path.open(encoding="utf-8") as f:
        spec: dict[str, Any] = json.load(f)

    if not isinstance(spec, dict):
        raise ValueError("JSON root must be an object.")

    graph_payload = _extract_ctlr_graph_payload(spec)
    sales: dict[str, Any] | None = None
    commerce = _merge_commerce(spec, graph_payload)

    if graph_payload is not None:
        sales = _build_sales_brief(graph_payload)

    success_url, cancel_url = _resolve_urls(spec, commerce)
    if not success_url or not cancel_url:
        raise EnvironmentError(
            "success_url and cancel_url must be set on the JSON, under commerce, or via "
            "STRIPE_SUCCESS_URL / STRIPE_CANCEL_URL."
        )

    stripe.api_key = key
    line_items = _build_line_items(commerce, spec, sales)

    session = stripe.checkout.Session.create(
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        line_items=line_items,
        metadata={"absorption": absorption_route},
    )
    out: dict[str, Any] = {
        "checkout_url": session.url,
        "session_id": session.id,
        "sales": sales,
        "absorption": absorption_route,
    }
    return out
