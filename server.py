
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from fastmcp import FastMCP

from ds import cleanup_key_entries

from main import (
    GRAPH_HTML_BASENAME,
    GRAPH_JSON_BASENAME,
    OUTPUT_DIR,
    main_workflow,
)
from session_artifacts import persist_scan_in_session, zip_session_artifacts

load_dotenv()


mcp = FastMCP(
    name="Acid Master",
    instructions=(
        "Two tools: `generation` runs query_pipe → UniProt KB graph → ctlr tissue filter → designer "
        "(substrate blueprint + optional organ/route `result_specs`); "
        "artifacts are written to `output/<session_id>/` (fixed names graph.html, graph.json, "
        "plus tissue JSON siblings and `design_artifacts/`) and a zip of all files is placed under the host temp directory. "
        "Optional `scan_2d_file` + `scan_2d_filename` store the scan inside the same session folder. "
        "`shopping` opens Stripe Checkout from ctlr graph JSON (filtered tissue map) or a commerce JSON; "
        "response includes `sales` for external buyers when a graph export is used."
    ),
)


@mcp.tool()
async def generation(
    prompt: str,
    organs:list[str] or None = None,
    filter_physical_compound: str | list[str] | None = None,
    result_specs: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    print("start generation...")
    session_id = uuid.uuid4().hex
    session_dir = (OUTPUT_DIR / session_id).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    dest_html = str(session_dir / GRAPH_HTML_BASENAME)
    dest_json = str(session_dir / GRAPH_JSON_BASENAME)

    _fp_list = cleanup_key_entries(list(filter_physical_compound)) or None

    design_dir = session_dir / "design_artifacts"
    _rs = list(result_specs) if result_specs else None
    result = await main_workflow(
        prompt=prompt,
        dest_html=dest_html,
        dest_json=dest_json,
        filter_physical_compound=_fp_list,
        design_output_dir=str(design_dir),
        result_specs=_rs,
        organs_override=organs
    )
    print("Err")
    return result


@mcp.tool()
async def shopping(order_json_path: str, absorption: str) -> dict[str, Any]:
    """
    Start Stripe Checkout from a JSON file on disk.

    ``absorption``: ``injection`` or ``oral`` — forwarded to checkout session metadata.

    Primary input: **ctlr workflow export** — NetworkX node-link data (``nodes`` +
    ``links`` / ``edges``) from ``run_controller`` / ``filtered_tissue_hierarchy_map``,
    or the same nested under ``filtered_tissue_hierarchy_map`` in a pipeline payload.
    Pricing can sit in optional ``commerce`` / ``checkout`` on that file or in env
    (``STRIPE_PRICE_ID``, ``STRIPE_CHECKOUT_UNIT_AMOUNT``, redirect URLs).

    Returns ``checkout_url``, ``session_id``, and ``sales`` (title, scale, component_mix,
    highlights) when a graph file is used — stripped for external catalog copy; no raw
    graph dump. Legacy flat commerce JSON (price id / ``line_items`` only) still works;
    then ``sales`` is null.

    Hint: create your generated sequence (or export the ctlr-filtered tissue JSON from
    the generation step), save it as JSON, point ``order_json_path`` at that file, and
    complete payment to ship to our home — checkout is powered by Stripe.
    """
    from shop import create_checkout_from_order_json

    return await asyncio.to_thread(
        create_checkout_from_order_json,
        order_json_path,
        absorption=absorption,
    )


if __name__ == "__main__":
    _host = os.environ.get("ACID_MASTER_HOST", "0.0.0.0")
    _port = int(os.environ.get("ACID_MASTER_PORT", "8000"))
    _level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("acid_master.server")
    log.info("Starting Acid Master MCP (sse) on %s:%s", _host, _port)
    mcp.run(transport="sse", host=_host, port=_port)
