"""
FastMCP server — two tools only: biological generation pipeline and Stripe shopping.

Prompt (user): Make this project a production ready system. Include test files if
needed inside a tests directory.

Prompt (user): Live SSE client + hardcoded ``generation`` cases:
``tests/test_mcp_generation_sse_client.py`` (default URL ``http://127.0.0.1:8000/sse``,
same port as Dockerfile ``EXPOSE 8000``).

Prompt: server: extend filter_physical_compound on merged workflow routes (historic).

Prompt: adapt the server.py to include only 2 routes (generation: call the pipe from
query engine -> uniprot -> ctlr) AND shopping route which receives a .json file, and
include a docstring hint: create your generated sequence and order it to our home -
powered by stripe.

Prompt: adapt shopping to accept ctlr workflow output JSON and return a sales-facing
brief alongside Stripe checkout URLs.

Prompt: remove scan_path and modality_hint from the server input params. include a 2d scan file
(optional and refer it in docstring comment -> save it in temp store -> provide paths to
uniprotkb.finalize_biological_graph)

Prompt: Session-scoped outputs: hardcoded ``graph.html`` / ``graph.json`` (see ``main`` constants)
under ``output/<session_id>/``; optional 2D scan bytes are written into the same session folder;
after the pipeline, all created files are zipped under the host temp dir and returned paths
include ``session_id``, ``artifacts_uri_prefix``, ``artifacts_zip_path``, and ``artifacts_manifest``.

Prompt (user): ``filter_physical_compound`` aligns with ``ds.PHYSICAL_CATEGORY_ALIASES``; pipeline
returns ``workflow_api_manifest`` / ``graph_api_validation`` for contract checks.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from fastmcp import FastMCP

from ds import coerce_physical_filter_tokens

from main import (
    GRAPH_HTML_BASENAME,
    GRAPH_JSON_BASENAME,
    OUTPUT_DIR,
    run_graph_pipeline,
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
    scan_2d_file: bytes | None = None,
    scan_2d_filename: str | None = None,
    filter_physical_compound: str | list[str] | None = None,
    result_specs: Sequence[dict[str, Any]] | None = None,
    design_include_peptide_chains: bool = False,
) -> dict[str, Any]:
    """
    Full biological knowledge workflow: query engine → UniProt KB → ctlr.

    Stages: ``run_query_pipe`` (Gemini) extracts organs plus outsrc / function cues;
    ``UniprotKB.finalize_biological_graph`` builds the hierarchical graph; serialized
    outputs are written under ``output/<session_id>/`` with fixed basenames
    ``graph.html`` and ``graph.json`` (tissue maps use the same stem).     ``ctlr.run_controller``
    filters the tissue hierarchy map using the pipe's ``outsrc_criteria`` and
    ``function_annotation`` lists. ``designer.Designer`` then writes ``design_artifacts/``
    under the same session directory (per-node-type JSON, blueprint, manifest).

    ``result_specs``: optional list of dicts like
    ``{"target_organs": ["liver"], "primary_route": "oral", "cited_guidance": []}``
    (routes: ``intramuscular``, ``intravenous``, ``oral``). Omit or pass ``[]`` for blueprint-only output.

    ``filter_physical_compound`` matches ``UniprotKB`` physical-layer gating: coarse
    class tokens (gene, protein, organ, tissue, …); empty list means full workflow;
    non-empty limits physical fetch phases while ontology and disease context remain.
    If unset and ``UNIKB_PHYSICAL_FILTER`` is defined (comma-separated), that list is used.

    Optional 2D medical scan: pass ``scan_2d_file`` (raw bytes) together with
    ``scan_2d_filename`` (original basename with extension, e.g. ``study.dcm``). The server
    writes the payload under ``output/<session_id>/`` and passes the absolute path as
    ``scan_path`` into ``UniprotKB.finalize_biological_graph``; modality is inferred from
    file metadata in ``ScanIngestionLayer``.

    Returns the pipeline dict plus ``session_id``, ``session_artifact_dir``,
    ``artifacts_uri_prefix`` (``output/<session_id>``), ``artifacts_zip_path``, and
    ``artifacts_manifest`` (existing files that were packed).

    parameters:
    prompt: str — describes the wanted outcome, constraints, toxicity flags, diseases to avoid.
    scan_2d_file: bytes | None — optional 2D scan payload (DICOM / NIfTI / raster per loader).
    scan_2d_filename: str | None — required when ``scan_2d_file`` is set; basename + extension only.
    filter_physical_compound: str | list[str] | None — physical components to gate fetches (gene, protein, …);
        comma/semicolon-separated string or token list (``ds.PHYSICAL_CATEGORY_ALIASES``).
    result_specs: list[dict] | None — organ delivery specs for designer formulation overlay (see above).
    design_include_peptide_chains: bool — when true, include ``MOLECULE_CHAIN`` nodes in formulation merge.
    """
    session_id = uuid.uuid4().hex
    session_dir = (OUTPUT_DIR / session_id).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    dest_html = str(session_dir / GRAPH_HTML_BASENAME)
    dest_json = str(session_dir / GRAPH_JSON_BASENAME)

    scan_path: str | None = None
    cond_file = scan_2d_file is not None
    cond_name = bool((scan_2d_filename or "").strip())
    if cond_file ^ cond_name:
        raise ValueError("Provide both scan_2d_file and scan_2d_filename, or omit both.")
    if cond_file and cond_name:
        raw = scan_2d_file
        if isinstance(raw, (bytes, bytearray)):
            payload = bytes(raw)
        else:
            payload = bytes(memoryview(raw))
        if not payload:
            raise ValueError("scan_2d_file must be non-empty when provided.")
        scan_path = await asyncio.to_thread(
            persist_scan_in_session,
            session_dir,
            payload,
            scan_2d_filename.strip(),
        )

    _fp = filter_physical_compound
    if isinstance(_fp, str):
        _fp_list: list[str] | None = coerce_physical_filter_tokens(_fp) or None
    elif _fp is None:
        _fp_list = None
    else:
        _fp_list = coerce_physical_filter_tokens(list(_fp)) or None

    design_dir = session_dir / "design_artifacts"
    _rs = list(result_specs) if result_specs else None
    result = await run_graph_pipeline(
        prompt=prompt,
        scan_path=scan_path,
        dest_html=dest_html,
        dest_json=dest_json,
        filter_physical_compound=_fp_list,
        design_output_dir=str(design_dir),
        result_specs=_rs,
        design_include_peptide_chains=design_include_peptide_chains,
    )

    paths_for_zip: list[str | None] = [
        result.get("html_path"),
        result.get("json_path"),
        result.get("tissue_hierarchy_json_path"),
        result.get("filtered_tissue_hierarchy_json_path"),
        result.get("scan_path"),
        result.get("design_manifest_path"),
    ]
    _dap = result.get("design_artifact_paths")
    if isinstance(_dap, dict):
        paths_for_zip.extend([p for p in _dap.values() if isinstance(p, str)])
    # CHAR: manifest is both ``design_manifest_path`` and ``__manifest__`` in artifact map — dedupe.
    _zip_seen: set[str] = set()
    _zip_list: list[str | None] = []
    for p in paths_for_zip:
        if not p or not isinstance(p, str):
            continue
        if p in _zip_seen:
            continue
        _zip_seen.add(p)
        _zip_list.append(p)
    zip_path = await asyncio.to_thread(zip_session_artifacts, session_id, _zip_list)
    manifest = [p for p in _zip_list if p and Path(p).is_file()]

    out = dict(result)
    out["session_id"] = session_id
    out["session_artifact_dir"] = str(session_dir)
    out["artifacts_uri_prefix"] = f"output/{session_id}"
    out["artifacts_zip_path"] = zip_path
    out["artifacts_manifest"] = manifest
    return out


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
