"""
FastMCP server — two tools only: biological generation pipeline and Stripe shopping.

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
"""
from __future__ import annotations

import asyncio
import uuid
import zipfile
from pathlib import Path, PurePath
from tempfile import gettempdir
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

from main import (
    GRAPH_HTML_BASENAME,
    GRAPH_JSON_BASENAME,
    OUTPUT_DIR,
    run_graph_pipeline,
)

load_dotenv()

# CHAR: zipped bundles for MCP clients — one zip per generation session.
_SESSION_ZIP_DIR = Path(gettempdir()) / "acid_master_session_zips"


def _persist_scan_in_session(session_dir: Path, file_bytes: bytes, original_filename: str) -> str:
    """
    Write optional 2D scan into ``session_dir`` and return absolute path for
    ``UniprotKB.finalize_biological_graph(..., scan_path=...)``.
    """
    base = PurePath(original_filename).name.strip()
    if not base or base in (".", ".."):
        raise ValueError("scan_2d_filename must be a non-empty basename (no path segments).")
    if not Path(base).suffix:
        raise ValueError(
            "scan_2d_filename must include a supported extension for ScanIngestionLayer "
            "(e.g. .dcm, .nii, .nii.gz, .png, .tif, .jpg)."
        )
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / base
    if dest.exists():
        dest = session_dir / f"{uuid.uuid4().hex}_{base}"
    dest.write_bytes(file_bytes)
    return str(dest.resolve())


def _zip_session_artifacts(session_id: str, paths: list[str | None]) -> str:
    """Pack existing files into ``<temp>/acid_master_session_zips/<session_id>.zip`` (flat tree per session)."""
    _SESSION_ZIP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _SESSION_ZIP_DIR / f"{session_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if not p:
                continue
            pp = Path(p)
            if not pp.is_file():
                continue
            arc = f"{session_id}/{pp.name}"
            zf.write(pp, arcname=arc)
    return str(zip_path.resolve())


mcp = FastMCP(
    name="Acid Master",
    instructions=(
        "Two tools: `generation` runs query_pipe → UniProt KB graph → ctlr tissue filter; "
        "artifacts are written to `output/<session_id>/` (fixed names graph.html, graph.json, "
        "plus tissue JSON siblings) and a zip of all files is placed under the host temp directory. "
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
    filter_physical_compound: str | None = None,
) -> dict[str, Any]:
    """
    Full biological knowledge workflow: query engine → UniProt KB → ctlr.

    Stages: ``run_query_pipe`` (Gemini) extracts organs plus outsrc / function cues;
    ``UniprotKB.finalize_biological_graph`` builds the hierarchical graph; serialized
    outputs are written under ``output/<session_id>/`` with fixed basenames
    ``graph.html`` and ``graph.json`` (tissue maps use the same stem). ``ctlr.run_controller``
    filters the tissue hierarchy map using the pipe's ``outsrc_criteria`` and
    ``function_annotation`` lists.

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
    filter_physical_compound: list[str] | None — physical components to gate fetches (gene, protein, …).
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
            _persist_scan_in_session,
            session_dir,
            payload,
            scan_2d_filename.strip(),
        )

    result = await run_graph_pipeline(
        prompt=prompt,
        scan_path=scan_path,
        dest_html=dest_html,
        dest_json=dest_json,
        filter_physical_compound=filter_physical_compound,
    )

    paths_for_zip: list[str | None] = [
        result.get("html_path"),
        result.get("json_path"),
        result.get("tissue_hierarchy_json_path"),
        result.get("filtered_tissue_hierarchy_json_path"),
        result.get("scan_path"),
    ]
    zip_path = await asyncio.to_thread(_zip_session_artifacts, session_id, paths_for_zip)
    manifest = [p for p in paths_for_zip if p and Path(p).is_file()]

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
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
