"""
FastMCP server for the merged master workflow.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

from uniprot.master_workflow import process_master_query

load_dotenv()

mcp = FastMCP(
    name="Acid Master — Merged Workflow",
    instructions=(
        "Bioinformatics MCP server with a merged peptide and amino-acid workflow. "
        "Use generate_case_fasta(prompt) to let the system route the query into the correct branch."
    ),
)

_DATA_DIR = Path("data")


def _get_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env or environment.")
    return key


@mcp.tool()
async def generate_case_fasta(
    prompt: str,
    workflow_hint: str = "auto",
    max_per_category: int = 25,
    render_top_n: int = 10,
) -> str:
    """
    Run the merged master workflow.

    The input query decides whether the peptide or amino-acid branch is used,
    unless `workflow_hint` is explicitly set to `peptide` or `acid`.
    Returns the generated FASTA content as a string.
    """
    result = await asyncio.to_thread(
        process_master_query,
        user_query=prompt,
        gem_api_key=_get_key(),
        workflow_hint=None if workflow_hint == "auto" else workflow_hint,
        max_per_category=max_per_category,
        render_top_n=render_top_n,
        data_dir=_DATA_DIR,
    )
    artifact_path = Path(result["artifact_path"])
    return artifact_path.read_text(encoding="utf-8")


@mcp.tool()
async def inspect_case(
    prompt: str,
    workflow_hint: str = "auto",
    max_per_category: int = 25,
    render_top_n: int = 10,
) -> dict[str, Any]:
    """
    Run the merged workflow and return the structured result payload.
    """
    return await asyncio.to_thread(
        process_master_query,
        user_query=prompt,
        gem_api_key=_get_key(),
        workflow_hint=None if workflow_hint == "auto" else workflow_hint,
        max_per_category=max_per_category,
        render_top_n=render_top_n,
        data_dir=_DATA_DIR,
    )


@mcp.tool()
async def generate_peptide_fasta(
    prompt: str,
    max_per_category: int = 25,
    render_top_n: int = 10,
) -> str:
    """
    Compatibility alias that forces the peptide branch.
    """
    return await generate_case_fasta(
        prompt=prompt,
        workflow_hint="peptide",
        max_per_category=max_per_category,
        render_top_n=render_top_n,
    )


@mcp.tool()
async def generate_acid_fasta(
    prompt: str,
    max_per_category: int = 25,
    render_top_n: int = 10,
) -> str:
    """
    Compatibility alias that forces the amino-acid branch.
    """
    return await generate_case_fasta(
        prompt=prompt,
        workflow_hint="acid",
        max_per_category=max_per_category,
        render_top_n=render_top_n,
    )

@mcp.tool()
async def create(
    db: str = "uniprot",
    organs: list[str] | None = None,
    function_annotation: list[str] | None = None,
    outsrc_criteria: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build the cfg-driven UniProtKB knowledge graph and return it.

    Parameters
    ----------
    db : "uniprot" or "pubchem" — seed data source.
    organs : organ / tissue terms for hierarchical extraction.
    function_annotation : functional annotation terms.
    outsrc_criteria : harmful / disease exclusion criteria.
    """
    from firegraph.graph import GUtils
    import networkx as nx
    from uniprot_kb import UniprotKB

    cfg = {
        "db": db,
        "organs": organs or [],
        "function_annotation": function_annotation or [],
        "outsrc_criteria": outsrc_criteria or [],
    }
    g = GUtils()
    kb = UniprotKB(g)
    try:
        print("SOLO [1/2] UniprotKB graph build ...")
        build_result = await kb.finalize_biological_graph(cfg)
        n_nodes = g.G.number_of_nodes()
        n_edges = g.G.number_of_edges()
        print(f"SOLO [1/2] DONE — {n_nodes} nodes, {n_edges} edges")

        print("SOLO [2/2] Serialising knowledge graph ...")
        nodes = {
            nid: dict(ndata) for nid, ndata in g.G.nodes(data=True)
        }
        edges = [
            {"src": u, "dst": v, **dict(edata)}
            for u, v, edata in g.G.edges(data=True)
        ]
        print(f"SOLO [2/2] DONE — returning {n_nodes} nodes, {n_edges} edges")

        tissue_mx = build_result.get("tissue_hierarchy_map") if isinstance(build_result, dict) else None
        tissue_json = nx.node_link_data(tissue_mx) if tissue_mx is not None else {"nodes": [], "links": []}

        return {
            "nodes": nodes,
            "edges": edges,
            "tissue_hierarchy_map": tissue_json,
            "stats": {
                "nodes": n_nodes,
                "edges": n_edges,
                "tissue_map_nodes": tissue_mx.number_of_nodes() if tissue_mx else 0,
                "tissue_map_edges": tissue_mx.number_of_edges() if tissue_mx else 0,
            },
        }
    finally:
        await kb.close()


@mcp.tool()
async def visualize(
    db: str = "uniprot",
    organs: list[str] | None = None,
    function_annotation: list[str] | None = None,
    outsrc_criteria: list[str] | None = None,
    dest_path: str = "output/graph.html",
) -> dict[str, Any]:
    """
    Build the cfg-driven UniProtKB knowledge graph and render it as an
    interactive HTML file with a node-type legend.

    Parameters
    ----------
    db : "uniprot" or "pubchem" — seed data source.
    organs : organ / tissue terms for hierarchical extraction.
    function_annotation : functional annotation terms.
    outsrc_criteria : harmful / disease exclusion criteria.
    dest_path : local file path where the HTML will be written.

    Returns
    -------
    dict with 'html_path', 'nodes', 'edges' on success.
    """
    from firegraph.graph import GUtils
    import networkx as nx
    from uniprot_kb import UniprotKB

    cfg = {
        "db": db,
        "organs": organs or [],
        "function_annotation": function_annotation or [],
        "outsrc_criteria": outsrc_criteria or [],
    }
    g = GUtils()
    kb = UniprotKB(g)
    try:
        print("VISUALIZE [1/3] Building UniprotKB graph …")
        build_result = await kb.finalize_biological_graph(cfg)

        n_nodes = g.G.number_of_nodes()
        n_edges = g.G.number_of_edges()
        print(f"VISUALIZE [2/3] Graph built — {n_nodes} nodes, {n_edges} edges")

        print("VISUALIZE [3/3] Rendering HTML …")
        kb.visualize_graph(dest_path=dest_path)
        print(f"VISUALIZE DONE — HTML written to {dest_path}")

        tissue_mx = build_result.get("tissue_hierarchy_map") if isinstance(build_result, dict) else None
        tissue_json = nx.node_link_data(tissue_mx) if tissue_mx is not None else {"nodes": [], "links": []}

        return {
            "html_path": dest_path,
            "tissue_hierarchy_map": tissue_json,
            "stats": {
                "nodes": n_nodes,
                "edges": n_edges,
                "tissue_map_nodes": tissue_mx.number_of_nodes() if tissue_mx else 0,
                "tissue_map_edges": tissue_mx.number_of_edges() if tissue_mx else 0,
            },
        }
    finally:
        await kb.close()


@mcp.tool()
async def classify_scan_2d(
    scan_path: str,
    db: str = "uniprot",
    organs: list[str] | None = None,
    function_annotation: list[str] | None = None,
    outsrc_criteria: list[str] | None = None,
    modality_hint: str = "auto",
) -> dict[str, Any]:
    """
    Classify a 2D medical image (DICOM / NIfTI / PNG) ontologically.

    Builds the cfg-driven knowledge graph (Stages A–C + Phases 2–19),
    then runs the 2D Scan Ontology Pipeline (Phases 20–22c) to segment,
    bridge anatomy to UBERON, extract modality features, and infer
    PATHOLOGY_FINDING → DISEASE links via HPO and cosine similarity.

    Parameters
    ----------
    scan_path : path to the 2D scan file.
    db : "uniprot" or "pubchem" — seed data source.
    organs : organ / tissue terms for hierarchical extraction.
    function_annotation : functional annotation terms.
    outsrc_criteria : harmful / disease exclusion criteria.
    modality_hint : override auto-detection (MRI, CT, PET, ULTRASOUND, …).

    Returns
    -------
    dict with 'findings' (PATHOLOGY_FINDING nodes) and 'disease_links'.
    """
    from firegraph.graph import GUtils
    import networkx as nx
    from uniprot_kb import UniprotKB

    cfg = {
        "db": db,
        "organs": organs or [],
        "function_annotation": function_annotation or [],
        "outsrc_criteria": outsrc_criteria or [],
    }
    hint = None if modality_hint == "auto" else modality_hint
    g = GUtils()
    kb = UniprotKB(g)
    try:
        build_result = await kb.finalize_biological_graph(cfg, scan_path=scan_path, modality_hint=hint)
        tissue_mx = build_result.get("tissue_hierarchy_map") if isinstance(build_result, dict) else None
        tissue_json = nx.node_link_data(tissue_mx) if tissue_mx is not None else {"nodes": [], "links": []}

        findings = {
            nid: dict(ndata)
            for nid, ndata in g.G.nodes(data=True)
            if ndata.get("type") == "PATHOLOGY_FINDING"
        }

        disease_links = [
            {"src": u, "dst": v, **dict(edata)}
            for u, v, edata in g.G.edges(data=True)
            if edata.get("rel") == "INFERRED_DISEASE"
            and g.G.nodes.get(u, {}).get("type") == "PATHOLOGY_FINDING"
        ]

        return {
            "findings": findings,
            "disease_links": disease_links,
            "tissue_hierarchy_map": tissue_json,
            "stats": {
                "nodes": g.G.number_of_nodes(),
                "edges": g.G.number_of_edges(),
                "pathology_findings": len(findings),
                "disease_inferences": len(disease_links),
                "tissue_map_nodes": tissue_mx.number_of_nodes() if tissue_mx else 0,
                "tissue_map_edges": tissue_mx.number_of_edges() if tissue_mx else 0,
            },
        }
    finally:
        await kb.close()


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
