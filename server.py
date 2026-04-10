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
async def solo() -> dict[str, Any]:
    """
    Build the full 18-phase UniProtKB knowledge graph and return it.
    """
    from firegraph.graph import GUtils
    from uniprot_kb import UniprotKB

    g = GUtils()
    kb = UniprotKB(g)
    try:
        print("SOLO [1/2] UniprotKB graph build ...")
        await kb.finalize_biological_graph()
        n_nodes = g.G.number_of_nodes()
        n_edges = g.G.number_of_edges()
        print(f"SOLO [1/2] DONE — {n_nodes} nodes, {n_edges} edges")

        # SERIALISE KNOWLEDGE GRAPH
        print("SOLO [2/2] Serialising knowledge graph ...")
        nodes = {
            nid: dict(ndata) for nid, ndata in g.G.nodes(data=True)
        }
        edges = [
            {"src": u, "dst": v, **dict(edata)}
            for u, v, edata in g.G.edges(data=True)
        ]
        print(f"SOLO [2/2] DONE — returning {n_nodes} nodes, {n_edges} edges")

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {"nodes": n_nodes, "edges": n_edges},
        }
    finally:
        await kb.close()


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
