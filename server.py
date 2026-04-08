"""
FastMCP server — Acid Master Peptide Pipeline
Each pipeline step is exposed as its own MCP tool.
Full pipeline: generate_peptide_fasta(prompt) → FASTA text
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv
from fastmcp import FastMCP

from uniprot.peptide_pipeline import (
    _build_fields_param,
    _build_structural_sequence,
    _classify_with_gemini,
    _fetch_live_peptide_fields,
    _fetch_uniprot_peptides,
    _gem_harmony_request,
    _run_pipeline,
    _save_fasta,
)

load_dotenv()

mcp = FastMCP(
    name="Acid Master — Peptide Pipeline",
    instructions=(
        "Bioinformatics MCP server. "
        "Use generate_peptide_fasta(prompt) for the full end-to-end pipeline. "
        "Individual step tools are available for fine-grained control."
    ),
)

_OUTPUT_DIR = Path("output/fasta")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env or environment.")
    return key


def _get_model() -> Any:
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=_get_key())
    return genai.GenerativeModel("gemini-1.5-flash")


def _make_goal(prompt: str, selected_labels: list[str]) -> str:
    return (
        f"Analyze UniProt peptides relevant to: \"{prompt}\". "
        f"Focus on: {', '.join(selected_labels)}. "
        f"Evaluate functional role, catalytic activity, and subcellular location."
    )


# ── Tool 1: live UniProt field catalog ────────────────────────────────────────

@mcp.tool()
async def get_uniprot_fields() -> list[dict]:
    """
    Fetch all live UniProt ft_* peptide feature field definitions.
    Returns field_id, label, and group for each available peptide feature type.
    """
    async with aiohttp.ClientSession() as session:
        fields = await _fetch_live_peptide_fields(session)
    return [{"field_id": f.field_id, "label": f.label, "group": f.group} for f in fields]


# ── Tool 2: classify prompt → 3 categories ────────────────────────────────────

@mcp.tool()
async def classify_query(prompt: str) -> list[dict]:
    """
    Classify a free-text biological query into the 3 most relevant
    UniProt peptide feature categories using Gemini.
    """
    model = _get_model()
    async with aiohttp.ClientSession() as session:
        fields = await _fetch_live_peptide_fields(session)
        selected = await _classify_with_gemini(prompt, fields, model)
    return [{"field_id": f.field_id, "label": f.label, "group": f.group} for f in selected]


# ── Tool 3: fetch peptide entries from UniProt ─────────────────────────────────

@mcp.tool()
async def fetch_peptides(prompt: str, max_per_category: int = 25) -> list[dict]:
    """
    Fetch UniProt peptide entries for the 3 categories most relevant to the prompt.
    Returns accession, name, gene names, category, sequence length, and functional specs.
    """
    model = _get_model()
    async with aiohttp.ClientSession() as session:
        fields = await _fetch_live_peptide_fields(session)
        selected = await _classify_with_gemini(prompt, fields, model)
        fields_param = await _build_fields_param(selected, session)
        peptides = await _fetch_uniprot_peptides(
            selected, session, fields_param, size=max_per_category
        )
    return [
        {
            "accession": p.accession,
            "name": p.name,
            "gene_names": p.gene_names,
            "category": p.category,
            "sequence_length": len(p.sequence),
            "functional_specs": p.functional_specs,
        }
        for p in peptides
    ]


# ── Tool 4: Gemini harmony scoring ────────────────────────────────────────────

@mcp.tool()
async def score_peptide_harmony(prompt: str, max_per_category: int = 25) -> list[dict]:
    """
    Fetch UniProt peptides and rank them by Gemini harmony score
    (relevance to the goal derived from the prompt).
    """
    model = _get_model()
    async with aiohttp.ClientSession() as session:
        fields = await _fetch_live_peptide_fields(session)
        selected = await _classify_with_gemini(prompt, fields, model)
        goal = _make_goal(prompt, [f.label for f in selected])
        fields_param = await _build_fields_param(selected, session)
        peptides = await _fetch_uniprot_peptides(
            selected, session, fields_param, size=max_per_category
        )

    scores = await _gem_harmony_request(goal, peptides, model)
    for p in peptides:
        p.harmony_score = float(scores.get(p.accession, 0.5))
    peptides.sort(key=lambda p: p.harmony_score, reverse=True)

    return [
        {
            "accession": p.accession,
            "name": p.name,
            "gene_names": p.gene_names,
            "category": p.category,
            "harmony_score": p.harmony_score,
            "sequence_length": len(p.sequence),
            "functional_specs": p.functional_specs,
        }
        for p in peptides
    ]


# ── Tool 5: assemble structural sequence ──────────────────────────────────────

@mcp.tool()
async def assemble_structural_sequence(
    prompt: str, max_per_category: int = 25
) -> dict:
    """
    Fetch, score, and logically order all stack peptides via Gemini.
    Returns the concatenated structural sequence and the node order.
    """
    model = _get_model()
    async with aiohttp.ClientSession() as session:
        fields = await _fetch_live_peptide_fields(session)
        selected = await _classify_with_gemini(prompt, fields, model)
        goal = _make_goal(prompt, [f.label for f in selected])
        fields_param = await _build_fields_param(selected, session)
        peptides = await _fetch_uniprot_peptides(
            selected, session, fields_param, size=max_per_category
        )

    scores = await _gem_harmony_request(goal, peptides, model)
    for p in peptides:
        p.harmony_score = float(scores.get(p.accession, 0.5))

    ordered, joined = await _build_structural_sequence(peptides, goal, model)
    return {
        "structural_sequence": joined,
        "total_length": len(joined),
        "node_count": len(ordered),
        "node_order": [
            {"accession": p.accession, "category": p.category, "name": p.name}
            for p in ordered
        ],
    }


# ── Tool 6: full pipeline → FASTA (primary tool) ──────────────────────────────

@mcp.tool()
async def generate_peptide_fasta(
    prompt: str,
    max_per_category: int = 25,
    render_top_n: int = 10,
) -> str:
    """
    Full end-to-end pipeline:
      1. Fetch live UniProt field catalog
      2. Classify prompt → 3 categories (Gemini)
      3. Fetch peptide entries with functional specs
      4. Score stack by Gemini harmony
      5. Assemble structural sequence (Gemini ordering)
      6. Save FASTA + JSON sidecar to output/fasta/
      7. Return FASTA content as string

    Args:
        prompt:             Free-text biological query.
        max_per_category:   Max UniProt results per category (default 25).
        render_top_n:       Top-N peptides rendered to server log (default 10).
    """
    result = await _run_pipeline(
        user_prompt=prompt,
        gem_api_key=_get_key(),
        max_per_category=max_per_category,
        render_top_n=render_top_n,
        output_dir=_OUTPUT_DIR,
    )
    return Path(result["fasta_path"]).read_text(encoding="utf-8")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
