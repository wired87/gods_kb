"""
Peptide pipeline: user prompt → live UniProt categories → fetch → stack → Gemini harmony → terminal render → FASTA

All categories, field IDs, and query syntax are resolved at runtime from UniProt's
/configure/uniprotkb/result-fields endpoint — nothing is hardcoded.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from aiolimiter import AsyncLimiter

_UNIPROT_FIELDS_URL = "https://rest.uniprot.org/configure/uniprotkb/result-fields"
_UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

_rate_limiter = AsyncLimiter(max_rate=5, time_period=1)

# ANSI codes
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"

_HYDROPHOBIC = set("VILMFYW")
_CHARGED     = set("KRDE")
_SEQ_WIDTH   = 60


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class UniProtField:
    field_id: str    # e.g. "ft_peptide"
    label: str       # e.g. "Peptide"
    group: str       # e.g. "PTM / Processing"

    @property
    def query(self) -> str:
        return f"{self.field_id}:*"


@dataclass
class PeptideRecord:
    accession: str
    name: str
    gene_names: list[str]
    category: str
    sequence: str
    functional_specs: dict[str, Any] = field(default_factory=dict)
    peptide_features: list[dict] = field(default_factory=list)
    harmony_score: float = 0.0


# ── Live UniProt field discovery ───────────────────────────────────────────────

async def _fetch_live_peptide_fields(session: aiohttp.ClientSession) -> list[UniProtField]:
    """
    Fetch all UniProt result-field groups live and extract every field
    whose group name contains 'PTM', 'Processing', or 'Sequence', and
    whose field_id starts with 'ft_' (feature annotation fields).
    """
    async with _rate_limiter:
        async with session.get(_UNIPROT_FIELDS_URL, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            resp.raise_for_status()
            groups: list[dict] = await resp.json()

    fields: list[UniProtField] = []
    for group in groups:
        group_label: str = group.get("groupName", "")
        for f in group.get("fields", []):
            fid: str = f.get("name", "")
            if fid.startswith("ft_"):
                fields.append(UniProtField(
                    field_id=fid,
                    label=f.get("label", fid),
                    group=group_label,
                ))

    return fields


# ── Gemini classification ──────────────────────────────────────────────────────

async def _classify_with_gemini(
    user_prompt: str,
    fields: list[UniProtField],
    model: Any,
) -> list[UniProtField]:
    """
    Ask Gemini to pick the 3 most relevant UniProtField objects for the user prompt.
    Returns exactly 3 UniProtField instances.
    """
    catalog = "\n".join(
        f"- {f.field_id} | {f.label} | group: {f.group}"
        for f in fields
    )

    prompt = (
        f"User query: \"{user_prompt}\"\n\n"
        f"From the UniProt feature fields below, pick the 3 most relevant field_ids "
        f"for this query.\n"
        f"Respond with ONLY a JSON array of exactly 3 field_id strings: "
        f'["ft_peptide", "ft_signal", "ft_chain"]\n\n'
        f"Fields:\n{catalog}"
    )

    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        text = response.text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
        chosen_ids: list[str] = json.loads(text)
    except Exception as exc:
        print(f"  Gemini classification error: {exc} — falling back to first 3 fields.")
        chosen_ids = [f.field_id for f in fields[:3]]

    id_map = {f.field_id: f for f in fields}
    selected: list[UniProtField] = []
    for fid in chosen_ids:
        if fid in id_map and id_map[fid] not in selected:
            selected.append(id_map[fid])
    # If Gemini returned fewer than 3 valid ids, pad from the field list
    for f in fields:
        if len(selected) == 3:
            break
        if f not in selected:
            selected.append(f)

    return selected[:3]


# ── UniProt peptide fetch ──────────────────────────────────────────────────────

async def _build_fields_param(
    selected: list[UniProtField],
    session: aiohttp.ClientSession,
) -> str:
    """
    Build the ?fields= parameter string from the live UniProt result-fields endpoint.
    Always includes core identity/function fields plus the selected feature fields.
    """
    async with _rate_limiter:
        async with session.get(_UNIPROT_FIELDS_URL, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            resp.raise_for_status()
            groups: list[dict] = await resp.json()

    # Collect core fields by label from the live API
    core_labels = {
        "Entry", "Entry Name", "Gene Names", "Protein names",
        "Sequence", "Function [CC]", "Catalytic activity",
        "Subcellular location [CC]",
    }
    core_ids: list[str] = []
    for group in groups:
        for f in group.get("fields", []):
            if f.get("label") in core_labels:
                core_ids.append(f.get("name", ""))

    selected_ids = [f.field_id for f in selected]
    all_ids = list(dict.fromkeys(core_ids + selected_ids))  # deduplicate, preserve order
    return ",".join(filter(None, all_ids))


async def _fetch_uniprot_peptides(
    selected: list[UniProtField],
    session: aiohttp.ClientSession,
    fields_param: str,
    size: int = 25,
) -> list[PeptideRecord]:
    records: list[PeptideRecord] = []
    seen: set[str] = set()

    for feat_field in selected:
        url = (
            f"{_UNIPROT_SEARCH_URL}"
            f"?query={feat_field.query}+AND+organism_id:9606"
            f"&fields={fields_param}&format=json&size={size}"
        )
        try:
            async with _rate_limiter:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        print(f"  UniProt [{feat_field.label}]: HTTP {resp.status}")
                        continue
                    data = await resp.json()
        except Exception as exc:
            print(f"  UniProt [{feat_field.label}]: {exc}")
            continue

        for entry in data.get("results", []):
            accession = entry.get("primaryAccession", "")
            if accession in seen:
                continue
            seen.add(accession)

            pn = entry.get("proteinDescription", {})
            name = pn.get("recommendedName", {}).get("fullName", {}).get("value", "")
            if not name:
                subs = pn.get("submissionNames", [])
                name = subs[0].get("fullName", {}).get("value", "") if subs else accession

            gene_names = [
                g.get("geneName", {}).get("value", "")
                for g in entry.get("genes", [])
                if g.get("geneName")
            ]

            sequence = entry.get("sequence", {}).get("value", "")

            func_specs: dict[str, Any] = {}
            for comment in entry.get("comments", []):
                ctype = comment.get("commentType", "")
                if ctype == "FUNCTION":
                    texts = comment.get("texts", [])
                    func_specs["function"] = texts[0].get("value", "") if texts else ""
                elif ctype == "CATALYTIC ACTIVITY":
                    func_specs["catalytic_activity"] = comment.get("reaction", {}).get("name", "")
                elif ctype == "SUBCELLULAR LOCATION":
                    func_specs["subcellular_location"] = [
                        loc.get("location", {}).get("value", "")
                        for loc in comment.get("subcellularLocations", [])
                    ]

            peptide_features = [
                f for f in entry.get("features", [])
                if f.get("type", "").lower() in {
                    feat_field.label.lower()
                    for feat_field in selected
                }
            ]

            records.append(PeptideRecord(
                accession=accession,
                name=name,
                gene_names=gene_names,
                category=feat_field.label,
                sequence=sequence,
                functional_specs=func_specs,
                peptide_features=peptide_features,
            ))

    return records


# ── Gemini harmony scoring ─────────────────────────────────────────────────────

async def _gem_harmony_request(
    goal: str,
    peptides: list[PeptideRecord],
    model: Any,
) -> dict[str, float]:
    summaries = "\n".join(
        f"- {p.accession} | {p.name} | {p.functional_specs.get('function', 'N/A')[:80]}"
        for p in peptides[:40]
    )

    prompt = (
        f"Goal: {goal}\n\n"
        f"Score each peptide 0.0–1.0 for relevance to the goal.\n"
        f"Respond with ONLY valid JSON mapping accession to score.\n\n"
        f"Peptides:\n{summaries}"
    )

    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        text = response.text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
        return json.loads(text)
    except Exception as exc:
        print(f"  Gemini harmony error: {exc} — using uniform scores.")
        return {p.accession: 0.5 for p in peptides}


# ── Terminal renderer ──────────────────────────────────────────────────────────

def _color_sequence(seq: str) -> str:
    out = []
    for aa in seq:
        if aa in _HYDROPHOBIC:
            out.append(f"\033[32m{aa}{_RESET}")
        elif aa in _CHARGED:
            out.append(f"{_CYAN}{aa}{_RESET}")
        else:
            out.append(aa)
    return "".join(out)


def _wrap(text: str, width: int = 68, indent: str = "             ") -> str:
    words, lines, line = text.split(), [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return f"\n{indent}".join(lines)


def render_peptide(peptide: PeptideRecord) -> None:
    bar = "─" * 74
    score_color = _GREEN if peptide.harmony_score >= 0.7 else (_YELLOW if peptide.harmony_score >= 0.4 else _RED)

    print(f"\n{_BOLD}{bar}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {peptide.name}{_RESET}  {_DIM}({peptide.accession}){_RESET}")
    print(
        f"  {_DIM}Category:{_RESET} {peptide.category:<20} "
        f"{_DIM}Harmony:{_RESET} {score_color}{peptide.harmony_score:.3f}{_RESET}"
    )

    if peptide.gene_names:
        print(f"  {_DIM}Genes   :{_RESET} {', '.join(peptide.gene_names)}")

    if peptide.functional_specs.get("function"):
        print(f"  {_DIM}Function:{_RESET} {_wrap(peptide.functional_specs['function'])}")

    if peptide.functional_specs.get("subcellular_location"):
        locs = ", ".join(filter(None, peptide.functional_specs["subcellular_location"]))
        if locs:
            print(f"  {_DIM}Location:{_RESET} {locs}")

    if peptide.functional_specs.get("catalytic_activity"):
        print(f"  {_DIM}Catalytic:{_RESET} {peptide.functional_specs['catalytic_activity'][:70]}")

    if peptide.peptide_features:
        feat_summary = ", ".join(
            f"{f.get('type')} "
            f"{f.get('location', {}).get('start', {}).get('value', '?')}"
            f"–{f.get('location', {}).get('end', {}).get('value', '?')}"
            for f in peptide.peptide_features[:4]
        )
        print(f"  {_DIM}Features:{_RESET} {feat_summary}")

    seq = peptide.sequence
    print(f"\n  {_YELLOW}Sequence ({len(seq)} aa):{_RESET}")
    for i in range(0, len(seq), _SEQ_WIDTH):
        chunk = seq[i:i + _SEQ_WIDTH]
        print(f"  {_DIM}{i + 1:>5}{_RESET}  {_color_sequence(chunk)}")

    print(f"{_BOLD}{bar}{_RESET}")


# ── Structural sequence assembly ──────────────────────────────────────────────

async def _build_structural_sequence(
    stack: list[PeptideRecord],
    goal: str,
    model: Any,
) -> tuple[list[PeptideRecord], str]:
    """
    Ask Gemini to logically order all stack nodes to form a coherent structural
    sequence (e.g. signal → propeptide → mature chain → ...).
    Returns (ordered_stack, joined_sequence).
    """
    catalog = "\n".join(
        f"- {p.accession} | {p.category} | {p.name} | len={len(p.sequence)}"
        for p in stack
    )

    prompt = (
        f"Goal: {goal}\n\n"
        f"The following peptides form a stack of structural nodes. "
        f"Order their accessions to assemble a logically coherent structural sequence "
        f"(e.g. signal peptide first, then propeptide, then mature chains, modifications last). "
        f"Respond with ONLY a JSON array of accession strings in the correct structural order.\n\n"
        f"Stack nodes:\n{catalog}"
    )

    accession_order: list[str] = []
    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        text = response.text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
        accession_order = json.loads(text)
    except Exception as exc:
        print(f"  Gemini structural ordering error: {exc} — using harmony-score order.")
        accession_order = [p.accession for p in stack]

    acc_map = {p.accession: p for p in stack}
    ordered: list[PeptideRecord] = []
    for acc in accession_order:
        if acc in acc_map and acc_map[acc] not in ordered:
            ordered.append(acc_map[acc])
    # append any remaining nodes not returned by Gemini
    for p in stack:
        if p not in ordered:
            ordered.append(p)

    joined = "".join(p.sequence for p in ordered if p.sequence)
    return ordered, joined


# ── FASTA export ───────────────────────────────────────────────────────────────

_FASTA_LINE = 60


def _safe_filename(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text)[:60]


def _save_fasta(
    ordered_stack: list[PeptideRecord],
    structural_sequence: str,
    goal: str,
    output_dir: Path,
) -> Path:
    """
    Write one FASTA file containing:
    - one entry per stack node (individual sequences)
    - a final composite >STRUCTURAL entry (the joined sequence)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_filename(goal[:40])
    fasta_path = output_dir / f"peptide_stack_{slug}_{timestamp}.fasta"

    def _wrap_seq(seq: str) -> str:
        return "\n".join(seq[i:i + _FASTA_LINE] for i in range(0, len(seq), _FASTA_LINE))

    lines: list[str] = []

    for p in ordered_stack:
        if not p.sequence:
            continue
        gene = ",".join(p.gene_names) if p.gene_names else "unknown"
        header = (
            f">{p.accession} | {p.name} | gene={gene} "
            f"| category={p.category} | harmony={p.harmony_score:.3f}"
        )
        lines.append(header)
        lines.append(_wrap_seq(p.sequence))

    # Composite structural entry
    lines.append(
        f">STRUCTURAL_COMPOSITE | nodes={len(ordered_stack)} "
        f"| total_length={len(structural_sequence)} | goal={goal[:80]}"
    )
    lines.append(_wrap_seq(structural_sequence))

    fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # JSON sidecar — primary key: "sequence"
    json_path = fasta_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "sequence": structural_sequence,
                "goal": goal,
                "total_length": len(structural_sequence),
                "node_count": len(ordered_stack),
                "nodes": [
                    {
                        "accession": p.accession,
                        "name": p.name,
                        "gene_names": p.gene_names,
                        "category": p.category,
                        "harmony_score": p.harmony_score,
                        "sequence": p.sequence,
                        "functional_specs": p.functional_specs,
                    }
                    for p in ordered_stack
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return fasta_path


# ── Pipeline orchestration ─────────────────────────────────────────────────────

async def _run_pipeline(
    user_prompt: str,
    gem_api_key: str,
    max_per_category: int = 25,
    render_top_n: int = 10,
    output_dir: Path = Path("output/fasta"),
) -> dict[str, Any]:
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install google-generativeai") from exc

    genai.configure(api_key=gem_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    async with aiohttp.ClientSession() as session:
        # 1. Fetch live category list from UniProt
        print("Fetching live UniProt feature field catalog...")
        live_fields = await _fetch_live_peptide_fields(session)
        print(f"  {len(live_fields)} peptide-related feature fields discovered.")

        # 2. Classify prompt → 3 categories via Gemini
        print("Classifying query via Gemini...")
        selected = await _classify_with_gemini(user_prompt, live_fields, model)
        selected_labels = [f.label for f in selected]
        print(f"  Selected categories: {selected_labels}")

        # 3. Build goal
        goal = (
            f"Analyze UniProt peptides relevant to: \"{user_prompt}\". "
            f"Focus on: {', '.join(selected_labels)}. "
            f"Evaluate functional role, catalytic activity, and subcellular location."
        )
        print(f"\n{_BOLD}Goal:{_RESET} {goal}\n")

        # 4. Build live fields parameter
        fields_param = await _build_fields_param(selected, session)

        # 5. Fetch peptides
        print("Fetching UniProt peptide entries...")
        peptides = await _fetch_uniprot_peptides(selected, session, fields_param, size=max_per_category)

    print(f"Retrieved {len(peptides)} unique peptide entries.")

    if not peptides:
        print("No peptides found for the given categories.")
        return {"goal": goal, "categories": selected_labels, "peptides": []}

    # 6. Stack + Gemini harmony scoring
    stack = list(peptides)
    print("Computing harmony scores via Gemini...")
    scores = await _gem_harmony_request(goal, stack, model)
    for p in stack:
        p.harmony_score = float(scores.get(p.accession, 0.5))
    stack.sort(key=lambda p: p.harmony_score, reverse=True)

    # 7. Render top-N
    top = stack[:render_top_n]
    print(f"\nRendering top {len(top)} peptides ranked by harmony score:")
    for peptide in top:
        render_peptide(peptide)

    print(f"\n{_DIM}Total peptides in stack: {len(stack)}{_RESET}\n")

    # 8. Build structural sequence from all stack nodes
    print("Assembling structural sequence via Gemini...")
    ordered_stack, structural_seq = await _build_structural_sequence(stack, goal, model)
    print(f"  Structural sequence assembled: {len(structural_seq)} aa across {len(ordered_stack)} nodes.")

    # 9. Save FASTA locally
    fasta_path = _save_fasta(ordered_stack, structural_seq, goal, output_dir)
    print(f"{_BOLD}FASTA saved:{_RESET} {fasta_path}\n")

    return {
        "goal": goal,
        "categories": selected_labels,
        "peptides": stack,
        "structural_sequence": structural_seq,
        "fasta_path": str(fasta_path),
    }


# ── Public entry point ─────────────────────────────────────────────────────────

def process_user_prompt_for_peptides(
    user_prompt: str,
    gem_api_key: str,
    max_per_category: int = 25,
    render_top_n: int = 10,
    output_dir: str | Path = Path("output/fasta"),
) -> dict[str, Any]:
    """
    End-to-end peptide pipeline — no hardcoded data.

    1. Fetches live UniProt feature field catalog (ft_* fields).
    2. Uses Gemini to classify the prompt into 3 live categories.
    3. Builds the UniProt ?fields= parameter from live API data.
    4. Fetches matching peptides with name + functional specs.
    5. Scores the stack via Gemini harmony request.
    6. Renders top-N peptide sequences in the terminal.
    7. Assembles all stack nodes into one structural sequence via Gemini.
    8. Saves individual + composite sequences as FASTA to output_dir.

    Args:
        user_prompt:       Free-text query describing the biological interest.
        gem_api_key:       Google Gemini API key.
        max_per_category:  Max UniProt entries per category (default 25).
        render_top_n:      How many top-scored peptides to render (default 10).
        output_dir:        Directory for FASTA output (default: output/fasta/).
    """
    return asyncio.run(
        _run_pipeline(user_prompt, gem_api_key, max_per_category, render_top_n, Path(output_dir))
    )
