"""
query_pipe — Biological Prompt Pipeline

Prompt: include a step to analyze ``filter_physical_compound`` (``str`` or token ``list``)
and classify it to ``ds.PHYSICAL_CATEGORY_ALIASES`` canonical entries (shared vocabulary in
root ``ds.py`` to avoid circular imports).

Prompt: remove ``db_targets`` from the Stage 1 tool schema; derive database / workflow
slugs from ``EXECUTION_CFG`` at ``run_query_pipe`` start (project-wide categories), not
from the model.

Prompt: improve Stage 1 ``transformed_text`` instructions so the model infers the user’s
underlying goal, expands implicit scope, and emits a structured, modular technical brief
usable by later pipeline stages.

INPUT TEXT → Gemini calls for transformation, word splitting, classification
of prompt to database (e.g. PubChem, UniProt).
Classify prompt to organ (fetches UniProt API with all available organs
metadata — keys, TS-IDs, synonyms — via the canonical tisslist.txt
endpoint), functions (fetches UniProt for functional annotations of
proteins, genes, chemicals) based on the filtered organs (saved as
list[str]), and outsrc criteria (disease, unwanted outcome, ...).

Result is a cfg tool for Gem, a pipe that builds up on the next step
and fetches new data resulting in a master prompt for Gem.

Returns dict(organs, function_annotation, outsrc_criteria, filter_physical_compound, …).
"""
from __future__ import annotations

import asyncio
import functools
import json
import pprint
import warnings
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai
import httpx

from execution_cfg import EXECUTION_CFG

# ━━ ENDPOINTS (only hardcoded strings) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_TISSLIST_URL = "https://www.uniprot.org/docs/tisslist.txt"
_UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
_PUBCHEM_COMPOUND = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/JSON"

_HTTP_TIMEOUT = 30
_UNIPROT_FIELDS = "accession,protein_name,gene_names,cc_function,go,cc_disease,cc_tissue_specificity"
_MAX_PROTEINS_PER_ORGAN = 50


# ━━ TISSUE VOCABULARY FETCH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@functools.lru_cache(maxsize=1)
def _fetch_tissue_vocabulary() -> list[str]:
    """Fetch + parse UniProt controlled tissue vocabulary (tisslist.txt).
    Returns canonical tissue/organ names (ID lines)."""
    resp = httpx.get(_TISSLIST_URL, timeout=_HTTP_TIMEOUT, follow_redirects=True)
    resp.raise_for_status()
    tissues: list[str] = []
    for line in resp.text.splitlines():
        if line.startswith("ID   "):
            # strip trailing period and leading tag
            name = line[5:].rstrip(".")
            if name:
                tissues.append(name)
    return tissues


# ━━ ORGAN-BASED FUNCTIONAL ANNOTATION FETCH ━━━━━━━━━━━━━━━━━━━━━━━

async def _fetch_organ_annotations(organs: list[str]) -> list[str]:
    """Query UniProt REST per organ → aggregate functional annotation strings."""
    annotations: list[str] = []

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        for organ in organs:
            query = f'(cc_tissue_specificity:"{organ}") AND (reviewed:true) AND (organism_id:9606)'
            params = {
                "query": query,
                "fields": _UNIPROT_FIELDS,
                "format": "json",
                "size": str(_MAX_PROTEINS_PER_ORGAN),
            }
            try:
                r = await client.get(_UNIPROT_SEARCH, params=params)
                r.raise_for_status()
                data = r.json()
            except (httpx.HTTPError, json.JSONDecodeError):
                continue

            for entry in data.get("results", []):
                acc = entry.get("primaryAccession", "?")
                pname = ""
                rec = entry.get("proteinDescription", {}).get("recommendedName")
                if rec:
                    pname = rec.get("fullName", {}).get("value", "")

                genes = ", ".join(
                    g.get("geneName", {}).get("value", "")
                    for g in entry.get("genes", [])
                    if g.get("geneName")
                )

                funcs = []
                for comment in entry.get("comments", []):
                    if comment.get("commentType") == "FUNCTION":
                        for txt in comment.get("texts", []):
                            funcs.append(txt.get("value", ""))

                go_terms = []
                for xref in entry.get("uniProtKBCrossReferences", []):
                    if xref.get("database") == "GO":
                        props = {p["key"]: p["value"] for p in xref.get("properties", [])}
                        go_terms.append(props.get("GoTerm", xref.get("id", "")))

                diseases = []
                for comment in entry.get("comments", []):
                    if comment.get("commentType") == "DISEASE":
                        dis = comment.get("disease", {})
                        if dis.get("diseaseId"):
                            diseases.append(dis["diseaseId"])

                line = (
                    f"[{acc}] {pname} | genes={genes} | "
                    f"function={'; '.join(funcs)} | "
                    f"GO={'; '.join(go_terms[:10])} | "
                    f"disease={'; '.join(diseases)}"
                )
                annotations.append(line)

    return annotations


# ━━ GEMINI CFG FACTORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_tool_cfg(tools: list[dict], system: str) -> dict:
    """Build a GenerativeModel kwargs dict with forced function calling (mode=ANY)."""
    declarations = []
    for t in tools:
        declarations.append(genai.protos.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    k: genai.protos.Schema(
                        type=genai.protos.Type.STRING if v == "string"
                        else genai.protos.Type.ARRAY,
                        **({"items": genai.protos.Schema(type=genai.protos.Type.STRING)} if v == "array" else {}),
                        description=t.get("prop_descriptions", {}).get(k, k),
                    )
                    for k, v in t["properties"].items()
                },
                required=list(t["properties"].keys()),
            ),
        ))

    return {
        "tools": [genai.protos.Tool(function_declarations=declarations)],
        "tool_config": genai.protos.ToolConfig(
            function_calling_config=genai.protos.FunctionCallingConfig(
                mode=genai.protos.FunctionCallingConfig.Mode.ANY,
            ),
        ),
        "system_instruction": system,
    }


# ━━ STAGE RUNNER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_stage(model_name: str, cfg: dict, contents: str) -> dict[str, Any]:
    """Execute one Gem forced-tool call. Returns the function-call args dict."""
    model = genai.GenerativeModel(model_name, **cfg)
    resp = model.generate_content(contents)

    # EXTRACT FUNCTION CALL ARGS FROM RESPONSE
    for part in resp.candidates[0].content.parts:
        fc = part.function_call
        if fc and fc.args:
            return dict(fc.args)

    # FALLBACK: parse text as JSON if model returned text instead
    for part in resp.candidates[0].content.parts:
        if part.text:
            try:
                return json.loads(part.text)
            except json.JSONDecodeError:
                pass

    return {}


def _run_text_stage(model_name: str, system: str, contents: str) -> str:
    """Execute a free-text Gem call (no tools). Returns model text."""
    model = genai.GenerativeModel(model_name, system_instruction=system)
    resp = model.generate_content(contents)
    return resp.text or ""


# ━━ ROUTING CONTEXT BUILDER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_routing_context() -> str:
    """Compile database routing descriptions from execution_cfg."""
    parts = []
    for slug, spec in EXECUTION_CFG.items():
        r = spec["routing"]
        parts.append(f"- {slug}: {r['router_description']} (keywords: {', '.join(sorted(r['keywords']))})")
    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_query_pipe(
    prompt: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    #filter_physical_compound: str | list[str] | None = None,
) -> dict[str, Any]:
    """
    5-stage sequential Gemini-tool pipeline (+ physical-filter classification).

    Stage 1 — Deconstruct: word-split and transform (DB slugs come from ``EXECUTION_CFG``)
    Stage 1b — Physical filter: resolve ``filter_physical_compound`` via ``ds`` aliases
    Stage 2 — Organ Resolve: match against live UniProt tissue vocabulary
    Stage 3 — Function Annotate: fetch UniProt functional data per organ
    Stage 4 — Outsrc Criteria: extract exclusion / disease / unwanted outcomes
    Stage 5 — Master Prompt: compose enriched master prompt for downstream Gem use

    Returns dict(organs, function_annotation, outsrc_criteria, filter_physical_compound, …).
    """
    genai.configure(api_key=api_key)

    routing_ctx = _build_routing_context()
    # CHAR: fixed workflow slugs from project config — not model output (see Stage 1 tool).
    db_targets = sorted(EXECUTION_CFG.keys())

    # ── STAGE 1: DECONSTRUCT ──────────────────────────────────────────
    s1_tools = [{
        "name": "split_transform_classify",
        "description": (
            "Split the biological prompt into meaningful tokens and produce "
            "transformed_text: a structured technical brief that states the inferred goal, "
            "scope, modular tasks, and normalised terminology for downstream workflow stages."
        ),
        "properties": {
            "tokens": "array",
            "transformed_text": "string",
        },
        "prop_descriptions": {
            "tokens": (
                "Meaningful biological tokens (compounds and multi-word names as single tokens, "
                "e.g. 'vitamin B12'). Include implied entities the user did not name explicitly "
                "when they are required to interpret the goal."
            ),
            "transformed_text": (
                "Single string, plain text, using this exact section header lines in order "
                "(each header on its own line, followed by content):\n"
                "INFERRED_GOAL: One or two sentences — the technical objective implied by the user "
                "(what outcome the workflow should optimise for).\n"
                "SCOPE_AND_CONTEXT: Short paragraph — organism/tissue hints, scale (molecular/cellular/"
                "organ), domain (therapy, mechanism, biomarker, structure, …), and unstated assumptions "
                "you reasonably infer.\n"
                "MODULAR_TASKS: Bullet lines starting with '- ' — each line one atomic, executable "
                "sub-problem a pipeline module could own (retrieve, annotate, filter, rank, validate); "
                "no prose paragraphs in this section.\n"
                "RETRIEVAL_HOOKS: Comma-separated dense list of search-ready terms, synonyms, and "
                "standard names for genes, proteins, pathways, drugs, or anatomical keywords.\n"
                "NORMALISED_QUERY: One cohesive paragraph restating the user request with normalised "
                "scientific nomenclature (no section headers inside this paragraph).\n"
                "Be concise but information-dense; downstream stages consume this as the canonical "
                "expanded reading of the user prompt."
            ),
        },
    }]

    s1_cfg = _make_tool_cfg(s1_tools, system=(
        "You are a senior bioinformatics prompt engineer. Given an informal or clinical biological "
        "user prompt, you (1) extract tokens for lexical downstream use and (2) write "
        "transformed_text as a modular technical brief.\n\n"
        "Infer the user's underlying intent even when the wording is vague (e.g. colloquial drug "
        "requests → molecular targets, pathways, safety constraints). Expand implicit scope only when "
        "biologically standard (default human/clinical context when unspecified). Keep each "
        "MODULAR_TASKS bullet a single clear action boundary so independent workflow steps could "
        "implement it without rereading the original prompt.\n\n"
        "Do not emit per-token database labels; fixed execution categories for this project are: "
        f"{', '.join(db_targets)}.\n"
        f"Category reference (keywords only):\n{routing_ctx}\n\n"
        "Return ALL fields via the split_transform_classify function. transformed_text MUST follow "
        "the section headers and order given in the tool schema for transformed_text."
    ))

    s1_result = _run_stage(model, s1_cfg, prompt)
    tokens = list(s1_result.get("tokens", []))
    transformed = s1_result.get("transformed_text", prompt)

    print(f"[PIPE S1] tokens={len(tokens)}, db_targets (project)={db_targets}, transformed={transformed[:80]}...")

    # ── STAGE 1b: PHYSICAL FILTER (classify str / tokens → canonical ds slots) ──
    # CHAR: runs before organ resolve so logs show gating vocabulary early; idempotent for UniprotKB.
    #physical_resolved = resolve_physical_filter_canonical_list(filter_physical_compound)
    #print(f"[PIPE S1b] filter_physical_compound (canonical)={physical_resolved or '(none — full workflow)'}")

    # ── STAGE 2: ORGAN RESOLVE ────────────────────────────────────────
    tissue_vocab = _fetch_tissue_vocabulary()
    # PASS FULL VOCABULARY TO GEM SO IT CAN MATCH CANONICAL NAMES
    vocab_excerpt = "\n".join(tissue_vocab[:2000])

    s2_tools = [{
        "name": "classify_organs",
        "description": (
            "From the biological context and the canonical UniProt tissue "
            "vocabulary provided, identify which organs/tissues are relevant. "
            "Return ONLY names that exist in the vocabulary."
        ),
        "properties": {
            "organs": "array",
        },
        "prop_descriptions": {
            "organs": f"list of canonical UniProt tissue/organ names from the vocabulary that are relevant to this query",
        },
    }]

    s2_cfg = _make_tool_cfg(s2_tools, system=(
        "You are a biomedical organ/tissue classifier. "
        "Given a biological query and a canonical tissue vocabulary, "
        "select ONLY the organs/tissues from the vocabulary that are "
        "biologically relevant to the query. Be precise — only return "
        "names that exactly match entries in the vocabulary.\n\n"
        "CANONICAL TISSUE VOCABULARY:\n" + vocab_excerpt
    ))

    s2_input = (
        f"Original prompt: {prompt}\n"
        f"Transformed: {transformed}\n"
        f"Tokens: {json.dumps(tokens)}\n"
        f"Project execution categories (fixed): {json.dumps(db_targets)}"
    )
    s2_result = _run_stage(model, s2_cfg, s2_input)
    organs = list(s2_result.get("organs", []))

    print(f"[PIPE S2] organs={organs}")

    # ── STAGE 3: FUNCTION ANNOTATE ────────────────────────────────────
    raw_annotations = asyncio.run(_fetch_organ_annotations(organs)) if organs else []

    s3_tools = [{
        "name": "summarize_functional_annotations",
        "description": (
            "Summarise the raw UniProt protein/gene functional annotations "
            "into concise, non-redundant functional descriptors. "
            "Merge overlapping GO terms and collapse duplicates."
        ),
        "properties": {
            "function_annotation": "array",
        },
        "prop_descriptions": {
            "function_annotation": "list of concise functional annotation summaries (protein function, pathway, GO term groups, chemical compound, reactome, peptide, gene, amino acid,...). MAXIMAL 10 entries",
        },
    }]

    s3_cfg = _make_tool_cfg(s3_tools, system=(
        "You are a protein-function summarisation engine. "
        "Given raw UniProt annotation lines for proteins expressed in "
        "specific organs, produce a deduplicated list of concise "
        "functional descriptors. Group related GO terms. "
        "Remove redundancy. Keep each entry under 120 chars."
    ))

    s3_input = (
        f"Query context: {transformed}\n"
        f"Organs: {json.dumps(organs)}\n"
        f"Raw annotations ({len(raw_annotations)} entries):\n" +
        "\n".join(raw_annotations[:300])
    )
    s3_result = _run_stage(model, s3_cfg, s3_input)
    function_annotation = list(s3_result.get("function_annotation", []))

    print(f"[PIPE S3] function_annotation={len(function_annotation)} entries")

    # ── STAGE 4: OUTSRC CRITERIA ──────────────────────────────────────
    s4_tools = [{
        "name": "extract_outsrc_criteria",
        "description": (
            "Extract exclusion / outsourcing criteria from the biological "
            "context (include the Original prompt to your decision): diseases to avoid, unwanted outcomes, toxicity flags, "
            "contraindications, off-target effects, allergens."
        ),
        "properties": {
            "outsrc_criteria": "array",
        },
        "prop_descriptions": {
            "outsrc_criteria": "list of exclusion criteria, diseases, unwanted outcomes, contraindications derived from the biological context",
        },
    }]

    s4_cfg = _make_tool_cfg(s4_tools, system=(
        "You are a biomedical risk/exclusion analyst. "
        "Given the biological query, identified organs, and functional "
        "annotations, extract ALL relevant exclusion criteria: "
        "diseases to exclude, unwanted physiological outcomes, "
        "toxicity concerns, allergen risks, contraindications, "
        "off-target effects. Be thorough but concise."
    ))

    s4_input = (
        f"Original prompt: {prompt}\n"
        f"Transformed: {transformed}\n"
        f"Organs: {json.dumps(organs)}\n"
        f"Functional annotations: {json.dumps(function_annotation)}"
    )
    s4_result = _run_stage(model, s4_cfg, s4_input)
    outsrc_criteria = list(s4_result.get("outsrc_criteria", []))

    print(f"[PIPE S4] outsrc_criteria={outsrc_criteria}")

    # ── STAGE 5: MASTER PROMPT ────────────────────────────────────────
    master_prompt = _run_text_stage(model, system=(
        "You are a master-prompt composer for a bioinformatics pipeline. "
        "Synthesise all pipeline outputs into a single, dense, actionable "
        "prompt that a downstream Gemini model can use to execute the "
        "biological workflow. Include organ context, functional scope, "
        "and exclusion guardrails. No preamble — output ONLY the prompt."
    ), contents=(
        f"Original user query: {prompt}\n\n"
        f"Stage 1 — Tokens: {json.dumps(tokens)}\n"
        f"Stage 1 — Project DB / workflow slugs: {json.dumps(db_targets)}\n"
        f"Stage 1 — Transformed: {transformed}\n\n"
        f"Stage 2 — Organs: {json.dumps(organs)}\n\n"
        f"Stage 3 — Functional annotations:\n{json.dumps(function_annotation, indent=1)}\n\n"
        f"Stage 4 — Exclusion criteria: {json.dumps(outsrc_criteria)}\n\n"
        "Compose the master prompt now."
    ))

    print(f"[PIPE S5] master_prompt length={len(master_prompt)} chars")

    out = {
        "organs": organs,
        "function_annotation": function_annotation,
        "outsrc_criteria": outsrc_criteria,
        #"filter_physical_compound": physical_resolved,
        "_master_prompt": master_prompt,
        "_tokens": tokens,
        "_db_targets": db_targets,
    }
    print("QP out:")
    pprint.pp(out)
    return out

TEST_QUERY_SPECS = {
        "organs": ["Brain"],
        "function_annotation": [
            '**Functional Scope (Desired Outcomes in CNS):**\n'
           'Prioritize drugs that promote:\n'
           '*   **Neurogenesis, Neurodevelopment, and Neuronal '
           'Maturation:** Including brain and eye development, '
           'survival/renewal/proliferation of neural progenitor cells, '
           'neuronal differentiation (e.g., amacrine, ganglion cells, '
           'cerebellar maturation), axon growth, elongation, guidance, '
           'and major fiber tract formation.\n'
           '*   **Neuroprotection and Neuron Survival:** Through '
           'anti-apoptotic mechanisms (e.g., repressing pro-apoptotic '
           'transcripts, promoting BCL-2), enhancing cell survival '
           'pathways, and protecting glial cells (e.g., anti-apoptotic '
           'role in oligodendrocytes).\n'
           '*   **Synaptic Function and Plasticity:** Support for '
           'synapse formation, modulation of synaptic plasticity, and '
           'enhancement of short-term synaptic function and long-term '
           'potentiation to improve learning and memory.\n'
           '*   **Neuronal Structural Integrity:** Stabilization of '
           'the microtubule cytoskeleton, promotion of axonal '
           'transport of neurofilament proteins, and maintenance of '
           'the myelin sheath.\n'
           '*   **Modulation of CNS Functions:** Therapeutic sedation, '
           'promotion of NREM sleep, beneficial neuromodulation via '
           'receptor activity, and support for neuronal-glial '
           'communication.\n'
           '*   **Sensory System Support:** Maintenance of sensory '
        ],
        "outsrc_criteria": ["heat"],
        #"filter_physical_compound": physical_resolved,
        "_master_prompt": 'Identify pharmaceutical agents demonstrating a beneficial or therapeutic effect on the human **Central Nervous System (CNS)**.\n\n**Functional Scope (Desired Outcomes in CNS):**\nPrioritize drugs that promote:\n*   **Neurogenesis, Neurodevelopment, and Neuronal Maturation:** Including brain and eye development, survival/renewal/proliferation of neural progenitor cells, neuronal differentiation (e.g., amacrine, ganglion cells, cerebellar maturation), axon growth, elongation, guidance, and major fiber tract formation.\n*   **Neuroprotection and Neuron Survival:** Through anti-apoptotic mechanisms (e.g., repressing pro-apoptotic transcripts, promoting BCL-2), enhancing cell survival pathways, and protecting glial cells (e.g., anti-apoptotic role in oligodendrocytes).\n*   **Synaptic Function and Plasticity:** Support for synapse formation, modulation of synaptic plasticity, and enhancement of short-term synaptic function and long-term potentiation to improve learning and memory.\n*   **Neuronal Structural Integrity:** Stabilization of the microtubule cytoskeleton, promotion of axonal transport of neurofilament proteins, and maintenance of the myelin sheath.\n*   **Modulation of CNS Functions:** Therapeutic sedation, promotion of NREM sleep, beneficial neuromodulation via receptor activity, and support for neuronal-glial communication.\n*   **Sensory System Support:** Maintenance of sensory perception (e.g., hearing, vision) through development and integrity of related CNS structures.\n\n**Exclusion Guardrails (Avoid Any Drug Associated With):**\n*   Enhancement of CREB oncogenic signaling\n*   Uncontrolled cell proliferation\n*   Tumor progression (growth, invasion, angiogenesis)\n*   Induction of pro-inflammatory cytokines (IL6, CXCL8/IL8, CSF2/GM-CSF, CCL3, IL6R)\n*   Inflammatory response\n*   Chronic neuroinflammation\n*   Negative regulation of hemopoiesis\n*   Unwanted degradation of CTNNB1 (beta-catenin)\n*   Abnormal growth in non-neuronal cells\n*   Interference with cytokinesis\n*   Interference with platelet secretion',
        "_db_targets": list(EXECUTION_CFG.keys()),
    }
