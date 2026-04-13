"""
query_pipe — Biological Prompt Pipeline

INPUT TEXT → Gemini calls for transformation, word splitting, classification
of prompt to database (e.g. PubChem, UniProt).
Classify prompt to organ (fetches UniProt API with all available organs
metadata — keys, TS-IDs, synonyms — via the canonical tisslist.txt
endpoint), functions (fetches UniProt for functional annotations of
proteins, genes, chemicals) based on the filtered organs (saved as
list[str]), and outsrc criteria (disease, unwanted outcome, ...).

Result is a cfg tool for Gem, a pipe that builds up on the next step
and fetches new data resulting in a master prompt for Gem.

Returns dict(organs=list, function_annotation=list, outsrc_criteria=list).
"""
from __future__ import annotations

import asyncio
import functools
import json
import os
from typing import Any

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
) -> dict[str, list[str]]:
    """
    5-stage sequential Gemini-tool pipeline.

    Stage 1 — Deconstruct: word-split, transform, classify prompt to target DBs
    Stage 2 — Organ Resolve: match against live UniProt tissue vocabulary
    Stage 3 — Function Annotate: fetch UniProt functional data per organ
    Stage 4 — Outsrc Criteria: extract exclusion / disease / unwanted outcomes
    Stage 5 — Master Prompt: compose enriched master prompt for downstream Gem use

    Returns dict(organs, function_annotation, outsrc_criteria).
    """
    genai.configure(api_key=api_key)

    routing_ctx = _build_routing_context()

    # ── STAGE 1: DECONSTRUCT ──────────────────────────────────────────
    s1_tools = [{
        "name": "split_transform_classify",
        "description": (
            "Split the biological prompt into meaningful tokens, "
            "transform compound terms into searchable form, and "
            "classify which databases (e.g. uniprot, pubchem, rcsb_pdb) "
            "each token maps to."
        ),
        "properties": {
            "tokens": "array",
            "db_targets": "array",
            "transformed_text": "string",
        },
        "prop_descriptions": {
            "tokens": "list of meaningful biological tokens extracted from the prompt",
            "db_targets": "list of database slugs (uniprot, pubchem, rcsb_pdb, atom) each token maps to",
            "transformed_text": "the prompt rewritten with normalised scientific terminology",
        },
    }]

    s1_cfg = _make_tool_cfg(s1_tools, system=(
        "You are a bioinformatics NLP preprocessor. "
        "Given a biological prompt, split it into meaningful tokens, "
        "normalise compound terms (e.g. 'vitamin B12' stays as one token), "
        "and classify each token to a target database.\n\n"
        f"Available databases:\n{routing_ctx}\n\n"
        "Return ALL results via the split_transform_classify function."
    ))

    s1_result = _run_stage(model, s1_cfg, prompt)
    tokens = list(s1_result.get("tokens", []))
    db_targets = list(s1_result.get("db_targets", []))
    transformed = s1_result.get("transformed_text", prompt)

    print(f"[PIPE S1] tokens={len(tokens)}, db_targets={db_targets}, transformed={transformed[:80]}...")

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
            "organs": "list of canonical UniProt tissue/organ names from the vocabulary that are relevant to this query",
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
        f"DB targets: {json.dumps(db_targets)}"
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
            "function_annotation": "list of concise functional annotation summaries (protein function, pathway, GO term groups, chemical compound, reactome, peptide, gene, amino acid,...)",
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
        f"Stage 1 — DB targets: {json.dumps(db_targets)}\n"
        f"Stage 1 — Transformed: {transformed}\n\n"
        f"Stage 2 — Organs: {json.dumps(organs)}\n\n"
        f"Stage 3 — Functional annotations:\n{json.dumps(function_annotation, indent=1)}\n\n"
        f"Stage 4 — Exclusion criteria: {json.dumps(outsrc_criteria)}\n\n"
        "Compose the master prompt now."
    ))

    print(f"[PIPE S5] master_prompt length={len(master_prompt)} chars")

    return {
        "organs": organs,
        "function_annotation": function_annotation,
        "outsrc_criteria": outsrc_criteria,
        "_master_prompt": master_prompt,
        "_tokens": tokens,
        "_db_targets": db_targets,
    }


# ━━ CLI ENTRYPOINT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError("Set GEMINI_API_KEY in .env")

    user_prompt = input("Enter biological query: ").strip()
    result = run_query_pipe(user_prompt, api_key=key)

    print("\n" + "=" * 60)
    print("RESULT:")
    print(json.dumps(result, indent=2, default=str))


r"""


C:\Users\Bernhard\PycharmProjects\acid_master\.venv\Scripts\python.exe C:\Users\Bernhard\PycharmProjects\acid_master\query_pipe.py 
C:\Users\Bernhard\PycharmProjects\acid_master\query_pipe.py:25: FutureWarning: 

All support for the `google.generativeai` package has ended. It will no longer be receiving 
updates or bug fixes. Please switch to the `google.genai` package as soon as possible.
See README for more details:

https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md

  import google.generativeai as genai
Enter biological query: I need a drug that makes me smarter 
[PIPE S1] tokens=1, db_targets=['chemical'], transformed=I need a drug that makes me smarter...
[PIPE S2] organs=['Brain']
[PIPE S3] function_annotation=50 entries
[PIPE S4] outsrc_criteria=['Van Maldergem syndrome 2', 'Hennekam lymphangiectasia-lymphedema syndrome 2', 'Inhibition of neuroprogenitor cell proliferation and differentiation', 'Tremor, hereditary essential 1', 'Schizophrenia', 'Substance use disorders', 'Uncontrolled cell proliferation (through MAP kinase signaling)', 'Neurodevelopmental disorder, mitochondrial, with abnormal movements and lactic acidosis, with or without seizures', 'Parkinsonism-dystonia 3, childhood-onset', 'Snijders Blok-Fisher syndrome', 'Intellectual developmental disorder, autosomal dominant 62', 'Spinocerebellar ataxia 12', 'Proapoptotic activity leading to neuronal death', 'Neuronopathy, distal hereditary motor, autosomal dominant 14', 'Amyotrophic lateral sclerosis', 'Perry syndrome', 'Intellectual developmental disorder, autosomal dominant 10', 'Neurodevelopmental disorder with microcephaly, ataxia, and seizures', 'Intellectual developmental disorder with autism and speech delay', 'Increased formation of amyloid-beta (APP-beta)', 'Pitt-Hopkins-like syndrome 2', 'Schizophrenia 17', 'Host entry factor for influenza virus', 'Host entry factor for rabies virus', 'Host entry factor for SARS-CoV-2', 'Neurodevelopmental disorder, non-progressive, with spasticity and transient opisthotonus', 'Intellectual developmental disorder with neuropsychiatric features', 'Type 2 diabetes mellitus', 'Dystonia 2, torsion, autosomal recessive', 'Major depressive disorder', 'Attention deficit-hyperactivity disorder 7', 'Microcephaly-micromelia syndrome', 'Microcephaly, short stature, and limb abnormalities', 'Disruption of genome stability', 'Disruption of cell cycle checkpoints', 'Waardenburg syndrome 2E', 'Waardenburg syndrome 4C', 'Peripheral demyelinating neuropathy, central dysmyelinating leukodystrophy, Waardenburg syndrome and Hirschsprung disease', 'Highly potent vasoconstriction', 'Deafness, X-linked, 2', 'Opioid addiction/dependence', 'Opioid-related side effects (e.g., respiratory depression, constipation, sedation)', 'Involvement in feeding disorders', 'Neurodevelopmental disorder with speech impairment and with or without seizures', 'Off-target effects on cardiac nodal cells/pacemaking functions', 'Brown-Vialetto-Van Laere syndrome 2', 'Acting as a receptor for retroviruses (e.g., PERV-A)', 'Association with psychoactive substances (e.g., ergot alkaloids, DOI, LSD)', 'Psychotropic side effects (e.g., hallucinations)', 'Alterations in appetite and eating behavior', 'Altered responses to anxiogenic stimuli and stress', 'Impact on insulin sensitivity and glucose homeostasis']
[PIPE S5] master_prompt length=4789 chars

============================================================
RESULT:
{
  "organs": [
    "Brain"
  ],
  "function_annotation": [
    "[Q6V0I7] Protocadherin Fat 4 | genes=FAT4 | function=Cadherins are calcium-dependent cell adhesion proteins. FAT4 plays a role in the maintenance of planar cell polarity as well as in inhibition of YAP1-mediated neuroprogenitor cell proliferation and differentiation (By similarity) | GO=C:adherens junction; C:extracellular exosome; C:plasma membrane; F:calcium ion binding; P:axonogenesis; P:cell-cell adhesion mediated by cadherin; P:cerebral cortex development; P:epithelial cell differentiation; P:heterophilic cell-cell adhesion via plasma membrane cell adhesion molecules; P:hippo signaling | disease=Van Maldergem syndrome 2; Hennekam lymphangiectasia-lymphedema syndrome 2",
    "[O94818] Nucleolar protein 4 | genes=NOL4 | function= | GO=C:nucleolus; F:RNA binding | disease=",
    "[Q9HBT6] Cadherin-20 | genes=CDH20 | function=Cadherins are calcium-dependent cell adhesion proteins. They preferentially interact with themselves in a homophilic manner in connecting cells; cadherins may thus contribute to the sorting of heterogeneous cell types | GO=C:adherens junction; C:catenin complex; F:beta-catenin binding; F:cadherin binding; F:calcium ion binding; P:adherens junction organization; P:calcium-dependent cell-cell adhesion via plasma membrane cell adhesion molecules; P:cell migration; P:cell morphogenesis; P:cell-cell adhesion mediated by cadherin | disease=",
    "[O95670] V-type proton ATPase subunit G 2 | genes=ATP6V1G2 | function=Subunit of the V1 complex of vacuolar(H+)-ATPase (V-ATPase), a multisubunit enzyme composed of a peripheral complex (V1) that hydrolyzes ATP and a membrane integral complex (V0) that translocates protons. V-ATPase is responsible for acidifying and maintaining the pH of intracellular compartments and in some cell types, is targeted to the plasma membrane, where it is responsible for acidifying the extracellular environment | GO=C:clathrin-coated vesicle membrane; C:cytosol; C:extrinsic component of synaptic vesicle membrane; C:melanosome; C:synaptic vesicle membrane; C:vacuolar proton-transporting V-type ATPase, V1 domain; F:ATP hydrolysis activity; F:proton-transporting ATPase activity, rotational mechanism; P:regulation of macroautophagy; P:synaptic vesicle lumen acidification | disease=",
    "[P35462] D(3) dopamine receptor | genes=DRD3 | function=Dopamine receptor that is primarily expressed in limbic areas of the brain and is involved in the modulation of cognitive, emotional, and endocrine functions (PubMed:39984436). Plays a key role in regulating neuronal signaling pathways associated with motivation, reward, and behavior (PubMed:39984436). Coupled to G(i)/G(o) proteins; activation leads to inhibition of adenylate cyclase and decreased intracellular cAMP levels (PubMed:10578130). Involved in the control of locomotor activity and implicated in several neuropsychiatric disorders, including schizophrenia and substance use disorders (PubMed:39984436). Promotes cell proliferation through MAP kinase signaling (PubMed:19520868). Also involved in autophagy regulation: receptor activation stimulates AMPK, which phosphorylates RPTOR and enhances its interaction with MTOR, thereby inhibiting MTORC1 signaling and its downstream target RPS6KB1. This leads to activation of ULK1 and initiation of the autophagy cascade (PubMed:31538542). Forms heterotetramers with DRD1 to potentiate beta-arrestin recruitment and mediate locomotor activity (By similarity) | GO=C:plasma membrane; C:synapse; F:dopamine neurotransmitter receptor activity, coupled via Gi/Go; F:G protein-coupled receptor activity; P:acid secretion; P:adenylate cyclase-activating dopamine receptor signaling pathway; P:adenylate cyclase-inhibiting dopamine receptor signaling pathway; P:arachidonate secretion; P:behavioral response to cocaine; P:circadian regulation of gene expression | disease=Tremor, hereditary essential 1; Schizophrenia",
    "[Q8IUM7] Neuronal PAS domain-containing protein 4 | genes=NPAS4 | function=Transcription factor expressed in neurons of the brain that regulates the excitatory-inhibitory balance within neural circuits and is required for contextual memory in the hippocampus (By similarity). Plays a key role in the structural and functional plasticity of neurons (By similarity). Acts as an early-response transcription factor in both excitatory and inhibitory neurons, where it induces distinct but overlapping sets of late-response genes in these two types of neurons, allowing the synapses that form on inhibitory and excitatory neurons to be modified by neuronal activity in a manner specific to their function within a circuit, thereby facilitating appropriate circuit responses to sensory experience (By similarity). In excitatory neurons, activates transcription of BDNF, which in turn controls the number of GABA-releasing synapses that form on excitatory neurons, thereby promoting an increased number of inhibitory synapses on excitatory neurons (By similarity). In inhibitory neurons, regulates a distinct set of target genes that serve to increase excitatory input onto somatostatin neurons, probably resulting in enhanced feedback inhibition within cortical circuits (By similarity). The excitatory and inhibitory balance in neurons affects a number of processes, such as short-term and long-term memory, acquisition of experience, fear memory, response to stress and social behavior (By similarity). Acts as a regulator of dendritic spine development in olfactory bulb granule cells in a sensory-experience-dependent manner by regulating expression of MDM2 (By similarity). Efficient DNA binding requires dimerization with another bHLH protein, such as ARNT, ARNT2 or BMAL1 (PubMed:14701734). Can activate the CME (CNS midline enhancer) element (PubMed:14701734) | GO=C:chromatin; C:cytosol; C:nucleoplasm; C:nucleus; C:postsynapse; C:transcription regulator complex; F:DNA-binding transcription activator activity, RNA polymerase II-specific; F:DNA-binding transcription factor activity, RNA polymerase II-specific; F:protein heterodimerization activity; F:protein-containing complex binding | disease=",
    "[Q9UGM6] Tryptophan--tRNA ligase, mitochondrial | genes=WARS2 | function=Catalyzes the attachment of tryptophan to tRNA(Trp) in a two-step reaction: tryptophan is first activated by ATP to form Trp-AMP and then transferred to the acceptor end of tRNA(Trp) | GO=C:mitochondrial matrix; C:mitochondrion; C:nucleoplasm; C:plasma membrane; F:ATP binding; F:tryptophan-tRNA ligase activity; P:mitochondrial tryptophanyl-tRNA aminoacylation; P:positive regulation of angiogenesis; P:tRNA aminoacylation for protein translation; P:vasculogenesis | disease=Neurodevelopmental disorder, mitochondrial, with abnormal movements and lactic acidosis, with or without seizures; Parkinsonism-dystonia 3, childhood-onset",
    "[P20264] POU domain, class 3, transcription factor 3 | genes=POU3F3 | function=Transcription factor that acts synergistically with SOX11 and SOX4. Plays a role in neuronal development (PubMed:31303265). Is implicated in an enhancer activity at the embryonic met-mesencephalic junction; the enhancer element contains the octamer motif (5'-ATTTGCAT-3') (By similarity) | GO=C:chromatin; C:nucleoplasm; C:nucleus; F:DNA-binding transcription factor activity; F:DNA-binding transcription factor activity, RNA polymerase II-specific; F:HMG box domain binding; F:protein homodimerization activity; F:RNA polymerase II cis-regulatory region sequence-specific DNA binding; F:sequence-specific DNA binding; P:central nervous system development | disease=Snijders Blok-Fisher syndrome",
    "[P55289] Cadherin-12 | genes=CDH12 | function=Cadherins are calcium-dependent cell adhesion proteins. They preferentially interact with themselves in a homophilic manner in connecting cells; cadherins may thus contribute to the sorting of heterogeneous cell types | GO=C:adherens junction; C:catenin complex; C:plasma membrane; F:beta-catenin binding; F:cadherin binding; F:calcium ion binding; P:adherens junction organization; P:calcium-dependent cell-cell adhesion via plasma membrane cell adhesion molecules; P:cell migration; P:cell morphogenesis | disease=",
    "[Q14194] Dihydropyrimidinase-related protein 1 | genes=CRMP1 | function=Necessary for signaling by class 3 semaphorins and subsequent remodeling of the cytoskeleton (PubMed:25358863). Plays a role in axon guidance (PubMed:25358863). During the axon guidance process, acts downstream of SEMA3A to promote FLNA dissociation from F-actin which results in the rearrangement of the actin cytoskeleton and the collapse of the growth cone (PubMed:25358863). Involved in invasive growth and cell migration (PubMed:11562390). May participate in cytokinesis (PubMed:19799413) | GO=C:actin cytoskeleton; C:centrosome; C:cytosol; C:dendrite; C:growth cone; C:midbody; C:perikaryon; C:postsynapse; C:presynapse; C:spindle | disease=",
    "[P78352] Disks large homolog 4 | genes=DLG4 | function=Postsynaptic scaffolding protein that plays a critical role in synaptogenesis and synaptic plasticity by providing a platform for the postsynaptic clustering of crucial synaptic proteins. Interacts with the cytoplasmic tail of NMDA receptor subunits and shaker-type potassium channels. Required for synaptic plasticity associated with NMDA receptor signaling. Overexpression or depletion of DLG4 changes the ratio of excitatory to inhibitory synapses in hippocampal neurons. May reduce the amplitude of ASIC3 acid-evoked currents by retaining the channel intracellularly. May regulate the intracellular trafficking of ADR1B. Also regulates AMPA-type glutamate receptor (AMPAR) immobilization at postsynaptic density keeping the channels in an activated state in the presence of glutamate and preventing synaptic depression (By similarity). Under basal conditions, cooperates with FYN to stabilize palmitoyltransferase ZDHHC5 at the synaptic membrane through FYN-mediated phosphorylation of ZDHHC5 and its subsequent inhibition of association with endocytic proteins (PubMed:26334723) | GO=C:adherens junction; C:AMPA glutamate receptor complex; C:cell junction; C:cerebellar mossy fiber; C:cortical cytoskeleton; C:cytoplasm; C:cytosol; C:dendrite cytoplasm; C:dendritic spine; C:endocytic vesicle membrane | disease=Intellectual developmental disorder, autosomal dominant 62",
    "[Q8NFZ8] Cell adhesion molecule 4 | genes=CADM4 | function=Involved in the cell-cell adhesion. Has calcium- and magnesium-independent cell-cell adhesion activity. May have tumor-suppressor activity | GO=C:cell leading edge; C:cell-cell contact zone; C:membrane; F:protein phosphatase binding; F:receptor tyrosine kinase binding; F:vascular endothelial growth factor receptor 1 binding; F:vascular endothelial growth factor receptor 2 binding; P:homophilic cell adhesion via plasma membrane adhesion molecules; P:negative regulation of peptidyl-threonine phosphorylation; P:negative regulation of peptidyl-tyrosine phosphorylation | disease=",
    "[P78559] Microtubule-associated protein 1A | genes=MAP1A | function=Structural protein involved in the filamentous cross-bridging between microtubules and other skeletal elements | GO=C:axon; C:axon cytoplasm; C:axon initial segment; C:cytoplasm; C:cytosol; C:dendrite; C:dendritic branch; C:dendritic microtubule; C:dendritic shaft; C:microtubule | disease=",
    "[P80723] Brain acid soluble protein 1 | genes=BASP1 | function= | GO=C:cell junction; C:chromatin; C:cytoplasm; C:cytoskeleton; C:extracellular exosome; C:growth cone; C:nuclear matrix; C:nuclear speck; C:nucleus; C:plasma membrane | disease=",
    "[Q00005] Serine/threonine-protein phosphatase 2A 55 kDa regulatory subunit B beta isoform | genes=PPP2R2B | function=The B regulatory subunit might modulate substrate selectivity and catalytic activity, and might also direct the localization of the catalytic enzyme to a particular subcellular compartment. Within the PP2A holoenzyme complex, isoform 2 is required to promote proapoptotic activity (By similarity). Isoform 2 regulates neuronal survival through the mitochondrial fission and fusion balance (By similarity) | GO=C:cytoskeleton; C:cytosol; C:mitochondrial outer membrane; C:mitochondrion; C:protein phosphatase type 2A complex; F:protein phosphatase regulator activity; P:apoptotic process | disease=Spinocerebellar ataxia 12",
    "[Q14203] Dynactin subunit 1 | genes=DCTN1 | function=Part of the dynactin complex that activates the molecular motor dynein for ultra-processive transport along microtubules (By similarity). Plays a key role in dynein-mediated retrograde transport of vesicles and organelles along microtubules by recruiting and tethering dynein to microtubules. Binds to both dynein and microtubules providing a link between specific cargos, microtubules and dynein. Essential for targeting dynein to microtubule plus ends, recruiting dynein to membranous cargos and enhancing dynein processivity (the ability to move along a microtubule for a long distance without falling off the track). Can also act as a brake to slow the dynein motor during motility along the microtubule (PubMed:25185702). Can regulate microtubule stability by promoting microtubule formation, nucleation and polymerization and by inhibiting microtubule catastrophe in neurons. Inhibits microtubule catastrophe by binding both to microtubules and to tubulin, leading to enhanced microtubule stability along the axon (PubMed:23874158). Plays a role in metaphase spindle orientation (PubMed:22327364). Plays a role in centriole cohesion and subdistal appendage organization and function. Its recruitment to the centriole in a KIF3A-dependent manner is essential for the maintenance of centriole cohesion and the formation of subdistal appendage. Also required for microtubule anchoring at the mother centriole (PubMed:23386061). Plays a role in primary cilia formation (PubMed:25774020) | GO=C:acrosomal vesicle; C:axon; C:cell cortex; C:cell cortex region; C:cell leading edge; C:centriolar subdistal appendage; C:centriole; C:centrosome; C:ciliary basal body; C:cilium | disease=Neuronopathy, distal hereditary motor, autosomal dominant 14; Amyotrophic lateral sclerosis; Perry syndrome",
    "[Q9Y698] Voltage-dependent calcium channel gamma-2 subunit | genes=CACNG2 | function=Regulates the trafficking and gating properties of AMPA-selective glutamate receptors (AMPARs). Promotes their targeting to the cell membrane and synapses and modulates their gating properties by slowing their rates of activation, deactivation and desensitization. Does not show subunit-specific AMPA receptor regulation and regulates all AMPAR subunits. Thought to stabilize the calcium channel in an inactivated (closed) state | GO=C:AMPA glutamate receptor complex; C:cell surface; C:cerebellar mossy fiber; C:endocytic vesicle membrane; C:glutamatergic synapse; C:hippocampal mossy fiber to CA3 synapse; C:plasma membrane; C:postsynaptic density membrane; C:Schaffer collateral - CA1 synapse; C:somatodendritic compartment | disease=Intellectual developmental disorder, autosomal dominant 10",
    "[P49591] Serine--tRNA ligase, cytoplasmic | genes=SARS1 | function=Catalyzes the attachment of serine to tRNA(Ser) in a two-step reaction: serine is first activated by ATP to form Ser-AMP and then transferred to the acceptor end of tRNA(Ser) (PubMed:22353712, PubMed:24095058, PubMed:26433229, PubMed:28236339, PubMed:34570399, PubMed:36041817, PubMed:9431993). Is probably also able to aminoacylate tRNA(Sec) with serine, to form the misacylated tRNA L-seryl-tRNA(Sec), which will be further converted into selenocysteinyl-tRNA(Sec) (PubMed:26433229, PubMed:28236339, PubMed:34570399, PubMed:9431993). In the nucleus, binds to the VEGFA core promoter and prevents MYC binding and transcriptional activation by MYC (PubMed:24940000). Recruits SIRT2 to the VEGFA promoter, promoting deacetylation of histone H4 at 'Lys-16' (H4K16). Thereby, inhibits the production of VEGFA and sprouting angiogenesis mediated by VEGFA (PubMed:19423847, PubMed:19423848, PubMed:24940000) | GO=C:cytoplasm; C:cytosol; C:extracellular exosome; C:nucleus; F:ATP binding; F:enzyme binding; F:molecular adaptor activity; F:protein homodimerization activity; F:RNA polymerase II cis-regulatory region sequence-specific DNA binding; F:selenocysteine-tRNA ligase activity | disease=Neurodevelopmental disorder with microcephaly, ataxia, and seizures",
    "[Q16650] T-box brain protein 1 | genes=TBR1 | function=Transcriptional repressor involved in multiple aspects of cortical development, including neuronal migration, laminar and areal identity, and axonal projection (PubMed:25232744, PubMed:30250039). As transcriptional repressor of FEZF2, it blocks the formation of the corticospinal (CS) tract from layer 6 projection neurons, thereby restricting the origin of CS axons specifically to layer 5 neurons (By similarity) | GO=C:chromatin; C:nucleus; F:chromatin DNA binding; F:DNA-binding transcription factor activity; F:DNA-binding transcription factor activity, RNA polymerase II-specific; F:protein kinase binding; F:RNA polymerase II cis-regulatory region sequence-specific DNA binding; P:amygdala development; P:brain development; P:cell fate specification | disease=Intellectual developmental disorder with autism and speech delay",
    "[Q99767] Amyloid-beta A4 precursor protein-binding family A member 2 | genes=APBA2 | function=Putative function in synaptic vesicle exocytosis by binding to STXBP1, an essential component of the synaptic vesicle exocytotic machinery. May modulate processing of the amyloid-beta precursor protein (APP) and hence formation of APP-beta | GO=C:cytoplasm; C:dendritic spine; C:plasma membrane; C:presynapse; C:Schaffer collateral - CA1 synapse; F:amyloid-beta binding; F:identical protein binding; P:chemical synaptic transmission; P:in utero embryonic development; P:locomotory behavior | disease=",
    "[Q9ULB1] Neurexin-1 | genes=NRXN1 | function=Cell surface protein involved in cell-cell-interactions, exocytosis of secretory granules and regulation of signal transmission. Function is isoform-specific. Alpha-type isoforms have a long N-terminus with six laminin G-like domains and play an important role in synaptic signal transmission. Alpha-type isoforms play a role in the regulation of calcium channel activity and Ca(2+)-triggered neurotransmitter release at synapses and at neuromuscular junctions. They play an important role in Ca(2+)-triggered exocytosis of secretory granules in pituitary gland. They may affect their functions at synapses and in endocrine cells via their interactions with proteins from the exocytotic machinery. Likewise, alpha-type isoforms play a role in regulating the activity of postsynaptic NMDA receptors, a subtype of glutamate-gated ion channels. Both alpha-type and beta-type isoforms may play a role in the formation or maintenance of synaptic junctions via their interactions (via the extracellular domains) with neuroligin family members, CBLN1 or CBLN2. In vitro, triggers the de novo formation of presynaptic structures. May be involved in specification of excitatory synapses. Alpha-type isoforms were first identified as receptors for alpha-latrotoxin from spider venom | GO=C:cell surface; C:endoplasmic reticulum; C:neuronal cell body; C:nuclear membrane; C:nucleolus; C:plasma membrane; C:presynaptic membrane; C:vesicle; F:acetylcholine receptor binding; F:calcium channel regulator activity | disease=Pitt-Hopkins-like syndrome 2; Schizophrenia 17",
    "[Q14416] Metabotropic glutamate receptor 2 | genes=GRM2 | function=Dimeric G protein-coupled receptor which is activated by the excitatory neurotransmitter L-glutamate (PubMed:37286794). Plays critical roles in modulating synaptic transmission and neuronal excitability. Upon activation by glutamate, inhibits presynaptic calcium channels, reducing further glutamate release and dampening excitatory signaling (By similarity). Mechanistically, ligand binding causes a conformation change that triggers signaling via guanine nucleotide-binding proteins (G proteins) and modulates the activity of down-stream effectors, such as adenylate cyclase. May mediate suppression of neurotransmission or may be involved in synaptogenesis or synaptic stabilization; (Microbial infection) Plays an important role in influenza virus internalization; (Microbial infection) Acts as a host entry factor for rabies virus that hijacks the endocytosis of GRM2 to enter cells; (Microbial infection) Acts as a host entry factor for SARS-CoV-2 that hijacks the endocytosis of GRM2 to enter cells | GO=C:astrocyte projection; C:axon; C:dendrite; C:glutamatergic synapse; C:plasma membrane; C:postsynaptic membrane; C:presynaptic membrane; F:calcium channel regulator activity; F:G protein-coupled receptor activity; F:glutamate receptor activity | disease=",
    "[Q5XKL5] BTB/POZ domain-containing protein 8 | genes=BTBD8 | function=Involved in clathrin-mediated endocytosis at the synapse. Plays a role in neuronal development and in synaptic vesicle recycling in mature neurons, a process required for normal synaptic transmission | GO=C:AP-2 adaptor complex; C:axon; C:extrinsic component of synaptic vesicle membrane; C:neuron projection terminus; C:nucleoplasm; C:nucleus; P:synaptic vesicle budding from endosome; P:synaptic vesicle endocytosis | disease=",
    "[Q92752] Tenascin-R | genes=TNR | function=Neural extracellular matrix (ECM) protein involved in interactions with different cells and matrix components. These interactions can influence cellular behavior by either evoking a stable adhesion and differentiation, or repulsion and inhibition of neurite growth. Binding to cell surface gangliosides inhibits RGD-dependent integrin-mediated cell adhesion and results in an inhibition of PTK2/FAK1 (FAK) phosphorylation and cell detachment. Binding to membrane surface sulfatides results in a oligodendrocyte adhesion and differentiation. Interaction with CNTN1 induces a repulsion of neurons and an inhibition of neurite outgrowth. Interacts with SCN2B may play a crucial role in clustering and regulation of activity of sodium channels at nodes of Ranvier. TNR-linked chondroitin sulfate glycosaminoglycans are involved in the interaction with FN1 and mediate inhibition of cell adhesion and neurite outgrowth. The highly regulated addition of sulfated carbohydrate structure may modulate the adhesive properties of TNR over the course of development and during synapse maintenance (By similarity) | GO=C:cell surface; C:extracellular matrix; C:extracellular region; C:extracellular space; C:glutamatergic synapse; C:membrane raft; C:perineuronal net; C:Schaffer collateral - CA1 synapse; C:tenascin complex; P:associative learning | disease=Neurodevelopmental disorder, non-progressive, with spasticity and transient opisthotonus",
    "[P0C0E4] Ras-related protein Rab-40A-like | genes=RAB40AL | function=May act as substrate-recognition component of the ECS(RAB40) E3 ubiquitin ligase complex which mediates the ubiquitination and subsequent proteasomal degradation of target proteins (By similarity). The Rab40 subfamily belongs to the Rab family that are key regulators of intracellular membrane trafficking, from the formation of transport vesicles to their fusion with membranes. Rabs cycle between an inactive GDP-bound form and an active GTP-bound form that is able to recruit to membranes different sets of downstream effectors directly responsible for vesicle formation, movement, tethering and fusion (By similarity) | GO=C:cytoplasm; C:endosome; C:mitochondrion; C:plasma membrane; C:synaptic vesicle; F:G protein activity; F:GTP binding; F:GTPase activity; F:metal ion binding; P:exocytosis | disease=",
    "[Q9Y2W3] Proton-associated sugar transporter A | genes=SLC45A1 | function=Proton-associated glucose transporter in the brain | GO=C:membrane; F:D-glucose:proton symporter activity; F:galactose:proton symporter activity; F:sucrose:proton symporter activity; P:D-glucose transmembrane transport; P:galactose transmembrane transport | disease=Intellectual developmental disorder with neuropsychiatric features",
    "[Q9BQJ4] Transmembrane protein 47 | genes=TMEM47 | function=Regulates cell junction organization in epithelial cells. May play a role in the transition from adherens junction to tight junction assembly. May regulate F-actin polymerization required for tight junctional localization dynamics and affect the junctional localization of PARD6B. During podocyte differentiation may negatively regulate activity of FYN and subsequently the abundance of nephrin (By similarity) | GO=C:adherens junction; C:cell-cell junction; C:plasma membrane; P:cell-cell adhesion | disease=",
    "[Q9P0L2] Serine/threonine-protein kinase MARK1 | genes=MARK1 | function=Serine/threonine-protein kinase (PubMed:23666762). Involved in cell polarity and microtubule dynamics regulation. Phosphorylates DCX, MAP2 and MAP4. Phosphorylates the microtubule-associated protein MAPT/TAU (PubMed:23666762). Involved in cell polarity by phosphorylating the microtubule-associated proteins MAP2, MAP4 and MAPT/TAU at KXGS motifs, causing detachment from microtubules, and their disassembly. Involved in the regulation of neuronal migration through its dual activities in regulating cellular polarity and microtubule dynamics, possibly by phosphorylating and regulating DCX. Also acts as a positive regulator of the Wnt signaling pathway, probably by mediating phosphorylation of dishevelled proteins (DVL1, DVL2 and/or DVL3) | GO=C:cytoplasm; C:cytoskeleton; C:dendrite; C:glutamatergic synapse; C:microtubule cytoskeleton; C:plasma membrane; C:postsynapse; F:ATP binding; F:magnesium ion binding; F:phosphatidic acid binding | disease=",
    "[Q9UBZ4] DNA-(apurinic or apyrimidinic site) endonuclease 2 | genes=APEX2 | function=Functions as a weak apurinic/apyrimidinic (AP) endodeoxyribonuclease in the DNA base excision repair (BER) pathway of DNA lesions induced by oxidative and alkylating agents (PubMed:16687656). Initiates repair of AP sites in DNA by catalyzing hydrolytic incision of the phosphodiester backbone immediately adjacent to the damage, generating a single-strand break with 5'-deoxyribose phosphate and 3'-hydroxyl ends. Also displays double-stranded DNA 3'-5' exonuclease, 3'-phosphodiesterase activities (PubMed:16687656, PubMed:19443450, PubMed:32516598). Shows robust 3'-5' exonuclease activity on 3'-recessed heteroduplex DNA and is able to remove mismatched nucleotides preferentially (PubMed:16687656, PubMed:19443450). Also exhibits 3'-5' exonuclease activity on a single nucleotide gap containing heteroduplex DNA and on blunt-ended substrates (PubMed:16687656). Shows fairly strong 3'-phosphodiesterase activity involved in the removal of 3'-damaged termini formed in DNA by oxidative agents (PubMed:16687656, PubMed:19443450). In the nucleus functions in the PCNA-dependent BER pathway (PubMed:11376153). Plays a role in reversing blocked 3' DNA ends, problematic lesions that preclude DNA synthesis (PubMed:32516598). Required for somatic hypermutation (SHM) and DNA cleavage step of class switch recombination (CSR) of immunoglobulin genes (By similarity). Required for proper cell cycle progression during proliferation of peripheral lymphocytes (By similarity) | GO=C:fibrillar center; C:mitochondrion; C:nucleoplasm; C:nucleus; F:DNA binding; F:DNA-(apurinic or apyrimidinic site) endonuclease activity; F:double-stranded DNA 3'-5' DNA exonuclease activity; F:phosphoric diester hydrolase activity; F:zinc ion binding; P:base-excision repair | disease=",
    "[Q9UQB3] Catenin delta-2 | genes=CTNND2 | function=Has a critical role in neuronal development, particularly in the formation and/or maintenance of dendritic spines and synapses (PubMed:25807484). Involved in the regulation of Wnt signaling (PubMed:25807484). It probably acts on beta-catenin turnover, facilitating beta-catenin interaction with GSK3B, phosphorylation, ubiquitination and degradation (By similarity). Functions as a transcriptional activator when bound to ZBTB33 (By similarity). May be involved in neuronal cell adhesion and tissue morphogenesis and integrity by regulating adhesion molecules | GO=C:adherens junction; C:cytoplasm; C:dendrite; C:nucleus; C:perikaryon; C:plasma membrane; C:postsynaptic density; F:beta-catenin binding; F:cadherin binding; P:cell adhesion | disease=",
    "[Q86YR7] Probable guanine nucleotide exchange factor MCF2L2 | genes=MCF2L2 | function=Probably functions as a guanine nucleotide exchange factor | GO=C:cytoplasm; C:cytosol; F:guanyl-nucleotide exchange factor activity | disease=Type 2 diabetes mellitus",
    "[P84074] Neuron-specific calcium-binding protein hippocalcin | genes=HPCA | function=Calcium-binding protein that may play a role in the regulation of voltage-dependent calcium channels (PubMed:28398555). May also play a role in cyclic-nucleotide-mediated signaling through the regulation of adenylate and guanylate cyclases (By similarity) | GO=C:axon; C:cytoplasm; C:cytosol; C:dendrite cytoplasm; C:dendrite membrane; C:dendritic spine head; C:glutamatergic synapse; C:membrane; C:neuronal cell body membrane; C:perikaryon | disease=Dystonia 2, torsion, autosomal recessive",
    "[Q8IWU9] Tryptophan 5-hydroxylase 2 | genes=TPH2 | function= | GO=C:cytosol; C:neuron projection; F:iron ion binding; F:tryptophan 5-monooxygenase activity; P:aromatic amino acid metabolic process; P:serotonin biosynthetic process | disease=Major depressive disorder; Attention deficit-hyperactivity disorder 7",
    "[Q9BWW7] Transcriptional repressor scratch 1 | genes=SCRT1 | function=Transcriptional repressor that binds E-box motif CAGGTG. Can modulate the action of basic helix-loop-helix (bHLH) transcription factors, critical for neuronal differentiation | GO=C:nuclear body; F:DNA-binding transcription factor activity; F:DNA-binding transcription factor activity, RNA polymerase II-specific; F:DNA-binding transcription repressor activity, RNA polymerase II-specific; F:RNA polymerase II cis-regulatory region sequence-specific DNA binding; F:RNA polymerase II transcription regulatory region sequence-specific DNA binding; F:sequence-specific DNA binding; F:sequence-specific double-stranded DNA binding; F:zinc ion binding; P:negative regulation of transcription by RNA polymerase II | disease=",
    "[Q9BXU9] Calcium-binding protein 8 | genes=CALN1 | function=Negatively regulates Golgi-to-plasma membrane trafficking by interacting with PI4KB and inhibiting its activity. May play a role in the physiology of neurons and is potentially important in memory and learning | GO=C:perinuclear region of cytoplasm; C:plasma membrane; C:trans-Golgi network membrane; F:calcium ion binding | disease=",
    "[Q9NYP3] Protein downstream neighbor of Son | genes=DONSON | function=Replisome component that maintains genome stability by protecting stalled or damaged replication forks. After the induction of replication stress, required for the stabilization of stalled replication forks, the efficient activation of the intra-S-phase and G/2M cell-cycle checkpoints and the maintenance of genome stability | GO=C:nucleus; C:replication fork; C:replisome; P:DNA damage checkpoint signaling; P:DNA replication; P:mitotic DNA replication checkpoint signaling; P:mitotic G2 DNA damage checkpoint signaling; P:nuclear DNA replication; P:replication fork processing | disease=Microcephaly-micromelia syndrome; Microcephaly, short stature, and limb abnormalities",
    "[P56693] Transcription factor SOX-10 | genes=SOX10 | function=Transcription factor that plays a central role in developing and mature glia (By similarity). Specifically activates expression of myelin genes, during oligodendrocyte (OL) maturation, such as DUSP15 and MYRF, thereby playing a central role in oligodendrocyte maturation and CNS myelination (By similarity). Once induced, MYRF cooperates with SOX10 to implement the myelination program (By similarity). Transcriptional activator of MITF, acting synergistically with PAX3 (PubMed:21965087). Transcriptional activator of MBP, via binding to the gene promoter (By similarity) | GO=C:chromatin; C:mitochondrial outer membrane; C:nucleoplasm; C:nucleus; F:DNA binding; F:DNA-binding transcription activator activity; F:DNA-binding transcription activator activity, RNA polymerase II-specific; F:DNA-binding transcription factor activity; F:DNA-binding transcription factor activity, RNA polymerase II-specific; F:DNA-binding transcription factor binding | disease=Waardenburg syndrome 2E; Waardenburg syndrome 4C; Peripheral demyelinating neuropathy, central dysmyelinating leukodystrophy, Waardenburg syndrome and Hirschsprung disease",
    "[Q96PZ7] CUB and sushi domain-containing protein 1 | genes=CSMD1 | function=Potential suppressor of squamous cell carcinomas | GO=C:membrane; P:conditioned place preference; P:female gonad development; P:gene expression; P:glucose homeostasis; P:male gonad development; P:mammary gland branching involved in pregnancy; P:memory; P:oviduct epithelium development; P:startle response | disease=",
    "[O75122] CLIP-associating protein 2 | genes=CLASP2 | function=Microtubule plus-end tracking protein that promotes the stabilization of dynamic microtubules (PubMed:26003921). Involved in the nucleation of noncentrosomal microtubules originating from the trans-Golgi network (TGN). Required for the polarization of the cytoplasmic microtubule arrays in migrating cells towards the leading edge of the cell. May act at the cell cortex to enhance the frequency of rescue of depolymerizing microtubules by attaching their plus-ends to cortical platforms composed of ERC1 and PHLDB2 (PubMed:16824950). This cortical microtubule stabilizing activity is regulated at least in part by phosphatidylinositol 3-kinase signaling. Also performs a similar stabilizing function at the kinetochore which is essential for the bipolar alignment of chromosomes on the mitotic spindle (PubMed:16866869, PubMed:16914514). Acts as a mediator of ERBB2-dependent stabilization of microtubules at the cell cortex | GO=C:axonal growth cone; C:basal cortex; C:cell cortex; C:cell leading edge; C:centrosome; C:cortical microtubule cytoskeleton; C:cortical microtubule plus-end; C:cytoplasm; C:cytoplasmic microtubule; C:cytosol | disease=",
    "[O95399] Urotensin-2 | genes=UTS2 | function=Highly potent vasoconstrictor | GO=C:extracellular region; C:extracellular space; C:synapse; F:hormone activity; F:signaling receptor binding; P:blood vessel diameter maintenance; P:chemical synaptic transmission; P:muscle contraction; P:regulation of blood pressure | disease=",
    "[P49335] POU domain, class 3, transcription factor 4 | genes=POU3F4 | function=Probable transcription factor which exert its primary action widely during early neural development and in a very limited set of neurons in the mature brain | GO=C:chromatin; C:nucleus; F:DNA-binding transcription factor activity; F:DNA-binding transcription factor activity, RNA polymerase II-specific; F:RNA polymerase II cis-regulatory region sequence-specific DNA binding; F:sequence-specific double-stranded DNA binding; P:brain development; P:cochlea morphogenesis; P:negative regulation of mesenchymal cell apoptotic process; P:regulation of transcription by RNA polymerase II | disease=Deafness, X-linked, 2",
    "[P35372] Mu-type opioid receptor | genes=OPRM1 | function=Receptor for endogenous opioids such as beta-endorphin and endomorphin (PubMed:10529478, PubMed:12589820, PubMed:7891175, PubMed:7905839, PubMed:7957926, PubMed:9689128). Receptor for natural and synthetic opioids including morphine, heroin, DAMGO, fentanyl, etorphine, buprenorphin and methadone (PubMed:10529478, PubMed:10836142, PubMed:12589820, PubMed:19300905, PubMed:7891175, PubMed:7905839, PubMed:7957926, PubMed:9689128). Also activated by enkephalin peptides, such as Met-enkephalin or Met-enkephalin-Arg-Phe, with higher affinity for Met-enkephalin-Arg-Phe (By similarity). Agonist binding to the receptor induces coupling to an inactive GDP-bound heterotrimeric G-protein complex and subsequent exchange of GDP for GTP in the G-protein alpha subunit leading to dissociation of the G-protein complex with the free GTP-bound G-protein alpha and the G-protein beta-gamma dimer activating downstream cellular effectors (PubMed:7905839). The agonist- and cell type-specific activity is predominantly coupled to pertussis toxin-sensitive G(i) and G(o) G alpha proteins, GNAI1, GNAI2, GNAI3 and GNAO1 isoforms Alpha-1 and Alpha-2, and to a lesser extent to pertussis toxin-insensitive G alpha proteins GNAZ and GNA15 (PubMed:12068084). They mediate an array of downstream cellular responses, including inhibition of adenylate cyclase activity and both N-type and L-type calcium channels, activation of inward rectifying potassium channels, mitogen-activated protein kinase (MAPK), phospholipase C (PLC), phosphoinositide/protein kinase (PKC), phosphoinositide 3-kinase (PI3K) and regulation of NF-kappa-B (By similarity). Also couples to adenylate cyclase stimulatory G alpha proteins (By similarity). The selective temporal coupling to G-proteins and subsequent signaling can be regulated by RGSZ proteins, such as RGS9, RGS17 and RGS4 (By similarity). Phosphorylation by members of the GPRK subfamily of Ser/Thr protein kinases and association with beta-arrestins is involved in short-term receptor desensitization (By similarity). Beta-arrestins associate with the GPRK-phosphorylated receptor and uncouple it from the G-protein thus terminating signal transduction (By similarity). The phosphorylated receptor is internalized through endocytosis via clathrin-coated pits which involves beta-arrestins (By similarity). The activation of the ERK pathway occurs either in a G-protein-dependent or a beta-arrestin-dependent manner and is regulated by agonist-specific receptor phosphorylation (By similarity). Acts as a class A G-protein coupled receptor (GPCR) which dissociates from beta-arrestin at or near the plasma membrane and undergoes rapid recycling (By similarity). Receptor down-regulation pathways are varying with the agonist and occur dependent or independent of G-protein coupling (By similarity). Endogenous ligands induce rapid desensitization, endocytosis and recycling (By similarity). Heterooligomerization with other GPCRs can modulate agonist binding, signaling and trafficking properties (By similarity); Couples to GNAS and is proposed to be involved in excitatory effects; Does not bind agonists but may act through oligomerization with binding-competent OPRM1 isoforms and reduce their ligand binding activity; Does not bind agonists but may act through oligomerization with binding-competent OPRM1 isoforms and reduce their ligand binding activity | GO=C:axon; C:dendrite; C:endoplasmic reticulum; C:endosome; C:Golgi apparatus; C:neuron projection; C:perikaryon; C:plasma membrane; C:synapse; F:beta-endorphin receptor activity | disease=",
    "[Q9NR22] Protein arginine N-methyltransferase 8 | genes=PRMT8 | function=S-adenosyl-L-methionine-dependent and membrane-associated arginine methyltransferase that can both catalyze the formation of omega-N monomethylarginine (MMA) and asymmetrical dimethylarginine (aDMA) in proteins such as NIFK, myelin basic protein, histone H4, H2A and H2A/H2B dimer (PubMed:16051612, PubMed:17925405, PubMed:26529540, PubMed:26876602). Able to mono- and dimethylate EWS protein; however its precise role toward EWS remains unclear as it still interacts with fully methylated EWS (PubMed:18320585) | GO=C:cytoplasmic side of plasma membrane; C:plasma membrane; F:enzyme binding; F:histone H4 methyltransferase activity; F:histone methyltransferase activity; F:identical protein binding; F:protein homodimerization activity; F:protein-arginine omega-N asymmetric methyltransferase activity; F:protein-arginine omega-N monomethyltransferase activity; F:S-adenosyl-L-methionine binding | disease=",
    "[Q15761] Neuropeptide Y receptor type 5 | genes=NPY5R | function=Receptor for neuropeptide Y and peptide YY. The activity of this receptor is mediated by G proteins that inhibit adenylate cyclase activity. Seems to be associated with food intake. Could be involved in feeding disorders | GO=C:GABA-ergic synapse; C:neuron projection; C:plasma membrane; C:presynapse; F:neuropeptide binding; F:neuropeptide Y receptor activity; F:pancreatic polypeptide receptor activity; F:peptide YY receptor activity; P:cardiac left ventricle morphogenesis; P:chemical synaptic transmission | disease=",
    "[Q9P0X4] Voltage-dependent T-type calcium channel subunit alpha-1I | genes=CACNA1I | function=Voltage-sensitive calcium channels (VSCC) mediate the entry of calcium ions into excitable cells and are also involved in a variety of calcium-dependent processes, including muscle contraction, hormone or neurotransmitter release, gene expression, cell motility, cell division and cell death. This channel gives rise to T-type calcium currents. T-type calcium channels belong to the 'low-voltage activated (LVA)' group and are strongly blocked by nickel and mibefradil. A particularity of this type of channels is an opening at quite negative potentials, and a voltage-dependent inactivation. T-type channels serve pacemaking functions in both central neurons and cardiac nodal cells and support calcium signaling in secretory cells and vascular smooth muscle. They may also be involved in the modulation of firing patterns of neurons which is important for information processing as well as in cell growth processes. Gates in voltage ranges similar to, but higher than alpha 1G or alpha 1H; Voltage-sensitive calcium channels (VSCC) mediate the entry of calcium ions into excitable cells and are also involved in a variety of calcium-dependent processes, including muscle contraction, hormone or neurotransmitter release, gene expression, cell motility, cell division and cell death. This channel gives rise to T-type calcium currents; Voltage-sensitive calcium channels (VSCC) mediate the entry of calcium ions into excitable cells and are also involved in a variety of calcium-dependent processes, including muscle contraction, hormone or neurotransmitter release, gene expression, cell motility, cell division and cell death. This channel gives rise to T-type calcium currents | GO=C:plasma membrane; C:voltage-gated calcium channel complex; F:high voltage-gated calcium channel activity; F:voltage-gated calcium channel activity; P:calcium ion import across plasma membrane; P:neuronal action potential; P:signal transduction; P:sleep | disease=Neurodevelopmental disorder with speech impairment and with or without seizures",
    "[Q9UH03] Neuronal-specific septin-3 | genes=SEPTIN3 | function=Filament-forming cytoskeletal GTPase (By similarity). May play a role in cytokinesis (Potential) | GO=C:cell division site; C:microtubule cytoskeleton; C:presynapse; C:septin complex; C:septin ring; F:GTP binding; F:GTPase activity; F:identical protein binding; F:molecular adaptor activity; P:cytoskeleton-dependent cytokinesis | disease=",
    "[Q9Y6K8] Adenylate kinase isoenzyme 5 | genes=AK5 | function=Nucleoside monophosphate (NMP) kinase that catalyzes the reversible transfer of the terminal phosphate group between nucleoside triphosphates and monophosphates. Active on AMP and dAMP with ATP as a donor. When GTP is used as phosphate donor, the enzyme phosphorylates AMP, CMP, and to a small extent dCMP. Also displays broad nucleoside diphosphate kinase activity | GO=C:centriolar satellite; C:cytoplasm; C:cytosol; F:AMP kinase activity; F:ATP binding; F:nucleoside diphosphate kinase activity; P:ADP biosynthetic process; P:ATP metabolic process; P:dADP biosynthetic process; P:pyrimidine ribonucleotide biosynthetic process | disease=",
    "[Q9HAB3] Solute carrier family 52, riboflavin transporter, member 2 | genes=SLC52A2 | function=Plasma membrane transporter mediating the uptake by cells of the water soluble vitamin B2/riboflavin that plays a key role in biochemical oxidation-reduction reactions of the carbohydrate, lipid, and amino acid metabolism (PubMed:20463145, PubMed:22864630, PubMed:23243084, PubMed:24253200, PubMed:27702554). Humans are unable to synthesize vitamin B2/riboflavin and must obtain it via intestinal absorption (PubMed:20463145). May also act as a receptor for 4-hydroxybutyrate (Probable); (Microbial infection) In case of infection by retroviruses, acts as a cell receptor to retroviral envelopes similar to the porcine endogenous retrovirus (PERV-A) | GO=C:plasma membrane; F:4-hydroxybutyrate receptor activity; F:riboflavin transmembrane transporter activity; F:virus receptor activity; P:riboflavin metabolic process; P:riboflavin transport | disease=Brown-Vialetto-Van Laere syndrome 2",
    "[P28335] 5-hydroxytryptamine receptor 2C | genes=HTR2C | function=G-protein coupled receptor for 5-hydroxytryptamine (serotonin) (PubMed:12970106, PubMed:18703043, PubMed:19057895, PubMed:29398112, PubMed:7895773). Also functions as a receptor for various drugs and psychoactive substances, including ergot alkaloid derivatives, 1-2,5,-dimethoxy-4-iodophenyl-2-aminopropane (DOI) and lysergic acid diethylamide (LSD) (PubMed:19057895, PubMed:29398112). Ligand binding causes a conformation change that triggers signaling via guanine nucleotide-binding proteins (G proteins) and modulates the activity of downstream effectors (PubMed:18703043, PubMed:29398112). HTR2C is coupled to G(q)/G(11) G alpha proteins and activates phospholipase C-beta, releasing diacylglycerol (DAG) and inositol 1,4,5-trisphosphate (IP3) second messengers that modulate the activity of phosphatidylinositol 3-kinase and promote the release of Ca(2+) ions from intracellular stores, respectively (PubMed:18703043, PubMed:29398112). Beta-arrestin family members inhibit signaling via G proteins and mediate activation of alternative signaling pathways (PubMed:29398112). Regulates neuronal activity via the activation of short transient receptor potential calcium channels in the brain, and thereby modulates the activation of pro-opiomelanocortin neurons and the release of CRH that then regulates the release of corticosterone (By similarity). Plays a role in the regulation of appetite and eating behavior, responses to anxiogenic stimuli and stress (By similarity). Plays a role in insulin sensitivity and glucose homeostasis (By similarity) | GO=C:dendrite; C:G protein-coupled serotonin receptor complex; C:plasma membrane; C:synapse; F:1-(4-iodo-2,5-dimethoxyphenyl)propan-2-amine binding; F:G protein-coupled serotonin receptor activity; F:Gq/11-coupled serotonin receptor activity; F:identical protein binding; F:neurotransmitter receptor activity; F:serotonin binding | disease=",
    "[Q02446] Transcription factor Sp4 | genes=SP4 | function=Binds to GT and GC boxes promoters elements. Probable transcriptional activator | GO=C:chromatin; C:cytosol; C:nucleoplasm; F:DNA-binding transcription factor activity, RNA polymerase II-specific; F:identical protein binding; F:RNA polymerase II cis-regulatory region sequence-specific DNA binding; F:sequence-specific DNA binding; F:zinc ion binding; P:regulation of transcription by RNA polymerase II | disease="
  ],
  "outsrc_criteria": [
    "Van Maldergem syndrome 2",
    "Hennekam lymphangiectasia-lymphedema syndrome 2",
    "Inhibition of neuroprogenitor cell proliferation and differentiation",
    "Tremor, hereditary essential 1",
    "Schizophrenia",
    "Substance use disorders",
    "Uncontrolled cell proliferation (through MAP kinase signaling)",
    "Neurodevelopmental disorder, mitochondrial, with abnormal movements and lactic acidosis, with or without seizures",
    "Parkinsonism-dystonia 3, childhood-onset",
    "Snijders Blok-Fisher syndrome",
    "Intellectual developmental disorder, autosomal dominant 62",
    "Spinocerebellar ataxia 12",
    "Proapoptotic activity leading to neuronal death",
    "Neuronopathy, distal hereditary motor, autosomal dominant 14",
    "Amyotrophic lateral sclerosis",
    "Perry syndrome",
    "Intellectual developmental disorder, autosomal dominant 10",
    "Neurodevelopmental disorder with microcephaly, ataxia, and seizures",
    "Intellectual developmental disorder with autism and speech delay",
    "Increased formation of amyloid-beta (APP-beta)",
    "Pitt-Hopkins-like syndrome 2",
    "Schizophrenia 17",
    "Host entry factor for influenza virus",
    "Host entry factor for rabies virus",
    "Host entry factor for SARS-CoV-2",
    "Neurodevelopmental disorder, non-progressive, with spasticity and transient opisthotonus",
    "Intellectual developmental disorder with neuropsychiatric features",
    "Type 2 diabetes mellitus",
    "Dystonia 2, torsion, autosomal recessive",
    "Major depressive disorder",
    "Attention deficit-hyperactivity disorder 7",
    "Microcephaly-micromelia syndrome",
    "Microcephaly, short stature, and limb abnormalities",
    "Disruption of genome stability",
    "Disruption of cell cycle checkpoints",
    "Waardenburg syndrome 2E",
    "Waardenburg syndrome 4C",
    "Peripheral demyelinating neuropathy, central dysmyelinating leukodystrophy, Waardenburg syndrome and Hirschsprung disease",
    "Highly potent vasoconstriction",
    "Deafness, X-linked, 2",
    "Opioid addiction/dependence",
    "Opioid-related side effects (e.g., respiratory depression, constipation, sedation)",
    "Involvement in feeding disorders",
    "Neurodevelopmental disorder with speech impairment and with or without seizures",
    "Off-target effects on cardiac nodal cells/pacemaking functions",
    "Brown-Vialetto-Van Laere syndrome 2",
    "Acting as a receptor for retroviruses (e.g., PERV-A)",
    "Association with psychoactive substances (e.g., ergot alkaloids, DOI, LSD)",
    "Psychotropic side effects (e.g., hallucinations)",
    "Alterations in appetite and eating behavior",
    "Altered responses to anxiogenic stimuli and stress",
    "Impact on insulin sensitivity and glucose homeostasis"
  ],
  "_master_prompt": "**Objective:** Design a drug for cognitive enhancement by identifying and modulating specific protein targets within the brain.\n\n**Organ Context:** The intervention must exclusively target and impact brain-specific biological pathways and cellular functions to improve cognitive abilities.\n\n**Functional Scope (Inclusion Criteria):** Prioritize targets and modulation strategies that directly or indirectly:\n1.  **Enhance Synaptic Plasticity & Efficacy:** Promote synaptogenesis, strengthen neurotransmission, optimize AMPA and NMDA receptor trafficking/gating, and support the formation and maintenance of dendritic spines and synapses.\n2.  **Improve Memory & Learning:** Facilitate processes related to contextual memory, associative learning, and general memory acquisition and retention.\n3.  **Support Neuronal Development & Structure:** Contribute positively to neuronal migration, laminar and areal identity, axonogenesis, microtubule dynamics, and the structural plasticity of neurons.\n4.  **Optimize Neural Circuit Balance:** Regulate the excitatory-inhibitory balance within neural circuits.\n5.  **Modulate Neuronal Excitability & Information Processing:** Influence voltage-dependent calcium channels to fine-tune neuronal firing patterns and support calcium signaling crucial for information processing.\n6.  **Promote Myelination:** Support the development and maintenance of myelin to enhance the speed and efficiency of signal transduction in the CNS.\n7.  **Boost Brain Energy Metabolism:** Improve processes related to ATP synthesis and utilization necessary for sustained cognitive function.\n8.  **Regulate Cognition-Related Gene Expression:** Act as transcription factors or modulators of gene expression that underpin enhanced cognitive functions.\n\n**Exclusion Guardrails (STRICTLY AVOID ANY TARGET OR MODULATION LEADING TO):**\n*   **Neurodegenerative/Neurodevelopmental Disorders:** Van Maldergem syndrome 2, Hennekam lymphangiectasia-lymphedema syndrome 2, Tremor hereditary essential 1, Schizophrenia, Parkinsonism-dystonia 3 childhood-onset, Snijders Blok-Fisher syndrome, Intellectual developmental disorder (autosomal dominant 62, 10, with autism and speech delay, with neuropsychiatric features), Spinocerebellar ataxia 12, Neuronopathy distal hereditary motor autosomal dominant 14, Amyotrophic lateral sclerosis, Perry syndrome, Neurodevelopmental disorder with microcephaly ataxia and seizures, Pitt-Hopkins-like syndrome 2, Neurodevelopmental disorder non-progressive with spasticity and transient opisthotonus, Dystonia 2 torsion autosomal recessive, Major depressive disorder, Attention deficit-hyperactivity disorder 7, Microcephaly-micromelia syndrome, Microcephaly short stature and limb abnormalities, Waardenburg syndrome (2E, 4C), Peripheral demyelinating neuropathy central dysmyelinating leukodystrophy Waardenburg syndrome and Hirschsprung disease, Deafness X-linked 2, Neurodevelopmental disorder with speech impairment and with or without seizures, Brown-Vialetto-Van Laere syndrome 2.\n*   **Undesirable Biological Effects:** Inhibition of neuroprogenitor cell proliferation and differentiation, Uncontrolled cell proliferation (MAP kinase signaling), Proapoptotic activity leading to neuronal death, Increased formation of amyloid-beta (APP-beta), Disruption of genome stability, Disruption of cell cycle checkpoints, Highly potent vasoconstriction, Opioid addiction/dependence, Opioid-related side effects (e.g., respiratory depression, constipation, sedation), Involvement in feeding disorders, Off-target effects on cardiac nodal cells/pacemaking functions, Acting as a receptor for retroviruses (e.g., PERV-A), Association with psychoactive substances (e.g., ergot alkaloids, DOI, LSD), Psychotropic side effects (e.g., hallucinations), Alterations in appetite and eating behavior, Altered responses to anxiogenic stimuli and stress, Impact on insulin sensitivity and glucose homeostasis.\n\n**Gemini Model Task:**\nIdentify a maximum of **5** primary protein targets from the provided list that are most promising for developing a cognitive enhancement drug. For each chosen target:\n1.  **Propose a specific drug modulation strategy** (e.g., agonist, antagonist, allosteric modulator, expression enhancer/inhibitor) and clearly justify *how* this modulation would lead to cognitive enhancement, referencing its specific functions from the provided annotations.\n2.  **Explicitly demonstrate how each chosen target and proposed modulation strategy strictly adheres to all inclusion criteria and avoids all exclusion criteria.** This requires a detailed explanation linking the target's function to desired effects and providing specific justifications for safety and specificity in relation to each listed excluded condition or undesirable effect.",
  "_tokens": [
    "drug"
  ],
  "_db_targets": [
    "chemical"
  ]
}

Process finished with exit code 0




"""



