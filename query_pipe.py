from __future__ import annotations

import asyncio
import functools
import json
from typing import Any
from gem_core import Gem
import httpx
from graph import GUtils
from protein import get_protein_sequence

_TISSLIST_URL = "https://www.uniprot.org/docs/tisslist.txt"
_UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
_PUBCHEM_COMPOUND = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/JSON"

_HTTP_TIMEOUT = 30
_UNIPROT_FIELDS = "accession,protein_name,gene_names,cc_function,go,cc_disease,cc_tissue_specificity"
_MAX_PROTEINS_PER_ORGAN = 50

gem = Gem()

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



async def fetch_up_entries(organs: list[str], query) -> list[str]:
    """Query UniProt REST per organ → aggregate functional annotation strings."""
    annotations: list[str] = []

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        for organ in organs:
            query = query + f'(cc_tissue_specificity:"{organ}") AND (reviewed:true) AND (organism_id:9606)'
            params = {
                "query": query,
                "fields": "primaryAccession",
                "format": "json",
                "size": str(500),
            }
            try:
                r = await client.get(_UNIPROT_SEARCH, params=params)
                r.raise_for_status()
                data = r.json()
            except (httpx.HTTPError, json.JSONDecodeError):
                continue

            for item in data["results"]:
                annotations.append(item["primaryAccession"])
    return annotations


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
    }


#####################################


def stage1_deconstruct_prompt(prompt: str) -> dict[str, Any]:
    """
    Stage 1:
    Split the raw user prompt into biological tokens and produce a structured
    transformed_text technical brief for all later stages.
    """

    prompt = f"""
    You are a senior bioinformatics prompt engineer. Given an informal or clinical biological user prompt, you (1) extract tokens for lexical downstream use and (2) write transformed_text as a modular technical brief.

    Infer the user's underlying intent even when the wording is vague (e.g. colloquial drug requests → molecular targets, pathways, safety constraints). Expand implicit scope only when biologically standard (default human/clinical context when unspecified). Keep each MODULAR_TASKS bullet a single clear action boundary so independent workflow steps could implement it without rereading the original prompt.

    Do not emit per-token database labels; fixed execution categories for this project are:.
    Category reference (keywords only):

    Return ALL fields via the split_transform_classify function. transformed_text MUST follow the section headers and order given in the tool schema for transformed_text.\

    INPUT:
    {prompt}
    Return jsut comma sepparated keywords for a uniprot search, nothing else
    """

    result = gem.ask(prompt)
    result = result.split(",")

    return {
        "tokens": list(result.get("tokens", [])),
        "transformed_text": result.get("transformed_text", prompt),
    }


def stage1b_resolve_physical_filter(
    filter_physical_compound: str | list[str] | None,
    physical_aliases: dict[str, list[str]],
) -> list[str]:
    """
    Stage 1b:
    Resolve user-provided physical compound filters to canonical physical
    categories from ds.PHYSICAL_CATEGORY_ALIASES.

    Example:
    input: ["drug", "chemical"]
    output: ["compound"]
    """
    if not filter_physical_compound:
        return []

    raw_items = (
        [filter_physical_compound]
        if isinstance(filter_physical_compound, str)
        else filter_physical_compound
    )

    resolved: set[str] = set()

    for item in raw_items:
        item_norm = item.lower().strip()

        for canonical, aliases in physical_aliases.items():
            alias_set = {canonical.lower(), *(a.lower() for a in aliases)}
            if item_norm in alias_set:
                resolved.add(canonical)
    return sorted(resolved)


def get_organs(
    prompt: str,
) -> list[str]:
    tissue_vocab = _fetch_tissue_vocabulary()
    vocab_excerpt = "\n".join(tissue_vocab) # [:2000]

    prompt = f"""\
    You are a biomedical organ/tissue classifier. Return only exact entries from the vocabulary.

    CANONICAL TISSUE VOCABULARY:
    {vocab_excerpt}

    Original prompt: {prompt}
    return all organs sepparated by comma (,)
    """

    result = gem.ask(prompt)
    return result.split(",")


def summarize_function_annotations(
    organs: list[str],
    raw_prompt: str,
) -> list[str]:
    prompt = f"""\
    You are a protein-function summarisation engine. Deduplicate, group GO terms, remove redundancy, keep entries short.

    Query context: {raw_prompt}
    Organs: {json.dumps(organs)}
    return all functional annotations sepparated by comma (,)
    """

    result = gem.ask(prompt)
    return result.strip().split(",")


def extract_outsrc_criteria(
    prompt: str,
    organs: list[str],
) -> list[str]:
    # Functional annotations: {json.dumps(function_annotation)}
    exclusion_prompt = f"""
    You are a biomedical risk/exclusion analyst. 
    Extract all relevant exclusion criteria and unwanted outcomes.

    Original prompt: {prompt}
    Organs: {json.dumps(organs)}
    
    return all outsrc criteria points sepparated by comma (,)
    """

    result = gem.ask(exclusion_prompt)
    return result.split(",")


def build_master_prompt(
    prompt: str,
    organs: list[str],
    function_annotation: list[str],
    outsrc_criteria: list[str],
) -> str:
    """
    Stage 5:
    Compose final enriched master prompt for downstream Gemini workflow execution.
    """
    master_prompt = f"""\
    You are a master-prompt composer for a bioinformatics pipeline. Output only the final actionable prompt.

    Original user query: {prompt}

    Organs: {json.dumps(organs)}
    Functional annotations: {json.dumps(function_annotation)}
    Exclusion criteria: {json.dumps(outsrc_criteria)}

    Compose the master prompt now.\
    """
    return gem.ask(master_prompt)

def extract_keywords(master_prompt:str):
    prompt = f"""
    Use the given input to generate a bunch of keyworkds useful for a uniprot search.
    retunr all keywords separated by comma, nothing else.
    
    INPUT:
    {master_prompt}
    """
    return gem.ask(prompt).split(",")


async def create_graph_process(id_map:list[str]):
    async def get_string_graph(protein, client: httpx.AsyncClient):
        url = "https://string-db.org/api/json/network"
        params = {
            "identifiers": protein,
            "species": 9606
        }
        try:
            response = await client.get(url, params=params)
            return response.json()
        except Exception as e:
            print(f"Error fetching STRING data: {e}")
            return None

    def load_string_graph(g: GUtils, data, min_score=0.7):
        print("working STRING...")

        for e in data:
            if e.get("score", 0) < min_score:
                continue

            a = e["preferredName_A"]
            b = e["preferredName_B"]

            # add nodes
            g.add_node(dict(id=a, type="PROTEIN"))
            g.add_node(dict(id=b, type="PROTEIN"))

            # add edge
            g.add_edge(
                a,
                b,
                attr=dict(
                    rel="interacts_with",
                    src_layer="PROTEIN",
                    trgt_layer="PROTEIN",
                )
            )
        return g

    g = GUtils()

    client = httpx.AsyncClient()
    for fun_protein_map in id_map:
        protein_graph = [
            await asyncio.gather(
                *[
                    get_string_graph(p, client)
                    for p in fun_protein_map
                ]
            )
        ]

        for item in protein_graph:
            load_string_graph(g=g, data=item)

        # get sequence from rest
        result:list[tuple[str, dict]] = await asyncio.gather(
            *[
                get_protein_sequence(p["id"], client)
                for p in g.get_nodes(filter_key="type", filter_value="PROTEIN")
            ]
        )
        for pid, seq in result:
            g.G.nodes[pid].update(seq)
    g.print_status_G()
    print("get_human_entries... done")


def run_query_pipe(
    prompt: str,
    organs: list[str],
) -> GUtils:
    """
    organs
    functional annotations
    outsrc criteria
    keywords (uniprot search)
    stringdb
    return G
    """

    function_annotation = summarize_function_annotations(
        organs=organs,
        raw_prompt=prompt,
    )

    #
    outsrc_criteria = extract_outsrc_criteria(
        prompt=prompt,
        organs=organs,
    )

    master_prompt = build_master_prompt(
        prompt=prompt,
        organs=organs,
        function_annotation=function_annotation,
        outsrc_criteria=outsrc_criteria,
    )

    keywords = extract_keywords(
        master_prompt
    )

    up_ids:list[str] = asyncio.run(fetch_up_entries(
        organs,
        keywords
    ))

    # get string
    g = asyncio.run(create_graph_process(id_map=up_ids))
    return g



if __name__ == "__main__":
    run_query_pipe(
        prompt="cognitive enhancement",
        organs=["Brain"],
    )

