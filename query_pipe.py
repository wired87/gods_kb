from __future__ import annotations
import asyncio
import httpx

from app_utils import BRAIN_TERMS
from firegraph.graph.local_graph_utils import GUtils
from keyword_handler import build_keyword_graph, extract_keywords, get_proteins_from_keywords
from protein import get_protein_sequence


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






async def create_interaction_process(g):
    print("get_human_entries...")
    id_map: list[str] = [
        key
        for key, data in g.G.nodes(data=True)
        if data.get("type") == "PROTEIN"
    ]

    async def get_string_graph(protein, client: httpx.AsyncClient):
        print("get_string_graph for ", protein, " ...")
        url = "https://string-db.org/api/json/network"
        params = {
            "identifiers": protein,
            "species": 9606
        }
        try:
            response = await client.get(url, params=params)#
            item = response.json()
            print("item", item)
            return item
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
            if not g.get_node(a):
                g.add_node(dict(id=a, type="PROTEIN", sub_type="PURE_INFLUENCE"))
            if not g.get_node(b):
                g.add_node(dict(id=b, type="PROTEIN", sub_type="PURE_INFLUENCE"))

            # add edge
            g.add_edge(
                a,
                b,
                attrs=dict(
                    rel="interacts_with",
                    src_layer="PROTEIN",
                    trgt_layer="PROTEIN",
                )
            )
        print("interaction added")
        return g

    g = GUtils()
    client = httpx.AsyncClient()

    protein_graph = await asyncio.gather(
        *[
            get_string_graph(p, client)
            for p in id_map
        ]
    )

    for item in protein_graph:
        load_string_graph(g=g, data=item)

    """
    # get sequence from rest
    result: list[tuple[str, dict]] = await asyncio.gather(
        *[
            get_protein_sequence(p["id"], client)
            for p in g.get_nodes(filter_key="type", filter_value="PROTEIN")
        ]
    )

    for pid, seq in result:
        g.G.nodes[pid].update(seq)
    """

    g.print_status_G()
    print("get_human_entries... done")

def build_fun_annotations(g, fun_annotations:list[str]):
    for entry in fun_annotations:
        g.add_node(
            attrs=dict(
                id=entry,
                type="FUNCTION_ANNOTATION",
                embed_key="id",
            )
        )
    print("build_fun_annotations... done")


def include_organs(g, organs:list[str]):
    for entry in organs:
        g.add_node(
            attrs=dict(
                id=entry,
                type="ORGAN",
                embed_key="id",
            )
        )
    print("build_fun_annotations... done")


def run_query_pipe(
    function_annotations: list[str],
    organs: list[str] = BRAIN_TERMS,
    outsrc_criteria:list[str]=None,
) -> GUtils:
    """
    organs
    functional annotations
    outsrc criteria
    keywords (uniprot search)
    stringdb
    return G
    """
    g = GUtils()

    # FUN DEF -> G
    build_fun_annotations(g, function_annotations)
    include_organs(g, organs)

    # GET KEYWORDS
    build_keyword_graph(g)

    # XTRACT RELEVANT KEYWORDS
    extract_keywords(g)

    # EXTRACT PROTEINS FROM GIVEN KEYWORDS
    asyncio.run(get_proteins_from_keywords(g))

    # PP INTERACTION
    asyncio.run(create_interaction_process(g))



    # todo next version
    # handle DISEASE (patient tells alergikum etc -> recognize: in conection to upregulatin of protein A -> find inhibitor AND delete protein from graph)
    # handle outsrc criteria

    dest_file="data/result.json"
    g.save_graph(dest_file)
    return g



if __name__ == "__main__":
    run_query_pipe(
        function_annotations=[
            "learning, memory, brain"
        ]
    )


