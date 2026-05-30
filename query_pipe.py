from __future__ import annotations
import asyncio

from app_utils import BRAIN_TERMS, QUERY_TRANSFORM_PROMPT, gem
from firegraph.graph.local_graph_utils import GUtils
from keyword_handler import build_keyword_graph, extract_keywords, get_proteins_from_keywords
from string_db import create_interaction_process
from tissue.tissue import link_tissue_to_sub_regions
from tissue.uberon import build_brain_tissue_subgraph

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
    organ: str = "Thalamus",
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

    if function_annotations is None:
        # perform query transformation
        asyncio.run(gem.aask_gem(content=QUERY_TRANSFORM_PROMPT))

    # Build validate and include uberon
    build_brain_tissue_subgraph(
        g,
        tissue=organ
    )

    # Build bridge Tissue -> Cell, Gene, ...
    link_tissue_to_sub_regions(
        g,
        paths=[
            "cell/asct-b-allen-brain.csv",
            "cell/asct-b-allen-brain (1).csv",
            "cell/asct-b-allen-brain (2).csv",
        ]
    )

    # fetch cell ontology details (cl -> pcl & reverse)


    # fetch genes & details


    # fetch




    # FUN DEF -> G
    build_fun_annotations(g, function_annotations)
    include_organs(g, [organ])

    # GET KEYWORDS
    asyncio.run(build_keyword_graph(g))

    # XTRACT RELEVANT KEYWORDS
    extract_keywords(g)

    # EXTRACT PROTEINS FROM GIVEN KEYWORDS
    asyncio.run(get_proteins_from_keywords(g))

    # PP INTERACTION
    asyncio.run(create_interaction_process(g))



    # todo next version
    # handle KEGG DISEASE-Datenbank, OMIM, Orphanet, Reactome (patient tells alergikum etc -> recognize: in conection to upregulatin of protein A -> find inhibitor AND delete protein from graph)
    # handle outsrc criteria

    # todo filter result for protein, peptide, enzyme


    dest_file="data/result.json"
    g.save_graph(dest_file)
    return g


if __name__ == "__main__":
    run_query_pipe(
        prompt="focused lerning",
    )


"""

[
    "learning",
    "memory",                                # GO:0007613 (Gedächtnis)
    "associative learning",                  # GO:0008306 (Assoziatives Lernen)
    "spatial learning",                      # GO:0043044 (Räumliches Lernen)
    "long-term memory",                      # GO:0007616 (Langzeitgedächtnis)
    "short-term memory",                     # GO:0007614 (Kurzzeitgedächtnis)
    "cognition",                             # GO:0050890 (Kognition allgemein)

    "synaptic plasticity",                   # GO:0048167 (Synaptische Plastizität)
    "long-term synaptic potentiation",       # GO:0060291 (LTP - Langzeitpotenzierung)
    "long-term synaptic depression",        # GO:0060292 (LTD - Langzeitdepression)
    "regulation of synaptic plasticity",     # GO:0048172 (Regulation der Plastizität)
    "synaptic vesicle exocytosis",           # GO:0016079 (Neurotransmitter-Ausschüttung)

    "brain development",                     # GO:0007420 (Gehirnentwicklung)
    "hippocampus development",               # GO:0021766 (Hippocampus-Entwicklung - Hauptort für Gedächtnis)
    "cerebral cortex development",           # GO:0021987 (Großhirnrinde-Entwicklung)
    "central nervous system development",    # GO:0007417 (ZNS-Entwicklung)
    "neurogenesis",                          # GO:0022008 (Neurogenese / Bildung neuer Neuronen)
    "synaptogenesis",                        # GO:0050808 (Synapsenbildung)
    "axon guidance",                         # GO:0007411 (Axonwachstum / Verschaltung)

    "chemical synaptic transmission",        # GO:0007268 (Chemische Signalübertragung)
    "neurotransmitter receptor transport",   # GO:0098962 (Rezeptortransport - kritisch für LTP)
    "glutamate receptor signaling pathway",  # GO:0007215 (Glutamat-Signalweg - der Hauptlern-Botenstoff)
    "calcium-mediated signaling",            # GO:0019722 (Kalzium-Signale - steuern plastische Veränderungen)
]
"""

