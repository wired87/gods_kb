import numpy as np

from embedder import embed, similarity, embed_batch
from go_term import go_term_graph
from firegraph.graph.local_graph_utils import GUtils
import asyncio
from typing import Any, Dict, List
import aiohttp
from aiolimiter import AsyncLimiter
import json

from utils.deserialize import deserialize

# Define a safe rate limit for UniProt API (e.g., max 4 requests per second)
UNIPROT_LIMITER = AsyncLimiter(max_rate=4, time_period=1.0)


async def fetch_uniprot_protein(session: aiohttp.ClientSession, keyword: str, organs:list[str]) -> List[Dict[str, Any]]:
    """
    Asynchronously queries UniProt for a single keyword under human (9606) restriction.
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    organ_query = " OR ".join(
        f'tissue:{o}'
        for o in organs
    )

    raw_query = (
        f'(organism_id:9606) '
        f'AND (keyword:"{keyword.strip()}") '
        f'AND {organ_query}'
    )

    params = {
        "query": raw_query,
        "format": "json",
        "size": 500
    }

    async with UNIPROT_LIMITER:
        try:
            async with session.get(base_url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("results", [])

        except Exception as e:
            print(f"Error fetching UniProt keyword '{keyword}': {e}")
            return []


def matches_target_organs(protein_data: Dict[str, Any], target_organs: List[str]) -> bool:
    """
    Parses UniProt's comment blocks to see if the protein's tissue
    specificity matches any organ in the target list.
    """
    # If no target organ filter is provided, let everything through
    if not target_organs:
        return True

    comments = protein_data.get("comments", [])

    # Normalize target organs to lowercase for case-insensitive matching
    target_organs_lower = [organ.lower().strip() for organ in target_organs]

    for comment in comments:
        # We specifically want to look at the "TISSUE SPECIFICITY" text blocks
        if comment.get("commentType") == "TISSUE SPECIFICITY":
            texts = comment.get("texts", [])
            for text_obj in texts:
                value = text_obj.get("value", "").lower()

                # Check if any target organ string exists inside the text narrative
                if any(organ in value for organ in target_organs_lower):
                    return True
    return False


async def get_proteins_from_keywords(
    g:GUtils,
) -> List[Any]:
    """
    Receives a list of keywords and a list of target organs, runs queries concurrently,
    and returns a filtered list of matching human protein payloads.

    """
    organs = [key for key, attrs in g.G.nodes(data=True) if attrs.get("type") == "ORGAN"]
    keywords = [key for key, attrs in g.G.nodes(data=True) if attrs.get("type") == "KEYWORD"]

    if not keywords:
        return []

    async with aiohttp.ClientSession() as session:
        # Schedule concurrent API requests
        tasks = [fetch_uniprot_protein(session, kw, organs) for kw in keywords]
        batched_results = await asyncio.gather(*tasks)

        # Track unique accessions to avoid duplicate entries across different keyword results
        seen_accessions = set()

        for result_list, keyword in zip(batched_results, keywords):
            if not result_list:
                continue

            for protein in result_list:
                accession = protein.get("primaryAccession")

                if accession and accession not in seen_accessions:
                    seen_accessions.add(accession)
                    g.add_node(
                        attrs=dict(
                            id=accession,
                            type="PROTEIN",
                            **protein,
                            embed_key="proteinDescription__recommendedName__fullName__value"
                        )
                    )
                    g.add_edge(
                        src=keyword,
                        trgt=accession,
                        attrs=dict(
                            rel="keyword",
                            src_layer="KEYWORD",
                            trgt_layer="PROTEIN",
                        )
                    )
    g.print_status_G()
    print("protein query finished...")


def build_keyword_graph(
    g:GUtils,
) -> list[str]:
    print("build_keyword_graph...")
    dest_file = "data/key_word_g.json"
    try:
        g.load_graph(dest_file)
    except Exception as e:
        print("Err load exsiitng Graph:", e, "fetch manually...")

        def walk_parents(center_id, parents):
            """
            Recursive parent traversal.
            parent -> center
            """

            for parent in parents:
                p = parent["keyword"]

                g.add_node(
                    attrs=dict(
                        id=p["id"],
                        name=p["name"],
                        type="KEYWORD",
                        embed_key='name',
                    )
                )
                # center -> child
                g.add_edge(
                    center_id,
                    p["id"],
                    attrs=dict(
                        rel="parent",
                        src_layer="KEYWORD",
                        trgt_layer="KEYWORD",
                    )
                )


                # recursive parent chain
                if "parents" in parent:
                    walk_parents(
                        p["id"],
                        parent["parents"]
                    )

        def walk_children(center_id, children):
            """
            Recursive child traversal.
            center -> child
            """

            for child in children:
                c = child["keyword"]
                g.add_node(
                    attrs=dict(
                        id=c["id"],
                        name=c["name"],
                        type="KEYWORD",
                        embed_key='name',
                    )
                )

                # center -> child
                g.add_edge(
                    center_id,
                    c["id"],
                    attrs=dict(
                        rel="children",
                        src_layer="KEYWORD",
                        trgt_layer="KEYWORD",
                    )
                )

                # recursive children
                if "children" in child:
                    walk_children(
                        c["id"],
                        child["children"]
                    )

        data = json.load(
            open(
                "data/key_words_uniprot.json",
                "r",
                encoding="utf-8"
            )
        )

        # -----------------------------
        # build expensive graph
        # -----------------------------
        for item in data["results"]:

            center = item["keyword"]

            # -------------------------
            # CENTER NODE
            # -------------------------
            g.add_node(
                attrs=dict(
                    id=center["id"],
                    name=center["name"],
                    definition=item.get("definition"),
                    type="KEYWORD",
                    embed_key='definition',
                )
            )

            # -------------------------
            # CATEGORY NODE
            # -------------------------
            if "category" in item:
                category = item["category"]

                g.add_node(
                    attrs=dict(
                        id=category["id"],
                        name=category["name"],
                        type="KEYWORD",
                        embed_key="name",
                    )
                )

                g.add_edge(
                    center["id"],
                    category["id"],
                    attrs=dict(
                        rel="category",
                        src_layer="KEYWORD",
                        trgt_layer="KEYWORD",
                    )
                )

            for go in item.get("geneOntologies", []):
                g.add_node(
                    attrs=dict(
                        id=go["goId"],
                        name=go["name"],
                        type="GO_TERM",
                        embed_key="name",
                    )
                )

                g.add_edge(
                    center["id"],
                    go["goId"],
                    attrs=dict(
                        rel="go_term",
                        src_layer="KEYWORD",
                        trgt_layer="GO_TERM",
                    )
                )
            # -------------------------
            # SYNONYMS
            # -------------------------
            for synonym in item.get("synonyms", []):
                syn_id = f"SYN::{synonym}"

                g.add_node(
                    attrs=dict(
                        id=syn_id,
                        name=synonym,
                        type="SYNONYM",
                        embed_key="name",
                    )
                )

                #
                g.add_edge(
                    center["id"],
                    syn_id,
                    attrs=dict(
                        rel="go_term",
                        src_layer="KEYWORD",
                        trgt_layer="SYNONYM",
                    )
                )

            # -------------------------
            # PARENTS
            # -------------------------
            walk_parents(
                center["id"],
                item.get("parents", [])
            )

            # -------------------------
            # CHILDREN
            # -------------------------
            walk_children(
                center["id"],
                item.get("children", [])
            )

        # HANDE GO TERMS
        g = asyncio.run(go_term_graph(g))
        #g.save_graph(dest_file)

    g.print_status_G()
    print("finished keyword graph buildup...")
    return g


def extract_keywords(g: Any, similarity_threshold: float = 0.7):
    # Get fun list
    fun_list = []
    for nid, attrs in g.G.nodes(data=True):
        if attrs.get("type") == "FUNCTION_ANNOTATION":
            fun_list.append((nid, attrs))

    print("fun_list", len(fun_list), fun_list)

    # embed funcitons
    fun_vec:np.array = embed_batch(
        [
            attrs[attrs["embed_key"]]
            if attrs["embed_key"] != "id" else nid
            for nid,  attrs in fun_list
        ]
    )

    key_word_nodes = [(nid, attrs) for nid, attrs in g.G.nodes(data=True) if attrs.get("type") != "FUNCTION_ANNOTATION"]
    filtered_keyword_nodes = []
    for k in key_word_nodes:
        if "embed_key" not in k[1]:
            continue
        else:
            filtered_keyword_nodes.append(k)

    # embed keywords
    key_vec: np.array = embed_batch(
        [
            attrs[attrs["embed_key"]]
            if attrs["embed_key"] != "id" else nid
            for nid, attrs in filtered_keyword_nodes
        ]
    )

    fun_vec_norms = np.linalg.norm(fun_vec, axis=1, keepdims=True)
    key_vec_norms = np.linalg.norm(key_vec, axis=1, keepdims=True)

    similarity_matrix = np.dot(fun_vec, key_vec.T) / (fun_vec_norms @ key_vec_norms.T)

    # Get true false arr
    above_threshold = np.any(similarity_matrix >= similarity_threshold, axis=0)

    # 5. Filtere die originalen Node-Daten basierend auf der True/False-Maske
    match_nodes = [(nid, attrs) for idx, (nid, attrs) in enumerate(filtered_keyword_nodes) if above_threshold[idx]]

    for nid, matched_node in match_nodes:
        print("match node", nid)
        # mark keep
        matched_node["match"] = True

        # Fetch neighbors using your custom utility framework
        neighbors = g.get_neighbor_list(nid)
        if neighbors:
            for nnid, nattrs in neighbors.items():
                g.G.nodes[nnid]["match"] = True

    keyword_ids = [nid for nid, attrs in g.G.nodes(data=True) if attrs.get("match") is True and attrs.get("type") == "KEYWORD"]
    all_keywords = [nid for nid, attrs in g.G.nodes(data=True) if attrs.get("type") == "KEYWORD"]

    for nid in all_keywords:
        if nid not in keyword_ids:
            g.delete_node(nid)
    print("keywords extracted:", keyword_ids)

