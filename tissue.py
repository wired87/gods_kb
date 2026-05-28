from app_utils import _HTTP_TIMEOUT, _TISSLIST_URL, _UBERON_URL
from embedder import embed

import functools
import httpx
import obonet
import networkx as nx
import re

def norm(x: str) -> str:
    return re.sub(r"\s+", " ", x.lower().strip())


@functools.lru_cache(maxsize=1)
def fetch_tissue_vocabulary(g) -> dict[str, str]:
    """
    Fetch UniProt tissue vocabulary.

    Returns:
        normalized_name -> canonical_name
    """

    resp = httpx.get(
        _TISSLIST_URL,
        timeout=_HTTP_TIMEOUT,
        follow_redirects=True
    )
    resp.raise_for_status()

    tissue_map: dict[str, str] = {}

    for line in resp.text.splitlines():

        if line.startswith("ID   "):

            tissue: str = line[5:].rstrip(".")

            if not tissue:
                continue

            tissue_map[norm(tissue)] = tissue

            # ------------------------------------------
            # GRAPH NODE
            # ------------------------------------------

            g.add_node(
                attrs=dict(
                    id=tissue,
                    type="TISSUE",
                    source="UNIPROT",
                    embedding=embed(tissue),
                )
            )

    return tissue_map


@functools.lru_cache(maxsize=1)
def fetch_uberon_layer(g) -> nx.MultiGraph:
    obo_graph = obonet.read_obo(_UBERON_URL)

    for node_id, data in obo_graph.nodes(data=True):
        name = data.get("name")

        if not name:
            continue

        g.add_node(
            attrs=dict(
                id=node_id,
                name=name,
                type="UBERON_TERM",
                embedding=embed(name),
            )
        )

    for child_id, parent_id, edge_data in obo_graph.edges(data=True):
        child = obo_graph.nodes[child_id].get("name")
        parent = obo_graph.nodes[parent_id].get("name")

        if not child or not parent:
            continue

        rel = edge_data.get("typedef") or "is_a"

        g.add_edge(
            child,
            parent,
            attr=dict(
                rel=rel,
                src_layer="UBERON_TERM",
                trgt_layer="UBERON_TERM",
            )
        )

    return obo_graph


def connect_uniprot_to_uberon(g) -> None:
    """Maps UniProt tissues onto Uberon ontology."""
    tissue_map = fetch_tissue_vocabulary(g)
    obo_graph = fetch_uberon_layer(g)

    # 1. Build the Uberon lookup dictionary
    uberon_lookup: dict[str, tuple[str, str]] = {}

    for node_id, data in obo_graph.nodes(data=True):
        name = data.get("name")
        if not name:
            continue

        # Map the primary name
        uberon_lookup[norm(name)] = (node_id, name)

        # Map synonyms extracted via regex
        for syn in data.get("synonym", []):
            if match := re.search(r'"([^"]+)"', syn):
                uberon_lookup[norm(match.group(1))] = (node_id, name)

    # 2. Match tissues to Uberon terms
    for tissue_norm, tissue_name in tissue_map.items():
        # Exact match check (O(1) complexity)
        if tissue_norm in uberon_lookup:
            _, uberon_name = uberon_lookup[tissue_norm]
            g.add_edge(
                tissue_name,
                uberon_name,
                attr={"rel": "ontology_match", "src_layer": "TISSUE", "trgt_layer": "UBERON_TERM"}
            )
            continue

        # Partial substring match check (Fallback)
        for uberon_norm, (_, uberon_name) in uberon_lookup.items():
            if tissue_norm in uberon_norm or uberon_norm in tissue_norm:
                g.add_edge(
                    tissue_name,
                    uberon_name,
                    attr={"rel": "ontology_partial_match", "src_layer": "TISSUE", "trgt_layer": "UBERON_TERM"}
                )
                break





