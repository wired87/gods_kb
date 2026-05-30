import asyncio

import httpx

from graph import GUtils


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



    g.print_status_G()
    print("get_human_entries... done")