import asyncio
import logging
from typing import Dict, Any
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class RateLimitException(Exception): pass


RATE_LIMITER = asyncio.Semaphore(10)


class GoApiFetcher:
    # Offizieller Standard-Endpunkt der GO-SPARQL API v2.2

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        retry=retry_if_exception_type((RateLimitException, aiohttp.ClientError)),
        reraise=True
    )

    async def get_hierarchy(self, goid):
        url = f"https://api.geneontology.cloud/go/{goid.replace(':','_')}/hierarchy"
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def get_cam_pathway(self, go_term):
        url = f"https://api.geneontology.cloud/go/{go_term}/models"
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def get_cam_detailed(self, go_cam):
        url = f"https://api.geneontology.cloud/models/go?gocams={go_cam}"
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def get_term_details(self, term):
        url = f"https://api.geneontology.cloud/go/{term}"
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()


async def cam_for_term(fetcher, go_terms, g):
    """get cam for specific term"""
    missing_terms = set()

    #
    cam_pw_tasks = [fetcher.get_cam_pathway(item) for item in go_terms]
    cam_pw_results = await asyncio.gather(*cam_pw_tasks)

    #
    cam_pw_tasks = [fetcher.get_cam_detailed(_item) for items in cam_pw_results for _item in items]
    cam_detailed_results = await asyncio.gather(*cam_pw_tasks)

    start = 0
    for j, (cam_batch, term) in enumerate(zip(cam_pw_results, go_terms)):
        end = start + len(cam_batch)

        for k, cam_details in enumerate(cam_detailed_results[start:end]):

            # GOCAM -> G
            g.add_node(
                dict(
                    id=cam_details["gocam"].split("/")[-1],
                    type="GOCAM",
                    text=f"{cam_details['gonames']} {cam_details['definitions']}",
                    embed_key="text",
                )
            )

            # CENTER TERM -> CAM
            g.add_edge(
                src=term,
                trgt=cam_details["gocam"],
                attrs=dict(
                    rel="gocam",
                    src_layer="GO_TERM",
                    trgt_layer="GOCAM",
                )
            )
            cams_terms = [
                *[_i.split("/")[-1] for _i in cam_details["goclasses"]],
                *[_i.split("/")[-1] for _i in cam_details["goids"]]
            ]

            # CAMS TERMS -> G-> CAM
            for cams_term in cams_terms:
                if not g.get_node(cams_term):
                    missing_terms.add(cams_term)

                g.add_node(
                    dict(
                        id=cams_term.split("/")[-1],
                        type="GO_TERM",
                    )
                )
                g.add_edge(
                    src=cam_details["gocam"],
                    trgt=term,
                    attrs=dict(
                        rel="gocam",
                        src_layer="GOCAM",
                        trgt_layer="GO_TERM",
                    )
                )
        start += len(cam_batch)
    print("Cams for term extracted")

    await term_details(fetcher, missing_terms, g)
    print("com for terms finished...")
    return




async def term_details(fetcher, go_terms, g):
    #
    term_tasks = [fetcher.get_cam_pathway(item) for item in go_terms]
    term_results = await asyncio.gather(*term_tasks)

    for item in term_results:
        g.add_node(
            dict(
                id=item["goid"].split("/")[-1],
                type="GO_TERM",
                text=f"{item['definition']}, {item['label']} {item['synonyms']}",
                embed_key="text",
            )
        )



async def terms_for_fetched_cams(fetcher, go_terms, g):
    """get cam for specific term"""
    cam_pw_tasks = [fetcher.get_cam_pathway(item) for item in go_terms]
    cam_pw_results = await asyncio.gather(*cam_pw_tasks)
    #
    for cam_batch, term in zip(cam_pw_results, go_terms):
        for item in cam_batch:
            g.add_node(
                dict(
                    id=item["gocam"].split("/")[-1],
                    type="GOCAM",
                    embed_key="text"
                )
            )
            # TERM -> CAM
            g.add_edge(
                src=term,
                trgt=item["gocam"],
                attrs=dict(
                    rel="gocam",
                    src_layer="GO_TERM",
                    trgt_layer="GOCAM",
                )
            )

    print("Cams for term extracted")





async def hierarchy_process(fetcher, go_terms, g):
    "get goterm hierarch for specific term"
    missing_nodes = set()
    # 1. Core-Metadaten parallel holen
    hierarchy_tasks = [fetcher.get_hierarchy(item) for item in go_terms]
    hierarchy_results = await asyncio.gather(*hierarchy_tasks)

    for hierarchy_stack, center_id in zip(hierarchy_results, go_terms):
        """
        {
        "GO": "http://purl.obolibrary.org/obo/GO_0032991",
        "label": "protein-containing complex",
        "hierarchy": "parent"
        },
        """
        for item in hierarchy_stack:
            if not g.get_node(item["GO"]):  # keep _
                g.add_node(
                    dict(
                        id=item["GO"],
                        type="GO_TERM",
                        embed_key="text",
                    )
                )
                g.add_edge(
                    src=center_id,
                    trgt=item["GO"],
                    attrs=dict(
                        rel=item["hierarchy"],
                        src_layer="GO_TERM",
                        trgt_layer="GO_TERM",
                    )
                )
            missing_nodes.add(item["GO"])

    # collect details
    await term_details(fetcher, go_terms, g)
    print("Err heirarchy_process ended")


def get_terms(g):
    go_terms = []
    for k, v in g.G.nodes(data=True):
        if v.get("type") == "GO_TERM" and "id" in v:
            go_terms.append(k)
    return go_terms

async def go_term_graph(g: Any) -> Any:
    """
    Get existing terms ancestors
    for each term get cams
    for each cam get terms
    todo maybe repeat terms -> cam -> terms
    """
    print("process go terms via GO-SPARQL API workflow...")

    go_terms = get_terms(g)

    async with aiohttp.ClientSession() as session:
        fetcher = GoApiFetcher(session)
        await hierarchy_process(fetcher, go_terms, g)

        go_terms = get_terms(g)

        await cam_for_term(fetcher, go_terms, g)

        await terms_for_fetched_cams(fetcher, go_terms, g)

    print("finished go_term_graph via pure SPARQL execution.")
    return g