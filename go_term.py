import asyncio
import logging
import urllib.parse
from typing import Any
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class RateLimitException(Exception): 
    pass

# Zentraler Semaphore zur Einhaltung der maximalen parallelen API-Anfragen
RATE_LIMITER = asyncio.Semaphore(10)

class GoApiFetcher:
    BASE_URL = "https://api.geneontology.cloud"

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        retry=retry_if_exception_type((RateLimitException, aiohttp.ClientError)),
        reraise=True
    )
    async def _execute_get(self, url: str) -> Any:
        """
        Zentrale Hilfsmethode, die alle GET-Anfragen bündelt.
        Sichert die Einhaltung des Rate-Limits und fängt HTTP-Fehler 
        wie 429 oder 503 für die Wiederholungs-Logik (Tenacity) ab.
        """
        async with RATE_LIMITER:
            try:
                logger.debug(f"Requesting URL: {url}")
                async with self.session.get(url, headers={"Accept": "application/json"}) as response:
                    if response.status == 429:
                        logger.warning("Rate limit hit (429). Backing off...")
                        raise RateLimitException("Rate limit exceeded")
                    if response.status in (503, 502, 504):
                        logger.warning(f"Server temporarily unavailable ({response.status}). Retrying...")
                        raise RateLimitException("Server overloaded/unavailable")

                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"Network error during request to {url}: {e}")
                raise

    async def get_term_hierarchy(self, goid: str) -> Any:
        """Holt die hierarchischen Beziehungen (Parents/Children) eines GO-Terms."""
        clean_id = goid.strip().replace(":", "_").split("/")[-1]
        url = f"{self.BASE_URL}/go/{clean_id}/hierarchy"
        return await self._execute_get(url)

    async def get_cam_pathway(self, go_term: str) -> Any:
        """Holt alle verknüpften GO-CAM Modell-IDs für einen spezifischen GO-Term."""
        clean_id = go_term.strip().split("/")[-1]
        url = f"{self.BASE_URL}/go/{clean_id}/models"
        return await self._execute_get(url)

    async def get_cam_detailed(self, go_cam: str) -> Any:
        """
        Holt die detaillierten Informationen (goids, goclasses) eines GO-CAM Modells.
        Nutzt URL-Encoding, um Probleme mit Slashes in Modell-URIs zu vermeiden.
        """
        clean_cam = []
        if isinstance(go_cam, str):
            clean_cam = [go_cam]
        elif isinstance(go_cam, list):
            for c in go_cam:
                clean_cam = c.strip()
        joined_cams = ",".join(clean_cam)
        encoded_cam = urllib.parse.quote(joined_cams, safe="")
        url = f"{self.BASE_URL}/models/go?gocams={encoded_cam}"
        return await self._execute_get(url)

    async def get_term_details(self, term: str) -> Any:
        """Holt die Kern-Metadaten (Label, Definition, Synonyms) eines GO-Terms."""
        clean_id = term.strip().split("/")[-1]
        url = f"{self.BASE_URL}/go/{clean_id}"
        return await self._execute_get(url)


async def cam_for_term(fetcher, go_terms, g):
    """get cam for specific term"""
    missing_terms = set()

    #
    cam_pw_tasks = [fetcher.get_cam_pathway(item) for item in go_terms]
    cam_pw_results = await asyncio.gather(*cam_pw_tasks)

    #
    for k, (cam_batch, term) in enumerate(zip(cam_pw_results, go_terms)):
        cam_detailed_results = await fetcher.get_cam_detailed([_item["gocam"].split("/")[-1] for _item in cam_batch])

        for k, cam_details in enumerate(cam_detailed_results):
            print("cam_details", len(cam_details))

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

    print("Cams for term extracted")
    await term_details(fetcher, missing_terms, g)
    print("com for terms finished...")
    return



async def term_details(fetcher, go_terms, g):
    #
    term_tasks = [fetcher.get_term_details(item) for item in go_terms]
    term_results = await asyncio.gather(*term_tasks)

    for item in term_results:
        print("term result item", item)

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
                    text=f"{item['gonames']} {item['definitions']}",
                    embed_key="text",
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

    hierarchy_tasks = [fetcher.get_term_hierarchy(item) for item in go_terms]
    hierarchy_results = await asyncio.gather(*hierarchy_tasks)

    for hierarchy_stack, center_id in zip(hierarchy_results, go_terms):
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
    await term_details(fetcher, missing_nodes, g)
    print("heirarchy_process finisehd...")


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
    print("process go terms...")

    go_terms = get_terms(g)

    async with aiohttp.ClientSession() as session:
        fetcher = GoApiFetcher(session)
        await hierarchy_process(fetcher, go_terms, g)

        """
        go_terms = get_terms(g)

        await cam_for_term(fetcher, go_terms, g)

        await terms_for_fetched_cams(fetcher, go_terms, g)
        """


    print("finished go_term_graph via pure SPARQL execution.")
    return g