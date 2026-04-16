import aiohttp

ACCESSION_MAP = {
    # OMIM / MIM
    "MIM": {
        "match": lambda x: x.isdigit() and len(x) in [6, 7],
        "url": lambda x: f"https://omim.org/entry/{x}"
    },

    # Orphanet
    "ORPHA": {
        "match": lambda x: x.isdigit() and len(x) <= 6,
        "url": lambda x: f"https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Expert={x}"
    },

    # UniProt Disease
    "UNIPROT_DISEASE": {
        "match": lambda x: x.startswith("DI-"),
        "url": lambda x: f"https://www.uniprot.org/diseases/{x}"
    },

    # DisGeNET
    "DISGENET": {
        "match": lambda x: x.startswith("C") or x.isdigit(),
        "url": lambda x: f"https://www.disgenet.org/search/0/{x}"
    },

    # MalaCards
    "MALACARDS": {
        "match": lambda x: x.isupper() and not x.startswith("DI-"),
        "url": lambda x: f"https://www.malacards.org/card/{x.lower()}"
    },

    # OpenTargets (Ensembl gene-based)
    "OPENTARGETS": {
        "match": lambda x: x.startswith("ENSG"),
        "url": lambda x: f"https://platform.opentargets.org/target/{x}"
    }
}

async def add_disease_node(
    g,
    accession: str,
    protein_id: str,
):
    # --- resolve db + url ---
    db = None
    url = None

    for name, cfg in ACCESSION_MAP.items():
        if cfg["match"](accession):
            db = name
            url = cfg["url"](accession)
            break

    if not db:
        return None  # unknown

    node_id = f"{db}:{accession}"

    # --- fetch extra data (case-specific) ---
    data = {}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    text = await resp.text()

                    # minimal parsing (keine heavy parser → schnell)
                    data = {
                        "fetched": True,
                        "preview": text[:500]  # snippet
                    }
                else:
                    data = {"fetched": False}
        except Exception:
            data = {"fetched": False}

    # --- build node ---
    node = {
        "id": node_id,
        "type": "DISEASE",
        "accession": accession,
        "source": db,
        "url": url,
        **data
    }

    # --- graph insert ---
    g.add_node(node_id, **node)

    # --- edge ---
    g.add_edge(protein_id, node_id, attrs=dict(rel="associated_with", src_layer="PROTEIN", trgt_ayer="DISEASE"))
