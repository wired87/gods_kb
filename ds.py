"""
ds — project-wide static datasets and shared physical-filter vocabulary.

Prompt: avoid circular imports by storing hardcoded variables (of the entire project)
within this module at the project root.

Prompt: analyze ``filter_physical_compound`` (``str`` or token list), classify tokens
against ``PHYSICAL_CATEGORY_ALIASES`` keys, and expose canonical slot names for
``query_pipe`` and ``UniprotKB`` gating.

CHAR: keep this module free of imports from ``data.*`` or ``query_pipe`` so ``ds`` stays
the acyclic leaf every consumer can import safely.
"""
from __future__ import annotations

import re

import httpx

from _db.manager import DBManager
from firegraph.graph import GUtils
import dotenv
dotenv.load_dotenv()

def cleanup_key_entries(value: str | list[str] | None) -> list[str]:
    """
    Normalize ``filter_physical_compound`` input into raw string tokens (not yet alias-resolved).

    Accepts a comma/semicolon/pipe-separated string or a list of fragments; list elements
    may themselves contain separators.
    """
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        parts = re.split(r"[,;|]+", cleaned)
        return [p.strip() for p in parts if p.strip()]
    out: list[str] = []
    for x in value:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        if re.search(r"[,;|]", s):
            out.extend(cleanup_key_entries(s))
        else:
            out.append(s)
    return out






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



import httpx

async def get_protein_sequence(protein: str, client: httpx.AsyncClient):
    """
    Fetch protein sequence from UniProt by gene/protein name.
    Returns: str | None
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"gene:{protein} AND organism_id:9606",
        "format": "json",
        "fields": "sequence",
        "size": 1
    }

    try:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        if not results:
            return None
        return protein, results[0]["sequence"]["value"]

    except Exception as e:
        print(f"Error fetching UniProt sequence for {protein}: {e}")
        return None

import asyncio

import os

import httpx
import requests

from ds import get_string_graph, get_protein_sequence
from embedder import embed, similarity
import re

db = DBManager()




def parse_uniprot_flatfile(data_string):
    print("[*] Starte Parsing der UniProt Daten...")
    entries = []
    # Splitte den String in einzelne Blöcke basierend auf dem Trenner //
    raw_entries = data_string.strip().split('//')

    for i, raw_entry in enumerate(raw_entries):
        print("work pentry", i)
        if not raw_entry.strip():
            print("entry is none -> contine")
            continue

        entry_dict = {
            "id": None,
            "accession": None,
            "description": "",
            "organism": "",
            "sequence": "",
            "length": None
        }

        lines = raw_entry.strip().split('\n')
        capture_sequence = False

        for line in lines:
            line = line.strip()

            # ID: Identifier und Länge
            if line.startswith('ID'):
                parts = line.split()
                entry_dict["id"] = parts[1]
                entry_dict["length"] = parts[3]

            # AC: Accession Number
            elif line.startswith('AC'):
                entry_dict["accession"] = line.replace('AC', '').replace(';', '').strip()

            # DE: Description (Name des Proteins)
            elif line.startswith('DE'):
                entry_dict["description"] += line.replace('DE', '').strip() + " "

            # OS: Organism Source
            elif line.startswith('OS'):
                entry_dict["organism"] += line.replace('OS', '').strip() + " "

            # SQ: Start der Sequenz-Sektion
            elif line.startswith('SQ'):
                capture_sequence = True
                continue

            # Sequenz-Daten auslesen (Zeilen nach SQ, die eingerückt sind)
            elif capture_sequence:
                # Entferne Leerzeichen und Zahlen aus der Sequenzzeile
                clean_seq = re.sub(r'[\d\s]', '', line)
                entry_dict["sequence"] += clean_seq

        # Cleanup der Texte
        entry_dict["description"] = entry_dict["description"].strip()
        entry_dict["organism"] = entry_dict["organism"].strip()
        entries.append(entry_dict)
        print(f"[OK] Eintrag geladen: {entry_dict['id']}")

    print(f"[FINISHED] Insgesamt {len(entries)} Einträge konvertiert.")
    return entries

def get_data():
    if os.getenv("LOCAL_DATA", None) is not None:

        def embed_protein_fun(results):
            for r in results:
                protein_function = get_p_fun(r)
                embedding = embed(protein_function)
                r["embedding"] = embedding
            return results

        url = "https://rest.uniprot.org/uniprotkb/stream"
        params = {
            "query": "organism_id:9606",  # syn:32630
            "format": "json",
        }
        r = requests.get(url, params=params, stream=True, timeout=990)
        r.raise_for_status()
        data = r.json()
        results = embed_protein_fun(data["results"])
    else:
        print("Fetch sprot data lcoal...")
        results = parse_uniprot_flatfile(open("_db/uniprot_sprot.dat", "r").read())
        for entry in results:
            entry["embedding"] = embed(entry["description"])
    return results

def get_human_entries() -> dict[str, list[float]]:
    # check duck db status
    if db.row_count("PROTEIN") == 0:
        print("read human entries...")
        results = get_data()
        print("results extracted...")
        # save
        db.insert(table="PROTEIN", rows=results)
        print("db insert")
    return db.get_rows(table_name="PROTEIN", select="primaryAccession, embedding")



def get_p_fun(entry) -> str:
    functions = []
    for c in entry.get("comments", []):
        if c.get("commentType") == "FUNCTION":
            for t in c.get("texts", []):
                if t.get("value"):
                    functions.append(t["value"][:500])
    return " | ".join(functions)


def load_string_graph(g:GUtils, data, min_score=0.7):
    """
    data: list[dict] (STRING API output)
    min_score: filter weak edges
    """


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

async def work_human_main(g, functions:list[float], outsrc:list[float]):
    print("work_human_main...")
    results = get_human_entries()
    print("work_human_main human results generated")

    # SS identify mathci functionaltiy of proteins
    print("perform ss...")
    fun_map:list[list[str]] = []
    for fun, out in zip(functions, outsrc):
        fun_sub_map = []
        for pid, embedding in results.items():
            in_score = similarity(fun, embedding)
            out_score = similarity(out, embedding)
            # function wanted AND NOT OUSRC CRITERIA?
            if in_score >= .7 and out_score < .6:
                fun_sub_map.append(pid)
        fun_map.append(fun_sub_map)

    print("working STRING...")
    # build STRING GRAPH
    client = httpx.AsyncClient()
    # loop protien id maps
    for fun_protein_map in fun_map:
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

        result = await asyncio.gather(
            *[
                get_protein_sequence(p["id"], client)
                for p in g.get_nodes(filter_key="type", filter_value="PROTEIN")
            ]
        )

        for pid, seq in result:
            g.G.nodes[pid]["sequence"] = seq
    g.print_status_G()
    print("get_human_entries... done")




if __name__ == "__main__":
    asyncio.run(work_human_main(
        g=GUtils(),
        functions=[embed("pos impact amygala")],
        outsrc=[embed("heuschnupfen")],
    ))
    print("done")


