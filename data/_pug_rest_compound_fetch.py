"""
Low-level PubChem PUG REST helpers (classification, similarity, substructure, periodic table).

Prompt (user): extend data dir with vitamin, fatty acids, cofactors, minerals via PubChem API;
each workflow entrypoint lives in its own module — this file only holds shared HTTP helpers.

CHAR: URLs follow https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest (classification/hnid, fastsearch, periodictable).
"""
from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

_PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def _parse_int_list(raw: str | None) -> list[int]:
    out: list[int] = []
    if not raw or not str(raw).strip():
        return out
    for part in str(raw).replace(";", ",").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except ValueError:
            continue
    return out


def parse_hnid_list(raw: str | None) -> list[int]:
    """Comma-separated PubChem classification HNIDs (see Classification Browser)."""
    return _parse_int_list(raw)


def parse_cid_list(raw: str | None) -> list[str]:
    out: list[str] = []
    if not raw or not str(raw).strip():
        return out
    for part in str(raw).replace(";", ",").split(","):
        s = part.strip()
        if s.isdigit():
            out.append(s)
    return out


async def pug_classification_hnid_cids(client: Any, hnid: int) -> list[str]:
    """GET .../classification/hnid/{hnid}/cids/JSON → CID strings."""
    url = f"{_PUG}/classification/hnid/{int(hnid)}/cids/JSON"
    try:
        r = await client.get(url, timeout=60.0)
        if r.status_code != 200:
            return []
        j = r.json()
        a = (((j.get("IdentifierList") or {}).get("CID")) or [])
        return [str(x) for x in a]
    except Exception:
        return []


async def pug_fastsimilarity_2d_cids(
    client: Any,
    seed_cid: str,
    *,
    threshold: int,
    max_records: int,
) -> list[str]:
    """GET compound/fastsimilarity_2d/cid/{seed}/cids/JSON?Threshold=&MaxRecords= ."""
    url = (
        f"{_PUG}/compound/fastsimilarity_2d/cid/{quote(str(seed_cid), safe='')}/cids/JSON"
        f"?Threshold={int(threshold)}&MaxRecords={int(max_records)}"
    )
    try:
        r = await client.get(url, timeout=60.0)
        if r.status_code != 200:
            return []
        j = r.json()
        a = (((j.get("IdentifierList") or {}).get("CID")) or [])
        return [str(x) for x in a]
    except Exception:
        return []


async def pug_fastsubstructure_smiles_cids(
    client: Any,
    smiles: str,
    *,
    max_records: int,
) -> list[str]:
    """GET compound/fastsubstructure/smiles/{smiles}/cids/JSON?MaxRecords= ."""
    enc = quote((smiles or "").strip(), safe="")
    if not enc:
        return []
    url = f"{_PUG}/compound/fastsubstructure/smiles/{enc}/cids/JSON?MaxRecords={int(max_records)}"
    try:
        r = await client.get(url, timeout=60.0)
        if r.status_code != 200:
            return []
        j = r.json()
        a = (((j.get("IdentifierList") or {}).get("CID")) or [])
        return [str(x) for x in a]
    except Exception:
        return []


async def pug_periodic_table_rows(client: Any) -> list[list[str]]:
    """GET /periodictable/JSON → list of Cell rows (string lists)."""
    url = f"{_PUG}/periodictable/JSON"
    try:
        r = await client.get(url, timeout=45.0)
        if r.status_code != 200:
            return []
        j = r.json()
        rows = ((j.get("Table") or {}).get("Row")) or []
        out: list[list[str]] = []
        for row in rows:
            cells = (row.get("Cell") or []) if isinstance(row, dict) else []
            if isinstance(cells, list):
                out.append([str(x) for x in cells])
        return out
    except Exception:
        return []


async def pug_compound_name_first_cid(client: Any, name: str) -> str | None:
    """Resolve element / chemical name to first CID (name/cids/JSON)."""
    q = (name or "").strip()
    if not q or len(q) > 220:
        return None
    url = f"{_PUG}/compound/name/{quote(q, safe='')}/cids/JSON"
    try:
        r = await client.get(url, timeout=40.0)
        if r.status_code != 200:
            return None
        j = r.json()
        a = (((j.get("IdentifierList") or {}).get("CID")) or [])
        if not a:
            return None
        return str(a[0])
    except Exception:
        return None


async def pug_compound_title_description(client: Any, cid: str) -> tuple[str, str]:
    """Title + Description text for embedding (property + description endpoints)."""
    title, desc = f"CID {cid}", ""
    try:
        u_prop = f"{_PUG}/compound/cid/{quote(str(cid), safe='')}/property/Title,MolecularFormula/JSON"
        r1 = await client.get(u_prop, timeout=35.0)
        if r1.status_code == 200:
            props = ((r1.json().get("PropertyTable") or {}).get("Properties") or [{}])[0]
            title = str(props.get("Title") or title)
            mf = props.get("MolecularFormula")
            if mf:
                title = f"{title} ({mf})"
        u_desc = f"{_PUG}/compound/cid/{quote(str(cid), safe='')}/description/JSON"
        r2 = await client.get(u_desc, timeout=35.0)
        if r2.status_code == 200:
            infos = (r2.json().get("InformationList") or {}).get("Information") or []
            if infos:
                desc = str((infos[0] or {}).get("Description") or "")
    except Exception:
        pass
    return title, desc


def env_int(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


async def wire_organs_to_typed_pubchem(
    self: Any,
    *,
    cid_list: list[str],
    node_type: str,
    edge_rel: str,
    phase_tag: str,
    id_prefix: str,
    max_per_organ: int,
    description_class: str,
) -> int:
    """
    For each active ORGAN, attach up to ``max_per_organ`` CIDs as ``node_type`` nodes (PUBCHEM-backed).
    Returns count of newly created nodes.
    """
    import asyncio

    from data.description_xref_wiring import apply_embedding_and_description_xrefs
    from data.graph_identity import phase_http_log, timed_ms

    organ_nodes = [
        (k, v)
        for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "ORGAN" and self._is_active(k)
    ]
    if not organ_nodes:
        phase_http_log(
            phase_tag, "skip_no_organs", "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
            status_code=None, elapsed_ms=0.0,
        )
        return 0
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cid_list:
        s = str(c).strip()
        if s.isdigit() and s not in seen:
            seen.add(s)
            ordered.append(s)
    if not ordered:
        return 0

    created = 0
    for organ_id, od in organ_nodes:
        organ_label = (od.get("label") or od.get("input_term") or "").strip()
        if not organ_label:
            continue
        use = ordered[: max(1, int(max_per_organ))]
        for cid in use:
            nid = f"{id_prefix}_{cid}"
            t0 = timed_ms()
            title, pdesc = await pug_compound_title_description(self.client, cid)
            phase_http_log(phase_tag, "record", cid, status_code=200, elapsed_ms=timed_ms() - t0)
            description = (
                f"{description_class} from PubChem PUG REST (CID {cid}). {pdesc} "
                f"Organ context: {organ_label}. PUBCHEM.COMPOUND:{cid}."
            ).strip()
            if not self.g.G.has_node(nid):
                node = {
                    "id": nid,
                    "type": node_type,
                    "label": title[:512],
                    "description": description[:12000],
                    "pubchem_cid": cid,
                    "source": "PUBCHEM_PUG",
                }
                self.g.add_node(node)
                await apply_embedding_and_description_xrefs(
                    self.g,
                    nid,
                    description=description[:12000],
                    ntype=node_type,
                    label=title[:512],
                )
                created += 1
            if nid not in self.g.G[organ_id]:
                self.g.add_edge(
                    src=organ_id,
                    trgt=nid,
                    attrs={
                        "rel": edge_rel,
                        "src_layer": "ORGAN",
                        "trgt_layer": node_type,
                        "source": "PUBCHEM_PUG",
                        "organ_context": organ_label,
                    },
                )
            await asyncio.sleep(0.12)
    return created
