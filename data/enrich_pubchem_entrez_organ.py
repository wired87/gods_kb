"""
Organ-scoped PubChem ingestion (PUG REST + optional NCBI Entrez).

Prompt (user): ToxNet / PubChem (via Entrez API) — NIH retired TOXNET; toxicology-related
annotations are continued in PubChem (PUG View) and linked NCBI resources.

Prompt (user): human-organ context, description xrefs + ``embedding`` on new nodes (``description_xref_wiring``).

CHAR: PUG REST has no API key; Entrez runs only when ``NCBI_ENTREZ_EMAIL`` is set (NCBI policy).
"""
from __future__ import annotations

import asyncio
import os
from urllib.parse import quote

from data.description_xref_wiring import apply_embedding_and_description_xrefs
from data.graph_identity import phase_http_log, timed_ms


async def enrich_pubchem_entrez_organ(self) -> None:
    """
    For each active ORGAN: discover PubChem CIDs via disease-linked queries + compound name search;
    attach PUBCHEM_COMPOUND nodes; optional Entrez esearch when email env is present.
    """
    _PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    _ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    organ_nodes = [
        (k, v)
        for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "ORGAN" and self._is_active(k)
    ]
    if not organ_nodes:
        print("Phase 10c-PubChem: no ORGAN nodes — skip")
        return

    email = (os.environ.get("NCBI_ENTREZ_EMAIL") or "").strip()
    tool = (os.environ.get("NCBI_ENTREZ_TOOL") or "acid_master").strip()
    created = 0

    async def _pug_cids_by_name(name: str) -> list[str]:
        if not name or len(name) > 200:
            return []
        t0 = timed_ms()
        url = f"{_PUG}/compound/name/{quote(name)}/cids/JSON"
        try:
            r = await self.client.get(url, timeout=45.0)
            phase_http_log("10c_pubchem", "pug_name", url, status_code=r.status_code, elapsed_ms=timed_ms() - t0)
            if r.status_code != 200:
                return []
            j = r.json()
            a = (((j.get("IdentifierList") or {}).get("CID")) or [])
            return [str(x) for x in a[:8]]
        except Exception as e:
            phase_http_log("10c_pubchem", "pug_name_err", url, err_class=type(e).__name__)
            return []

    async def _entrez_cids(term: str) -> list[str]:
        if not email or not term:
            return []
        t0 = timed_ms()
        params = {
            "db": "pccompound",
            "term": term,
            "retmax": "6",
            "retmode": "json",
            "tool": tool,
            "email": email,
        }
        try:
            r = await self.client.get(_ESEARCH, params=params, timeout=45.0)
            phase_http_log("10c_pubchem", "entrez", _ESEARCH, status_code=r.status_code, elapsed_ms=timed_ms() - t0)
            if r.status_code != 200:
                return []
            j = r.json()
            idlist = (j.get("esearchresult") or {}).get("idlist") or []
            return [str(x) for x in idlist[:6]]
        except Exception as e:
            phase_http_log("10c_pubchem", "entrez_err", _ESEARCH, err_class=type(e).__name__)
            return []

    async def _pug_record(cid: str) -> tuple[str, str]:
        """Return (title, description blob) for embedding."""
        t0 = timed_ms()
        title, desc = f"CID {cid}", ""
        try:
            u_prop = f"{_PUG}/compound/cid/{cid}/property/Title,MolecularFormula/JSON"
            r1 = await self.client.get(u_prop, timeout=35.0)
            phase_http_log("10c_pubchem", "pug_prop", u_prop, status_code=r1.status_code, elapsed_ms=timed_ms() - t0)
            if r1.status_code == 200:
                props = ((r1.json().get("PropertyTable") or {}).get("Properties") or [{}])[0]
                title = str(props.get("Title") or title)
                mf = props.get("MolecularFormula")
                if mf:
                    title = f"{title} ({mf})"
            u_desc = f"{_PUG}/compound/cid/{cid}/description/JSON"
            t1 = timed_ms()
            r2 = await self.client.get(u_desc, timeout=35.0)
            phase_http_log("10c_pubchem", "pug_desc", u_desc, status_code=r2.status_code, elapsed_ms=timed_ms() - t1)
            if r2.status_code == 200:
                infos = (r2.json().get("InformationList") or {}).get("Information") or []
                if infos:
                    desc = str((infos[0] or {}).get("Description") or "")
        except Exception as e:
            phase_http_log("10c_pubchem", "pug_record_err", cid, err_class=type(e).__name__)
        return title, desc

    for organ_id, od in organ_nodes:
        organ_label = (od.get("label") or od.get("input_term") or "").strip()
        if not organ_label:
            continue
        disease_labels: list[str] = []
        for nb in self.g.G.neighbors(organ_id):
            nd = self.g.G.nodes.get(nb, {})
            if nd.get("type") == "DISEASE":
                dl = (nd.get("label") or "").strip()
                if dl:
                    disease_labels.append(dl)
        disease_labels = disease_labels[:5]

        cid_set: list[str] = []
        for nm in [organ_label, *disease_labels]:
            for cid in await _pug_cids_by_name(nm):
                if cid not in cid_set:
                    cid_set.append(cid)
            await asyncio.sleep(0.35)
        if email:
            for nm in disease_labels or [organ_label]:
                for cid in await _entrez_cids(f"{nm} AND has_pccompound[filter]"):
                    if cid not in cid_set:
                        cid_set.append(cid)
                await asyncio.sleep(0.35)

        for cid in cid_set[:12]:
            nid = f"PUBCHEM_CID_{cid}"
            title, pdesc = await _pug_record(cid)
            description = (
                f"PubChem compound {cid}. {pdesc} "
                f"Context: organ {organ_label}. "
                f"PUBCHEM.COMPOUND:{cid}."
            ).strip()
            if not self.g.G.has_node(nid):
                self.g.add_node(
                    {
                        "id": nid,
                        "type": "PUBCHEM_COMPOUND",
                        "label": title[:512],
                        "description": description[:12000],
                        "pubchem_cid": cid,
                        "source": "PUBCHEM_PUG",
                    }
                )
                await apply_embedding_and_description_xrefs(
                    self.g,
                    nid,
                    description=description[:12000],
                    ntype="PUBCHEM_COMPOUND",
                    label=title[:512],
                )
                created += 1
            if nid not in self.g.G[organ_id]:
                self.g.add_edge(
                    src=organ_id,
                    trgt=nid,
                    attrs={
                        "rel": "ORGAN_ASSOCIATED_COMPOUND",
                        "src_layer": "ORGAN",
                        "trgt_layer": "PUBCHEM_COMPOUND",
                        "source": "PUBCHEM_ENTREZ",
                        "organ_context": organ_label,
                    },
                )
            await asyncio.sleep(0.25)

    print(f"Phase 10c-PubChem: {created} new compound nodes (PUBCHEM_COMPOUND)")
