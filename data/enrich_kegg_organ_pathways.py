"""
KEGG (Kyoto Encyclopedia of Genes and Genomes) REST API — disease → pathway linking for organ slices.

Documentation: https://www.kegg.jp/kegg/rest/keggapi.html — ``find``, ``link``, ``list``.

Prompt (user): human (``hsa``) pathway ids; ORGAN context via disease keyword discovery; descriptions + xrefs + ``embedding``.

CHAR: text endpoints; no API key; polite pacing with ``asyncio.sleep``.
"""
from __future__ import annotations

import asyncio
from urllib.parse import quote

from data.description_xref_wiring import apply_embedding_and_description_xrefs
from data.graph_identity import phase_http_log, timed_ms

_KEGG = "https://rest.kegg.jp"


async def enrich_kegg_organ_pathways(self) -> None:
    """Match ORGAN labels to KEGG disease entries; attach ``hsa`` pathway nodes linked to those diseases."""
    organ_nodes = [
        (k, v)
        for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "ORGAN" and self._is_active(k)
    ]
    if not organ_nodes:
        print("Phase 10c-KEGG: no ORGAN nodes — skip")
        return

    pathways_added = 0
    edges_added = 0

    async def _kegg_get(path: str) -> str:
        t0 = timed_ms()
        url = f"{_KEGG}{path}"
        try:
            r = await self.client.get(url, timeout=40.0)
            phase_http_log("10c_kegg", "get", url, status_code=r.status_code, elapsed_ms=timed_ms() - t0)
            if r.status_code != 200:
                return ""
            return r.text or ""
        except Exception as e:
            phase_http_log("10c_kegg", "get_err", url, err_class=type(e).__name__)
            return ""

    for organ_id, od in organ_nodes:
        organ_label = (od.get("label") or od.get("input_term") or "").strip()
        if not organ_label:
            continue
        q = organ_label.split()[0].lower()
        if len(q) < 3:
            continue
        find_body = await _kegg_get(f"/find/disease/{quote(q)}")
        await asyncio.sleep(0.4)
        ds_ids: list[str] = []
        for line in find_body.splitlines():
            line = line.strip()
            if not line or "\t" not in line:
                continue
            left = line.split("\t", 1)[0].strip()
            if left.startswith("ds:"):
                ds_ids.append(left)
        ds_ids = ds_ids[:4]

        for ds in ds_ids:
            link_body = await _kegg_get(f"/link/pathway/{ds}")
            await asyncio.sleep(0.4)
            hsa_ids: list[str] = []
            for line in link_body.splitlines():
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[1].startswith("path:hsa"):
                    hsa_ids.append(parts[1].replace("path:", ""))
            hsa_ids = list(dict.fromkeys(hsa_ids))[:8]

            for hsa in hsa_ids:
                list_body = await _kegg_get(f"/list/pathway/{hsa}")
                await asyncio.sleep(0.35)
                pname = hsa
                for line in list_body.splitlines():
                    if "\t" in line:
                        pname = line.split("\t", 1)[1].strip()
                        break

                nid = f"KEGG_PATHWAY_{hsa}"
                description = (
                    f"KEGG pathway {hsa}: {pname}. Linked from KEGG disease {ds} "
                    f"via organ keyword '{organ_label}'. KEGG:{hsa} path:{hsa}."
                ).strip()[:12000]

                if not self.g.G.has_node(nid):
                    self.g.add_node(
                        {
                            "id": nid,
                            "type": "KEGG_PATHWAY",
                            "label": pname[:512],
                            "description": description,
                            "kegg_pathway_id": hsa,
                            "kegg_disease_id": ds,
                            "source": "KEGG_REST",
                        }
                    )
                    await apply_embedding_and_description_xrefs(
                        self.g,
                        nid,
                        description=description,
                        ntype="KEGG_PATHWAY",
                        label=pname[:512],
                    )
                    pathways_added += 1

                if nid not in self.g.G[organ_id]:
                    self.g.add_edge(
                        src=organ_id,
                        trgt=nid,
                        attrs={
                            "rel": "ORGAN_KEGG_PATHWAY_BRIDGE",
                            "src_layer": "ORGAN",
                            "trgt_layer": "KEGG_PATHWAY",
                            "source": "KEGG",
                            "kegg_disease": ds,
                        },
                    )
                    edges_added += 1

    print(f"Phase 10c-KEGG: {pathways_added} KEGG_PATHWAY nodes, {edges_added} ORGAN-pathway edges")
