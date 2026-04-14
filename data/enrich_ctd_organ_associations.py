"""
CTD (Comparative Toxicogenomics Database) batch queries scoped from ORGAN → DISEASE context.

Documentation: https://ctdbase.org/tools/batchQuery.go — ``inputType=disease``, ``report=chem_curated``.

Prompt (user): human-organ graph slice; curated chemical associations; descriptions + xrefs + ``embedding``.

CHAR: only human-relevant rows when CTD returns an Organism field; Homo sapiens preferred.
"""
from __future__ import annotations

import asyncio
from urllib.parse import quote

from data.description_xref_wiring import apply_embedding_and_description_xrefs
from data.graph_identity import canonical_node_id, phase_http_log, timed_ms


def _ctd_chemical_name(row: dict) -> str | None:
    """CHAR: CTD batch rows vary slightly by endpoint — try common keys."""
    for k in (
        "ChemicalName",
        "Chemical",
        "ChemicalTerm",
    ):
        v = row.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    v = row.get("CasRN")
    if v and isinstance(v, str) and v.strip():
        return v.strip()
    return None


async def enrich_ctd_organ_associations(self) -> None:
    """ORGAN → neighboring DISEASE labels → CTD curated chemicals → CTD_CURATED_CHEMICAL nodes."""
    _CTD_BATCH = "http://ctdbase.org/tools/batchQuery.go"

    organ_nodes = [
        (k, v)
        for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "ORGAN" and self._is_active(k)
    ]
    if not organ_nodes:
        print("Phase 10c-CTD: no ORGAN nodes — skip")
        return

    seen_node: set[str] = set()
    nodes_made = 0

    for organ_id, od in organ_nodes:
        organ_label = (od.get("label") or od.get("input_term") or "").strip()
        diseases: list[tuple[str, str]] = []
        for nb in self.g.G.neighbors(organ_id):
            nd = self.g.G.nodes.get(nb, {})
            if nd.get("type") != "DISEASE":
                continue
            dlab = (nd.get("label") or nd.get("obo_id") or "").strip()
            if dlab:
                diseases.append((nb, dlab))
        diseases = diseases[:6]

        for d_nid, d_term in diseases:
            t0 = timed_ms()
            url = (
                f"{_CTD_BATCH}?inputType=disease&inputTerms={quote(d_term)}"
                f"&report=chem_curated&format=json"
            )
            try:
                r = await self.client.get(url, timeout=60.0)
                phase_http_log("10c_ctd", "batch_disease", url, status_code=r.status_code, elapsed_ms=timed_ms() - t0)
                if r.status_code != 200:
                    continue
                ct = r.headers.get("content-type", "")
                if "json" not in ct:
                    continue
                raw = r.json()
                rows = raw if isinstance(raw, list) else []
            except Exception as e:
                phase_http_log("10c_ctd", "batch_err", url, err_class=type(e).__name__)
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue
                orgn = str(row.get("Organism") or "")
                if orgn and "Homo sapiens" not in orgn and "9606" not in orgn:
                    continue
                cname = _ctd_chemical_name(row)
                if not cname:
                    continue
                nid = canonical_node_id(
                    "CTDCHEM",
                    {
                        "chemical": cname.lower(),
                        "disease_query": d_term.lower(),
                        "organ_edge": organ_id,
                    },
                    digest_bytes=10,
                )
                if nid in seen_node:
                    continue
                seen_node.add(nid)

                actions = str(row.get("InteractionActions") or row.get("InferenceScore") or "")
                description = (
                    f"CTD curated chemical association for disease context '{d_term}' "
                    f"(organ slice: {organ_label}). Substance: {cname}. "
                    f"Evidence/actions: {actions}. "
                    f"MESH and PubChem cross-refs may appear in source rows."
                ).strip()[:12000]

                self.g.add_node(
                    {
                        "id": nid,
                        "type": "CTD_CURATED_CHEMICAL",
                        "label": cname[:480],
                        "description": description,
                        "ctd_disease_query": d_term,
                        "organ_context": organ_label,
                        "source": "CTD_BATCH",
                    }
                )
                await apply_embedding_and_description_xrefs(
                    self.g,
                    nid,
                    description=description,
                    ntype="CTD_CURATED_CHEMICAL",
                    label=cname[:480],
                )
                nodes_made += 1

                if nid not in self.g.G[organ_id]:
                    self.g.add_edge(
                        src=organ_id,
                        trgt=nid,
                        attrs={
                            "rel": "ORGAN_CTD_CHEMICAL_CONTEXT",
                            "src_layer": "ORGAN",
                            "trgt_layer": "CTD_CURATED_CHEMICAL",
                            "source": "CTD",
                            "disease_node": d_nid,
                        },
                    )
                if nid not in self.g.G[d_nid]:
                    self.g.add_edge(
                        src=d_nid,
                        trgt=nid,
                        attrs={
                            "rel": "DISEASE_CTD_CHEMICAL",
                            "src_layer": "DISEASE",
                            "trgt_layer": "CTD_CURATED_CHEMICAL",
                            "source": "CTD",
                        },
                    )
            await asyncio.sleep(0.4)

    print(f"Phase 10c-CTD: {nodes_made} CTD_CURATED_CHEMICAL nodes")
