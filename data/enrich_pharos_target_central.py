"""
Pharos / Target Central Resource Data (TCRD) via public GraphQL API.

Documentation: https://pharos.nih.gov/api (GraphQL playground; example ``targets(filter:{associatedDisease:...})``).

Prompt (user): organ-scoped human graph — use diseases already linked to ORGAN nodes to pull associated targets;
link existing GENE / PROTEIN nodes; attach PHAROS_TARGET_SUMMARY stubs when missing.

CHAR: POST JSON to ``https://pharos.nih.gov/graphql``; resilient to HTTP/schema drift (skip on errors).
"""
from __future__ import annotations

import asyncio

from data.description_xref_wiring import apply_embedding_and_description_xrefs
from data.graph_identity import phase_http_log, timed_ms

_PHAROS_GQL = "https://pharos.nih.gov/graphql"

_ASSOCIATED_TARGETS_Q = """
query OrganSliceTargets($dis: String!) {
  targets(filter: { associatedDisease: $dis }) {
    targets(top: 15) {
      sym
      name
      uniprot tdl
      fam
    }
  }
}
"""


async def enrich_pharos_target_central(self) -> None:
    """For each ORGAN-linked disease label, query Pharos targets; wire ORGAN → GENE/PROTEIN/PHAROS_TARGET_SUMMARY."""
    organ_nodes = [
        (k, v)
        for k, v in self.g.G.nodes(data=True)
        if v.get("type") == "ORGAN" and self._is_active(k)
    ]
    if not organ_nodes:
        print("Phase 10c-Pharos: no ORGAN nodes — skip")
        return

    summaries = 0
    edges = 0

    for organ_id, od in organ_nodes:
        organ_label = (od.get("label") or od.get("input_term") or "").strip()
        disease_terms: list[str] = []
        for nb in self.g.G.neighbors(organ_id):
            nd = self.g.G.nodes.get(nb, {})
            if nd.get("type") == "DISEASE":
                dl = (nd.get("label") or "").strip()
                if dl and dl not in disease_terms:
                    disease_terms.append(dl)
        disease_terms = disease_terms[:5]

        for d_term in disease_terms:
            t0 = timed_ms()
            try:
                r = await self.client.post(
                    _PHAROS_GQL,
                    json={"query": _ASSOCIATED_TARGETS_Q, "variables": {"dis": d_term}},
                    timeout=45.0,
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                )
                phase_http_log(
                    "10c_pharos",
                    "graphql",
                    _PHAROS_GQL,
                    status_code=r.status_code,
                    elapsed_ms=timed_ms() - t0,
                )
                if r.status_code != 200:
                    continue
                payload = r.json()
                if payload.get("errors"):
                    continue
                block = (((payload.get("data") or {}).get("targets") or {}).get("targets")) or []
            except Exception as e:
                phase_http_log("10c_pharos", "gql_err", _PHAROS_GQL, err_class=type(e).__name__)
                continue

            for t in block:
                if not isinstance(t, dict):
                    continue
                sym = (t.get("sym") or "").strip()
                if not sym:
                    continue
                gid = f"GENE_{sym}"
                pname = (t.get("name") or "").strip()
                up = (t.get("uniprot") or "").strip()
                tdl = (t.get("tdl") or "").strip()
                fam = (t.get("fam") or "").strip()
                desc = (
                    f"Pharos target {sym}: {pname}. "
                    f"Target Development Level {tdl}; family {fam}. "
                    f"UniProtKB:{up}. Disease context '{d_term}' (organ {organ_label})."
                ).strip()[:12000]

                if self.g.G.has_node(gid):
                    trgt = gid
                    self.g.update_node({"id": gid, "pharos_tdl": tdl, "pharos_family": fam})
                elif up and self.g.G.has_node(up):
                    trgt = up
                else:
                    trgt = f"PHAROS_{sym}"
                    if not self.g.G.has_node(trgt):
                        self.g.add_node(
                            {
                                "id": trgt,
                                "type": "PHAROS_TARGET_SUMMARY",
                                "label": sym,
                                "description": desc,
                                "gene_symbol": sym,
                                "uniprot_ac": up,
                                "pharos_tdl": tdl,
                                "pharos_family": fam,
                                "source": "PHAROS_GRAPHQL",
                            }
                        )
                        await apply_embedding_and_description_xrefs(
                            self.g,
                            trgt,
                            description=desc,
                            ntype="PHAROS_TARGET_SUMMARY",
                            label=sym,
                        )
                        summaries += 1

                if trgt not in self.g.G[organ_id]:
                    self.g.add_edge(
                        src=organ_id,
                        trgt=trgt,
                        attrs={
                            "rel": "PHAROS_CONTEXT_TARGET",
                            "src_layer": "ORGAN",
                            "trgt_layer": self.g.G.nodes[trgt].get("type", "GENE"),
                            "source": "PHAROS",
                            "disease_context": d_term,
                        },
                    )
                    edges += 1
            await asyncio.sleep(0.5)

    print(f"Phase 10c-Pharos: {summaries} PHAROS_TARGET_SUMMARY stubs, {edges} ORGAN-target edges")
