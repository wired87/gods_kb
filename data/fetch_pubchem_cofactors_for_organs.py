"""
Organ-scoped cofactor-like compounds via PubChem PUG REST.

Prompt (user): extend data dir with cofactors using PubChem API; one fetch entrypoint per file.

CHAR: optional ``classification/hnid`` lists plus ``fastsimilarity_2d`` around coenzyme seed CIDs.
"""
from __future__ import annotations

import asyncio
import os

from data._pug_rest_compound_fetch import (
    env_int,
    parse_cid_list,
    parse_hnid_list,
    pug_classification_hnid_cids,
    pug_fastsimilarity_2d_cids,
    wire_organs_to_typed_pubchem,
)


async def fetch_pubchem_cofactors_for_organs(self) -> None:
    """
    Merge cofactor-related CIDs from classification HNIDs and/or similarity expansion.

    Env:
        ACID_PUBCHEM_CLASS_HNID_COFACTORS — comma-separated HNIDs
        ACID_PUBCHEM_COFACTOR_SIMILARITY_SEEDS — comma-separated seed CIDs
        ACID_PUBCHEM_COFACTOR_SIM_THRESHOLD (default 70)
        ACID_PUBCHEM_COFACTOR_SIM_MAX — MaxRecords per seed (default 14)
        ACID_PUBCHEM_COFACTOR_MAX_CIDS — merged cap (default 180)
        ACID_PUBCHEM_COFACTOR_MAX_PER_ORGAN (default 8)

    CHAR: optional HNIDs; similarity seeds default to common coenzyme-anchor CIDs unless overridden.
    """
    hn_env = (os.environ.get("ACID_PUBCHEM_CLASS_HNID_COFACTORS") or "").strip()
    seed_raw = (os.environ.get("ACID_PUBCHEM_COFACTOR_SIMILARITY_SEEDS") or "").strip()
    if not hn_env and not seed_raw:
        seed_raw = "5280875,87642,643975"

    merged: list[str] = []
    seen: set[str] = set()
    cap = max(8, env_int("ACID_PUBCHEM_COFACTOR_MAX_CIDS", 180))

    def _push(cids: list[str]) -> None:
        for c in cids:
            s = str(c).strip()
            if not s.isdigit() or s in seen:
                continue
            seen.add(s)
            merged.append(s)
            if len(merged) >= cap:
                return

    for hn in parse_hnid_list(hn_env):
        _push(await pug_classification_hnid_cids(self.client, hn))
        if len(merged) >= cap:
            break
        await asyncio.sleep(0.35)

    seeds = parse_cid_list(seed_raw)
    thr = max(60, min(100, env_int("ACID_PUBCHEM_COFACTOR_SIM_THRESHOLD", 70)))
    per_seed = max(4, env_int("ACID_PUBCHEM_COFACTOR_SIM_MAX", 14))
    for sd in seeds:
        _push(await pug_fastsimilarity_2d_cids(self.client, sd, threshold=thr, max_records=per_seed))
        if len(merged) >= cap:
            break
        await asyncio.sleep(0.35)

    if not merged:
        print("Phase 10d-Cofactors: no CIDs resolved — skip")
        return

    max_po = max(1, env_int("ACID_PUBCHEM_COFACTOR_MAX_PER_ORGAN", 8))
    n = await wire_organs_to_typed_pubchem(
        self,
        cid_list=merged,
        node_type="COFACTOR",
        edge_rel="ORGAN_ASSOCIATED_COFACTOR",
        phase_tag="10d_cofactor",
        id_prefix="COFACTOR_CID",
        max_per_organ=max_po,
        description_class="Cofactor / coenzyme-class compound",
    )
    print(f"Phase 10d-Cofactors: {n} new COFACTOR nodes (PubChem PUG)")
