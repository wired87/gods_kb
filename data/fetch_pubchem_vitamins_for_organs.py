"""
Organ-scoped vitamin-class compounds via PubChem PUG REST.

Prompt (user): extend data dir with vitamins using PubChem API; one fetch entrypoint per file.

CHAR: combines ``classification/hnid/.../cids`` (optional HNIDs from Classification Browser) with
optional ``fastsimilarity_2d`` seeds per https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest .
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


async def fetch_pubchem_vitamins_for_organs(self) -> None:
    """
    Discover vitamin-related PubChem CIDs (classification nodes + 2-D similarity seeds),
    then wire ORGAN → VITAMIN (typed) for active organs.

    Env (all optional unless noted):
        ACID_PUBCHEM_CLASS_HNID_VITAMINS — comma-separated classification HNIDs
        ACID_PUBCHEM_VITAMIN_SIMILARITY_SEEDS — optional seed CIDs (merged with HNID hits). If no HNIDs
            and this is unset, built-in reference CIDs are used.
        ACID_PUBCHEM_VITAMIN_SIM_THRESHOLD — Tanimoto threshold (default 75)
        ACID_PUBCHEM_VITAMIN_SIM_MAX — MaxRecords per seed (default 12)
        ACID_PUBCHEM_VITAMIN_MAX_CIDS — cap merged CID list before wiring (default 160)
        ACID_PUBCHEM_VITAMIN_MAX_PER_ORGAN — cap per organ (default 8)
    """
    hn_env = (os.environ.get("ACID_PUBCHEM_CLASS_HNID_VITAMINS") or "").strip()
    seed_raw = (os.environ.get("ACID_PUBCHEM_VITAMIN_SIMILARITY_SEEDS") or "").strip()
    if not hn_env and not seed_raw:
        # CHAR: default seeds only when no HNID list — avoids diluting curated classification fetches.
        seed_raw = "54670067,1130,5280795,14985"

    merged: list[str] = []
    seen: set[str] = set()
    cap = max(8, env_int("ACID_PUBCHEM_VITAMIN_MAX_CIDS", 160))

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
    thr = max(60, min(100, env_int("ACID_PUBCHEM_VITAMIN_SIM_THRESHOLD", 75)))
    per_seed = max(4, env_int("ACID_PUBCHEM_VITAMIN_SIM_MAX", 12))
    for sd in seeds:
        _push(await pug_fastsimilarity_2d_cids(self.client, sd, threshold=thr, max_records=per_seed))
        if len(merged) >= cap:
            break
        await asyncio.sleep(0.35)

    if not merged:
        print("Phase 10d-Vitamins: no CIDs resolved — skip")
        return

    max_po = max(1, env_int("ACID_PUBCHEM_VITAMIN_MAX_PER_ORGAN", 8))
    n = await wire_organs_to_typed_pubchem(
        self,
        cid_list=merged,
        node_type="VITAMIN",
        edge_rel="ORGAN_ASSOCIATED_VITAMIN",
        phase_tag="10d_vitamin",
        id_prefix="VITAMIN_CID",
        max_per_organ=max_po,
        description_class="Vitamin-class compound",
    )
    print(f"Phase 10d-Vitamins: {n} new VITAMIN nodes (PubChem PUG)")
