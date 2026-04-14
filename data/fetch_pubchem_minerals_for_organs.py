"""
Organ-scoped elemental minerals via PubChem PUG REST periodic table + name resolution.

Prompt (user): extend data dir with minerals using PubChem API; one fetch entrypoint per file.

CHAR: ``periodictable/JSON`` then ``compound/name/{ElementName}/cids`` for each selected row.
"""
from __future__ import annotations

import asyncio
import os

from data._pug_rest_compound_fetch import (
    env_int,
    pug_compound_name_first_cid,
    pug_periodic_table_rows,
    wire_organs_to_typed_pubchem,
)


async def fetch_pubchem_minerals_for_organs(self) -> None:
    """
    Resolve native-element PubChem compounds for a slice of the periodic table and wire ORGAN → MINERAL.

    Env:
        ACID_PUBCHEM_MINERAL_MIN_Z — minimum atomic number (default 11)
        ACID_PUBCHEM_MINERAL_MAX_Z — maximum atomic number (default 92)
        ACID_PUBCHEM_MINERAL_MAX_ELEMENTS — max successful name→CID resolutions (default 36)
        ACID_PUBCHEM_MINERAL_MAX_PER_ORGAN — per-organ edge cap (default 8)
    """
    rows = await pug_periodic_table_rows(self.client)
    if not rows:
        print("Phase 10d-Minerals: periodic table fetch failed — skip")
        return

    z_lo = max(1, env_int("ACID_PUBCHEM_MINERAL_MIN_Z", 11))
    z_hi = max(z_lo, env_int("ACID_PUBCHEM_MINERAL_MAX_Z", 92))
    max_el = max(1, env_int("ACID_PUBCHEM_MINERAL_MAX_ELEMENTS", 36))

    cids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if len(row) < 3:
            continue
        try:
            z = int(row[0])
        except ValueError:
            continue
        if z < z_lo or z > z_hi:
            continue
        name = (row[2] or "").strip()
        if not name:
            continue
        cid = await pug_compound_name_first_cid(self.client, name)
        await asyncio.sleep(0.2)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        cids.append(cid)
        if len(cids) >= max_el:
            break

    if not cids:
        print("Phase 10d-Minerals: no element CIDs resolved — skip")
        return

    max_po = max(1, env_int("ACID_PUBCHEM_MINERAL_MAX_PER_ORGAN", 8))
    n = await wire_organs_to_typed_pubchem(
        self,
        cid_list=cids,
        node_type="MINERAL",
        edge_rel="ORGAN_ASSOCIATED_MINERAL",
        phase_tag="10d_mineral",
        id_prefix="MINERAL_PUBCHEM_CID",
        max_per_organ=max_po,
        description_class="Native element / mineral (periodic table → PubChem CID)",
    )
    print(f"Phase 10d-Minerals: {n} new MINERAL nodes (PubChem PUG)")
