"""
Organ-scoped fatty-acid-like compounds via PubChem PUG REST fast substructure search.

Prompt (user): extend data dir with fatty acids using PubChem API; one fetch entrypoint per file.

CHAR: ``compound/fastsubstructure/smiles/.../cids`` with MaxRecords — see PUG REST structure search docs.
"""
from __future__ import annotations

import os

from data._pug_rest_compound_fetch import env_int, pug_fastsubstructure_smiles_cids, wire_organs_to_typed_pubchem


async def fetch_pubchem_fatty_acids_for_organs(self) -> None:
    """
    Use a carboxylic alkyl substructure SMILES to retrieve representative fatty-acid CIDs, then
    wire ORGAN → FATTY_ACID.

    Env:
        ACID_PUBCHEM_FATTY_ACID_SUBSTRUCTURE_SMILES — SMILES query (required non-empty)
        ACID_PUBCHEM_FATTY_ACID_MAX_RECORDS — MaxRecords (default 24)
        ACID_PUBCHEM_FATTY_ACID_MAX_PER_ORGAN — per-organ cap (default 8)
    """
    # CHAR: default SMILES is a short alkanoic acid — PUG fastsubstructure anchor for fatty-acid-like hits.
    smiles = (os.environ.get("ACID_PUBCHEM_FATTY_ACID_SUBSTRUCTURE_SMILES") or "CCCCCCCC(=O)O").strip()

    max_rec = max(4, env_int("ACID_PUBCHEM_FATTY_ACID_MAX_RECORDS", 24))
    cids = await pug_fastsubstructure_smiles_cids(self.client, smiles, max_records=max_rec)
    if not cids:
        print("Phase 10d-FattyAcids: substructure search returned no CIDs — skip")
        return

    max_po = max(1, env_int("ACID_PUBCHEM_FATTY_ACID_MAX_PER_ORGAN", 8))
    n = await wire_organs_to_typed_pubchem(
        self,
        cid_list=cids,
        node_type="FATTY_ACID",
        edge_rel="ORGAN_ASSOCIATED_FATTY_ACID",
        phase_tag="10d_fatty_acid",
        id_prefix="FATTY_ACID_CID",
        max_per_organ=max_po,
        description_class="Fatty-acid-class compound (substructure hit)",
    )
    print(f"Phase 10d-FattyAcids: {n} new FATTY_ACID nodes (PubChem PUG)")
