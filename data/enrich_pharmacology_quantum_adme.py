"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``main``.

CHAR: runs in-process on the same ``UniprotKB`` instance (``self``); keep signatures aligned
with the class delegator in ``uniprot_kb.py``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import google.generativeai as genai
import httpx
import networkx as nx
import numpy as np

async def enrich_pharmacology_quantum_adme(self):
    """
    Konsolidierte Inferenz-Phase: PROTEIN -> PHARMA_COMPOUND -> ATOMIC_STRUCTURE.
    1) ChEMBL mechanisms  2) ChEMBL molecule meta
    3) PubChem quantum signatures  4) Graph linking
    """
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True) if v.get("type") == "PROTEIN" and self._is_active(k)]
    _seen_mol: set[str] = set()

    for protein_id, node in protein_nodes:
        target_label = node.get("label")

        chembl_mech_url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism?target_uniprot_accession={protein_id}&format=json"

        try:
            mech_res = await self.client.get(chembl_mech_url)
            if mech_res.status_code != 200:
                continue
            mechanisms = mech_res.json().get("mechanisms", [])

            for mech in mechanisms:
                mol_chembl_id = mech.get("molecule_chembl_id")
                if not mol_chembl_id:
                    continue
                drug_node_id = f"DRUG_{mol_chembl_id}"

                # FAST PATH: already-fetched molecule — just add the edge
                if mol_chembl_id in _seen_mol:
                    if self.g.G.has_node(drug_node_id):
                        self.g.add_edge(
                            src=drug_node_id, trgt=protein_id,
                            attrs={
                                "rel": "MODULATES_TARGET",
                                "action": mech.get("action_type"),
                                "mechanism": mech.get("mechanism_of_action"),
                                "src_layer": "PHARMA", "trgt_layer": "PROTEIN",
                                "eco_code": "ECO:0000313",
                            },
                        )
                    continue
                _seen_mol.add(mol_chembl_id)

                mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_chembl_id}?format=json"
                mol_res = await self.client.get(mol_url)
                if mol_res.status_code != 200:
                    continue
                mol_data = mol_res.json()
                struct_data = mol_data.get("molecule_structures") or {}
                smiles_str = struct_data.get("canonical_smiles")
                if not smiles_str:
                    continue

                drug_name = mol_data.get("pref_name") or mol_chembl_id
                self.g.add_node({
                    "id": drug_node_id,
                    "type": "PHARMA_COMPOUND",
                    "label": drug_name,
                    "molecule_type": mol_data.get("molecule_type"),
                    "max_phase": mol_data.get("max_phase"),
                    "is_approved": mol_data.get("max_phase") == 4,
                    "chembl_id": mol_chembl_id,
                })

                self.g.add_edge(
                    src=drug_node_id, trgt=protein_id,
                    attrs={
                        "rel": "MODULATES_TARGET",
                        "action": mech.get("action_type"),
                        "mechanism": mech.get("mechanism_of_action"),
                        "src_layer": "PHARMA", "trgt_layer": "PROTEIN",
                        "eco_code": "ECO:0000313",
                    },
                )

                # PubChem quantum-chemical signatures (one-time per molecule)
                smiles_node_id = f"SMILES_{hash(smiles_str)}"
                pc_url = (
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                    f"{quote(smiles_str, safe='')}"
                    f"/property/DipoleMoment,XLogP3,Complexity,MolecularWeight,InChIKey/JSON"
                )

                try:
                    pc_res = await self.fetch_with_retry(pc_url)
                except Exception:
                    pc_res = None
                electron_props = {}
                if pc_res and "PropertyTable" in pc_res:
                    electron_props = pc_res["PropertyTable"]["Properties"][0]

                self.g.add_node({
                    "id": smiles_node_id,
                    "type": "ATOMIC_STRUCTURE",
                    "label": "Molecular_SMILES",
                    "smiles": smiles_str,
                    "inchikey": electron_props.get("InChIKey"),
                    "dipole_moment": electron_props.get("DipoleMoment"),
                    "lipophilicity_logp": electron_props.get("XLogP3"),
                    "molecular_weight": electron_props.get("MolecularWeight"),
                    "complexity": electron_props.get("Complexity"),
                })

                self.g.add_edge(
                    src=drug_node_id, trgt=smiles_node_id,
                    attrs={"rel": "HAS_STRUCTURE", "src_layer": "PHARMA",
                           "trgt_layer": "ATOMIC_STRUCTURE"},
                )
                self.g.add_edge(
                    src=smiles_node_id, trgt=protein_id,
                    attrs={"rel": "PHYSICAL_BINDING", "src_layer": "ATOMIC_STRUCTURE",
                           "trgt_layer": "PROTEIN"},
                )
                print(f"Enriched: {drug_name} -> Structure & Quantum Ingested")

        except Exception as e:
            print(f"Inference Error for Protein {target_label}: {e}")

