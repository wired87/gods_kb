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

async def enrich_molecular_structures(self):
    """
    Kategorie: MOLEKÜL -> SMILES.
    Zieht den atomaren Bauplan live von PubChem (Fallback: ChEBI).
    InChIKey dient als stabiler Primärschlüssel für Node-Deduplication.
    """
    molecule_nodes = [v for k, v in self.g.G.nodes(data=True) if v.get("type") in ["MINERAL", "MOLECULE_CHAIN"] and self._is_active(k)]

    for node in molecule_nodes:
        label = node.get("label", "")

        # PRIMARY: PubChem by compound name
        props = await self._pubchem_properties_by_name(label)

        # FALLBACK: ChEBI if PubChem returned nothing
        if not props or not props.get("CanonicalSMILES"):
            props = await self._chebi_fallback(label)

        if props and props.get("CanonicalSMILES"):
            node["smiles"] = props["CanonicalSMILES"]
            node["atomic_weight"] = props.get("MolecularWeight")
            node["inchikey"] = props.get("InChIKey")
            node["smiles_source"] = "PUBCHEM" if props.get("InChIKey") else "CHEBI"
            print(f"Atomic Structure Attached: {label} (SMILES: {node['smiles']})")
        else:
            node["smiles"] = None
            node["atomic_weight"] = None
            node["smiles_source"] = "NOT_FOUND"
            print(f"SMILES not resolved for: {label}")
