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

async def enrich_microbiome_axis(self):
    """
    Modelliert die Transformation von Molekülen durch das Mikrobiom via VMH.
    Pfad: ATOMIC_STRUCTURE -> MICROBIAL_STRAIN -> VMH_METABOLITE -> PROTEIN
    Erklärt indirekte Wirkstoffeffekte durch bakterielle Metabolisierung.
    """
    _VMH_BASE = "https://www.vmh.life/_api"
    mol_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                 if v.get("type") == "ATOMIC_STRUCTURE" and self._is_active(k)]

    for node_id, mol in mol_nodes:
        mol_label = mol.get("label", "")
        if not mol_label or mol_label == "Molecular_SMILES":
            # ATOMIC_STRUCTURE Nodes mit generischem Label brauchen InChIKey
            mol_label = mol.get("inchikey") or ""
        if not mol_label:
            continue

        try:
            # A: VMH Metabolit-Suche
            met_url = f"{_VMH_BASE}/metabolites/?search={quote(mol_label)}&format=json&page_size=3"
            met_res = await self.client.get(met_url, timeout=15.0)
            if met_res.status_code != 200:
                continue
            met_data = met_res.json()
            results = met_data.get("results", [])
            if not results:
                continue

            vmh_met = results[0]
            vmh_abbr = vmh_met.get("abbreviation")
            if not vmh_abbr:
                continue

            # B: VMH Metabolit-Node (Brücke zwischen Graph und Mikrobiom)
            vmh_met_id = f"VMH_MET_{vmh_abbr}"
            self.g.add_node({
                "id": vmh_met_id,
                "type": "VMH_METABOLITE",
                "label": vmh_met.get("fullName", vmh_abbr),
                "vmh_abbreviation": vmh_abbr,
                "charged_formula": vmh_met.get("chargedFormula"),
                "inchi_string": vmh_met.get("inchiString"),
            })

            # C: Mikroben die diesen Metaboliten verarbeiten
            microbe_url = f"{_VMH_BASE}/microbes/?metabolite={quote(vmh_abbr)}&format=json&page_size=10"
            microbe_res = await self.client.get(microbe_url, timeout=15.0)
            if microbe_res.status_code != 200:
                continue
            microbe_data = microbe_res.json().get("results", [])

            for mic in microbe_data:
                mic_name = mic.get("organism") or mic.get("reconstruction")
                if not mic_name:
                    continue

                mic_node_id = f"MICROBE_{mic_name.replace(' ', '_')}"
                self.g.add_node({
                    "id": mic_node_id,
                    "type": "MICROBIAL_STRAIN",
                    "label": mic_name,
                    "phylum": mic.get("phylum"),
                    "family": mic.get("family"),
                    "metabolic_role": "CONSUMER",
                })

                # Edge: Quellmolekül -> Mikrobe
                self.g.add_edge(
                    src=node_id,
                    trgt=mic_node_id,
                    attrs={
                        "rel": "METABOLIZED_BY",
                        "src_layer": "ATOMIC",
                        "trgt_layer": "MICROBIOME",
                    },
                )

            # D: Fermentationsprodukte dieses Metaboliten
            ferm_url = f"{_VMH_BASE}/fermcarbon/?metabolite={quote(vmh_abbr)}&format=json&page_size=10"
            ferm_res = await self.client.get(ferm_url, timeout=15.0)
            if ferm_res.status_code == 200:
                ferm_data = ferm_res.json().get("results", [])
                for ferm in ferm_data:
                    ferm_model = ferm.get("model")
                    source_type = ferm.get("sourcetype", "")
                    if not ferm_model:
                        continue

                    # Fermentations-Mikrobe als PRODUCER markieren
                    ferm_mic_id = f"MICROBE_{ferm_model.replace(' ', '_')}"
                    self.g.add_node({
                        "id": ferm_mic_id,
                        "type": "MICROBIAL_STRAIN",
                        "label": ferm_model,
                        "metabolic_role": "PRODUCER" if "Fermentation" in source_type else "CONSUMER",
                    })

                    self.g.add_edge(
                        src=ferm_mic_id,
                        trgt=vmh_met_id,
                        attrs={
                            "rel": "PRODUCES_METABOLITE",
                            "source_type": source_type,
                            "src_layer": "MICROBIOME",
                            "trgt_layer": "METABOLITE",
                        },
                    )

            # E: Rückverlinkung VMH_METABOLITE -> PROTEIN via VMH Gene-Mapping
            gene_url = f"{_VMH_BASE}/genes/?reaction={quote(vmh_abbr)}&format=json&page_size=5"
            gene_res = await self.client.get(gene_url, timeout=15.0)
            if gene_res.status_code == 200:
                gene_hits = gene_res.json().get("results", [])
                for gh in gene_hits:
                    gene_symbol = gh.get("symbol")
                    if not gene_symbol:
                        continue
                    # Suche passendes PROTEIN über GENE-Node
                    gene_node_id = f"GENE_{gene_symbol}"
                    if self.g.G.has_node(gene_node_id):
                        # TRAVERSE: GENE neighbors to find linked PROTEIN via ENCODED_BY edge
                        for gn_neighbor in self.g.G.neighbors(gene_node_id):
                            if self.g.G.nodes.get(gn_neighbor, {}).get("type") == "PROTEIN":
                                self.g.add_edge(
                                    src=vmh_met_id,
                                    trgt=gn_neighbor,
                                    attrs={
                                        "rel": "TARGETS_HUMAN",
                                        "src_layer": "METABOLITE",
                                        "trgt_layer": "PROTEIN",
                                    },
                                )
                                break

            print(f"Microbiome Enriched: {mol_label} -> VMH:{vmh_abbr} "
                  f"({len(microbe_data)} microbes)")

        except Exception as e:
            print(f"Microbiome Error for {mol_label}: {e}")
