"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``finalize_biological_graph``.

Prompt (user): data-dir graph hardening — EXCITATION_FREQUENCY uses ``EXCFREQ_*`` from SMILES + state only.

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

from data.graph_identity import canonical_node_id

def compute_electron_density_matrices(self):
    """
    PHASE 18: RDKit + PySCF DFT(B3LYP)/def2-SVP + ddCOSMO water; TD-DFT excitations.

    Inputs: active nodes with non-empty ``smiles`` (typically small-molecule / structure layer).
    Outputs: in-place attrs on the SMILES node (density matrix triangle, SCF metadata);
    ``EXCFREQ_*`` nodes (opaque ids from SMILES + state index only), edges ``HAS_EXCITATION``,
    ``HAS_FREQ_RESULT`` / ``HAS_DENSITY_RESULT`` to ``_PARENT_LAYER_TYPES`` neighbors.
    Side effects: heavy CPU/GPU-free quantum chemistry; requires ``rdkit`` and ``pyscf``.
    Failures: ImportError prints skip; per-node DFT/TD errors print and skip frequencies only.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from pyscf import gto, dft, tddft, solvent
        import numpy as np
    except ImportError as exc:
        print(f"PHASE 18 SKIP – fehlende Abhängigkeit: {exc}")
        return

    # ── ALLE NODES MIT GÜLTIGEM SMILES SAMMELN ──────────────────
    smiles_nodes = [
        (nid, data) for nid, data in self.g.G.nodes(data=True)
        if data.get("smiles") and data["smiles"] not in (None, "N/A") and self._is_active(nid)
    ]
    if not smiles_nodes:
        print("  Keine Nodes mit SMILES gefunden – überspringe.")
        return

    computed, skipped = 0, 0

    for node_id, node in smiles_nodes:
        smiles = node["smiles"]

        # ── SCHRITT 1: RDKit – SMILES → 3D-Koordinaten ──────────
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  RDKit parse-Fehler: {smiles[:60]} – skip")
            skipped += 1
            continue

        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            print(f"  3D-Embedding fehlgeschlagen: {smiles[:60]} – skip")
            skipped += 1
            continue
        AllChem.MMFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        atoms_pyscf = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atoms_pyscf.append([atom.GetSymbol(), [pos.x, pos.y, pos.z]])

        # ── SCHRITT 2: DFT(B3LYP) + ddCOSMO(Wasser) für in-vivo Relevanz
        try:
            pyscf_mol = gto.Mole(verbose=0)
            pyscf_mol.atom = atoms_pyscf
            pyscf_mol.basis = "def2-SVP"
            pyscf_mol.build()

            mf = dft.RKS(pyscf_mol)
            mf.xc = "B3LYP"
            mf = solvent.ddCOSMO(mf)
            mf.with_solvent.eps = 78.3553  # Wasser-Dielektrikum (in vivo)
            mf.kernel()

            rdm1 = mf.make_rdm1()
        except Exception as exc:
            print(f"  PySCF-Fehler für {node_id}: {exc} – skip")
            skipped += 1
            continue

        # ── SCHRITT 3: Ergebnisse auf Node + Eltern-Layer ────────
        n_basis = rdm1.shape[0]
        # OBERES DREIECK (symmetrisch) als flache Liste → JSON-tauglich
        upper_tri = rdm1[np.triu_indices(n_basis)].tolist()

        node["atom_decomposition"] = atoms_pyscf
        node["electron_density_matrix"] = upper_tri
        node["density_matrix_shape"] = [n_basis, n_basis]
        node["density_matrix_basis"] = "def2-SVP"
        node["total_electrons"] = float(np.trace(rdm1))
        node["total_energy_hartree"] = float(mf.e_tot)
        node["scf_converged"] = bool(mf.converged)

        parent_neighbors = [
            nb for nb in self.g.G.neighbors(node_id)
            if self.g.G.nodes[nb].get("type") in self._PARENT_LAYER_TYPES
        ]

        for neighbor in parent_neighbors:
            self.g.add_edge(
                src=neighbor, trgt=node_id,
                attrs={
                    "rel": "HAS_DENSITY_RESULT",
                    "total_electrons": node["total_electrons"],
                    "total_energy_hartree": node["total_energy_hartree"],
                    "basis": "def2-SVP",
                    "scf_converged": node["scf_converged"],
                    "src_layer": self.g.G.nodes[neighbor].get("type", "PARENT"),
                    "trgt_layer": node.get("type", "ATOMIC_STRUCTURE"),
                },
            )

        # ── SCHRITT 4: TD-DFT → Anregungsenergien + Laser-Frequenzen ─
        src_layer = node.get("type", "ATOMIC_STRUCTURE")
        try:
            td = mf.TDDFT()
            td.nstates = 10
            td.kernel()

            exc_energies_ev = td.e * self._HA_TO_EV
            osc_strengths = td.oscillator_strength(gauge='length')

            for state_idx, (e_ev, f_osc) in enumerate(
                zip(exc_energies_ev, osc_strengths), start=1
            ):
                if f_osc < self._MIN_OSC_STRENGTH:
                    continue

                wl_nm = self._EV_TO_NM / e_ev
                freq_hz = e_ev * self._EV_TO_HZ

                freq_id = canonical_node_id(
                    "EXCFREQ",
                    {"smiles": smiles, "state_idx": int(state_idx)},
                )
                self.g.add_node({
                    "id": freq_id,
                    "type": "EXCITATION_FREQUENCY",
                    "label": f"S0->S{state_idx} {wl_nm:.1f}nm",
                    "excitation_state_index": int(state_idx),
                    "source_smiles": smiles,
                    "excitation_energy_ev": round(float(e_ev), 6),
                    "wavelength_nm": round(float(wl_nm), 2),
                    "frequency_hz": float(freq_hz),
                    "oscillator_strength": round(float(f_osc), 6),
                    "in_nir_window": bool(self._NIR_LOW <= wl_nm <= self._NIR_HIGH),
                    "solvent": "water",
                    "basis": "def2-SVP",
                    "xc_functional": "B3LYP",
                })

                self.g.add_edge(
                    src=node_id, trgt=freq_id,
                    attrs={
                        "rel": "HAS_EXCITATION",
                        "src_layer": src_layer,
                        "trgt_layer": "PHOTOPHYSICS",
                    },
                )

                for neighbor in parent_neighbors:
                    self.g.add_edge(
                        src=neighbor, trgt=freq_id,
                        attrs={
                            "rel": "HAS_FREQ_RESULT",
                            "wavelength_nm": round(float(wl_nm), 2),
                            "oscillator_strength": round(float(f_osc), 6),
                            "in_nir_window": bool(self._NIR_LOW <= wl_nm <= self._NIR_HIGH),
                            "src_layer": self.g.G.nodes[neighbor].get("type", "PARENT"),
                            "trgt_layer": "EXCITATION_FREQUENCY",
                        },
                    )

                nir_tag = " [NIR]" if self._NIR_LOW <= wl_nm <= self._NIR_HIGH else ""
                print(f"    FREQ S{state_idx}: {wl_nm:.1f} nm  f={f_osc:.4f}{nir_tag}")

        except Exception as exc:
            print(f"  TD-DFT-Fehler für {node_id}: {exc} – Frequenzen übersprungen")

        computed += 1
        print(f"  RDM1+TD OK: {node_id}  ({n_basis}x{n_basis}, E={mf.e_tot:.6f} Ha)")

    print(f"  Fertig – {computed} berechnet, {skipped} übersprungen.")

# ═══════════════════════════════════════════════════════════════════
# PHASE 19: BIOELECTRIC → DISEASE SIGNAL PIPELINE
# Brücke von Biologie → messbares Signal → Krankheit.
# 7 Passes: Disease Ontology, Bioelectric State, EM Signature,
#           Multi-Scale Aggregation, Scan Signal, Inverse Inference.
# ═══════════════════════════════════════════════════════════════════

# ── OpenTargets: full disease query (broader than allergen-only) ──
_OT_FULL_DISEASE_QUERY = """
query($ensgId: String!) {
  target(ensemblId: $ensgId) {
    approvedSymbol
    associatedDiseases(page: {size: 200, index: 0}) {
      rows {
        disease { id name therapeuticAreas { id label } }
        score
        datatypeScores { id score }
      }
    }
  }
}
"""

