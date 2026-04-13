"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``finalize_biological_graph``.

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

def compute_sequence_hashes(self):
    """
    Content-Addressing: Jede Aminosäuresequenz bekommt einen SHA-256 Hash-Node.
    Proteine mit identischer Sequenz werden über SEQUENCE_HASH verschmolzen.
    """
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "PROTEIN" and self._is_active(k)]
    _hash_map: dict[str, list[str]] = {}
    hashed = 0

    for node_id, protein in protein_nodes:
        seq = protein.get("sequence")
        if not seq:
            continue

        seq_hash = self._sha256_sequence(seq)
        protein["sequence_hash"] = seq_hash

        if seq_hash not in _hash_map:
            _hash_map[seq_hash] = []
        _hash_map[seq_hash].append(node_id)

    for seq_hash, protein_ids in _hash_map.items():
        hash_node_id = f"SEQHASH_{seq_hash[:16]}"
        self.g.add_node({
            "id": hash_node_id,
            "type": "SEQUENCE_HASH",
            "label": f"SHA256:{seq_hash[:16]}",
            "full_hash": seq_hash,
            "sequence_count": len(protein_ids),
        })
        for pid in protein_ids:
            self.g.add_edge(
                src=pid, trgt=hash_node_id,
                attrs={
                    "rel": "HAS_SEQUENCE_IDENTITY",
                    "src_layer": "PROTEIN",
                    "trgt_layer": "SEQUENCE_HASH",
                },
            )
        hashed += len(protein_ids)

    # DEDUP MARKER: wenn mehrere Proteine denselben Hash teilen
    shared = {h: ids for h, ids in _hash_map.items() if len(ids) > 1}
    for seq_hash, protein_ids in shared.items():
        hash_node_id = f"SEQHASH_{seq_hash[:16]}"
        for i, pid_a in enumerate(protein_ids):
            for pid_b in protein_ids[i + 1:]:
                self.g.add_edge(
                    src=pid_a, trgt=pid_b,
                    attrs={
                        "rel": "SEQUENCE_IDENTICAL",
                        "via_hash": hash_node_id,
                        "src_layer": "PROTEIN",
                        "trgt_layer": "PROTEIN",
                    },
                )

    print(f"Sequence Hashing: {hashed} proteins hashed, "
          f"{len(shared)} shared sequences detected")

# ═══════════════════════════════════════════════════════════════════
# PHASE 11: STRUCTURAL INFERENCE (AlphaFold DB)
# Verknüpft Proteine mit vorhergesagten 3D-Strukturen.
# pLDDT-Score dient als Qualitätsmetrik der Vorhersage.
# ═══════════════════════════════════════════════════════════════════

