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

async def _wire_go_hierarchy(self):
    """
    Build IS_A / PART_OF / REGULATES edges between GO_TERM nodes.
    Uses QuickGO /ontology/go/terms/{ids}/children.

    IMPROVEMENT: when a child node is already in the graph but its intermediate
    parent is NOT, we add the parent as a minimal stub node so the DAG path is
    preserved and can be enriched later by _enrich_go_term_metadata.
    This prevents silent disconnections in sparse GO sub-graphs.
    """
    _CHILDREN_BASE = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"
    _BATCH_SIZE = 25
    _VALID_RELS = {"is_a", "part_of", "regulates", "positively_regulates", "negatively_regulates"}
    go_nodes = [(n, d) for n, d in self.g.G.nodes(data=True) if d.get("type") == "GO_TERM"]
    # MUTABLE set: new stub nodes are added so later batches can reference them
    go_ids_in_graph: set[str] = {n for n, _ in go_nodes}
    hierarchy_count = 0
    stubs_added = 0

    for i in range(0, len(go_nodes), _BATCH_SIZE):
        chunk = go_nodes[i : i + _BATCH_SIZE]
        ids_csv = ",".join(d["go_id"] for _, d in chunk if d.get("go_id"))
        if not ids_csv:
            continue

        try:
            res = await self.client.get(
                f"{_CHILDREN_BASE}/{quote(ids_csv, safe='')}/children",
                headers={"Accept": "application/json"},
                timeout=25.0,
            )
            if res.status_code != 200:
                continue

            for term in res.json().get("results", []):
                parent_go = term.get("id")
                if not parent_go:
                    continue
                parent_node_id = f"GO_{parent_go.replace(':', '_')}"
                # Parent is always one of our queried nodes, so it must be in graph
                if parent_node_id not in go_ids_in_graph:
                    continue

                for child in term.get("children", []):
                    child_go = child.get("id")
                    relation = child.get("relation", "").lower()
                    if not child_go or relation not in _VALID_RELS:
                        continue

                    child_node_id = f"GO_{child_go.replace(':', '_')}"

                    # STUB PARENT: child is in graph but sibling-path parent is absent
                    # → add a minimal GO_TERM stub so the hierarchy edge is not lost
                    if child_node_id not in go_ids_in_graph:
                        self.g.add_node({
                            "id": child_node_id, "type": "GO_TERM",
                            "label": child_go, "go_id": child_go,
                            "stub": True,  # marks node for future metadata enrichment
                        })
                        go_ids_in_graph.add(child_node_id)
                        stubs_added += 1

                    # DIRECTION: child -[IS_A/PART_OF/…]-> parent (standard GO DAG)
                    self.g.add_edge(
                        src=child_node_id, trgt=parent_node_id,
                        attrs={
                            "rel": relation.upper(),
                            "src_layer": "GO_HIERARCHY",
                            "trgt_layer": "GO_HIERARCHY",
                        },
                    )
                    hierarchy_count += 1

        except Exception as e:
            print(f"GO Hierarchy Error: {e}")

    print(f"Phase 13a++: {hierarchy_count} GO hierarchy edges created ({stubs_added} stub nodes added)")

# ── GENE -> GO_TERM DERIVED EDGES ────────────────────────────────
_ASPECT_TO_REL = {
    "molecular_function": "HAS_FUNCTION",
    "biological_process": "INVOLVED_IN_PROCESS",
    "cellular_component": "LOCATED_IN_COMPONENT",
}

