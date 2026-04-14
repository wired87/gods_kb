"""
uniprot_kb — compatibility shim (implementation in data.main).

Prompt: Allocate former ``uniprot_kb`` module to ``data/main.py``; keep this module for
``from uniprot_kb import UniprotKB``. Run CLI via ``python -m data.main`` or ``python uniprot_kb.py``.
"""
from __future__ import annotations

import runpy
from pathlib import Path

from data.main import ScanIngestionLayer, UniprotKB, graph_workflow_api_manifest

__all__ = ["ScanIngestionLayer", "UniprotKB", "graph_workflow_api_manifest"]

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "data" / "main.py"), run_name="__main__")
