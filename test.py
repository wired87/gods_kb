"""
Smoke the full stack through ``main.main_workflow`` only.

Prompt (user): adapt the test.py functions as a senior developer with simple and quick edits on
the codebase to rely heavily on the main workflow (just paste hardcoded data to it). AVOID double
graph ingest — build NetworkX once from the pipeline dict, not by re-reading the exported JSON.

Artifacts: ``*_graph.json``, ``*_graph.html``, ``validation_report.json`` next to this file.
"""
from __future__ import annotations

import asyncio
import json

import pprint
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

load_dotenv()

import warnings

# CHAR: upstream ``google.generativeai`` deprecation noise during query_pipe import.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"All support for the `google.generativeai` package has ended.*",
)


from main import main_workflow

_ROOT = Path(__file__).resolve().parent
OUT = _ROOT
_W = 72

CASES: list[dict] = [
    {
        "name": "brain_neurotransmission",
        "prompt": "Drug with positive impact on the CNS",
        "organs": ["brain"],
        "filter_physical_compound": ["organ", "tissue", "protein", "gene"],
        "result_specs": [{"target_organs": ["brain"], "primary_route": "intravenous"}],
    },
]




async def _run_one(entry: dict) -> dict:
    name = entry["name"]
    json_path = str(OUT / f"{name}_graph.json")
    html_path = str(OUT / f"{name}_graph.html")
    design_dir = OUT / f"{name}_design_artifacts"
    filtered_tissue_hierarchy_json_path = OUT / f"{name}_filtered_tissue_hierarchy"

    print("testing", name)

    t0 = time.perf_counter()

    pr = await main_workflow(
        prompt=entry["prompt"],
        dest_html=html_path,
        dest_json=json_path,
        result_specs=entry.get("result_specs"),
    )
    print(f"main workflow fin after {time.perf_counter() - t0} s")

    return {
        "name": name,
        "pr": pr,
        "error": None,
    }


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_run = time.perf_counter()
    print("start test...")

    tasks = [_run_one(c) for i, c in enumerate(CASES, start=1)]

    results = await asyncio.gather(*tasks)

    report_path = OUT / "validation_report.json"
    report_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": results,
                "total_elapsed_sec": round(time.perf_counter() - t_run, 2),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("results:")
    pprint.pp(results)

if __name__ == "__main__":
    asyncio.run(main())
