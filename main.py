from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from dotenv import load_dotenv

from embedder import embed
from ds import work_human_main
from firegraph.graph.visual import create_g_visual
from query_pipe import TEST_QUERY_SPECS

load_dotenv()

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = Path("output")
# CHAR: server + CLI use the same basenames when writing under a session directory.
GRAPH_HTML_BASENAME = "graph.html"
GRAPH_JSON_BASENAME = "graph.json"
_W = 72  # console width





TEST=True







def _banner(title: str, char: str = "=") -> None:
    print(f"\n{char * _W}\n  {title}\n{char * _W}")

def _ok(label: str, elapsed: float, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  [OK] {label}  ({elapsed:.2f}s){suffix}")


async def main_workflow(
    prompt: str,
    dest_html: str | None = None,
    dest_json: str | None = None,

    result_specs: Sequence[Any] | None = None,
) -> dict:
    from query_pipe import run_query_pipe
    from firegraph.graph import GUtils
    import networkx as nx

    api_key = os.environ.get("GEMINI_API_KEY", "")

    t_total = time.perf_counter()

    # ── STAGE 1: QUERY PIPE ───────────────────────────────────────────
    _banner("STAGE 1 / 3  —  QUERY PIPE: Analysing prompt …")
    print(f"  Prompt: {prompt}")

    if TEST == False:
        pipe_result = await asyncio.to_thread(
            run_query_pipe,
            prompt,
            api_key,
            #filter_physical_compound=filter_physical_compound,
        )
    else:
        pipe_result = TEST_QUERY_SPECS

    print(f"QUERY PIPE FINIESHED AFTER ({pipe_result})S")


    # ── STAGE 2: UNIPROTKB GRAPH BUILD ───────────────────────────────
    _banner("STAGE 2 / 3  —  UNIPROTKB: Building hierarchical knowledge graph …")
    g = GUtils()

    try:
        # BUILD KB UNFILTERED
        await work_human_main(
            g=g,
            functions=[
                embed(item)
                for item in pipe_result["function_annotation"]
            ],
            outsrc=[
                embed(item)
                for item in pipe_result["outsrc_criteria"]
            ],
        )
        # ── STAGE 3: SERIALIZE + SAVE ─────────────────────────────────
        _banner("STAGE 3 / 3  —  SERIALIZE & SAVE")
        g.check_serilize(g.G)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # HTML VISUALIZATION
        html_target = dest_html or str(OUTPUT_DIR / GRAPH_HTML_BASENAME)
        create_g_visual(dest_path=html_target)
  
        # JSON GRAPH
        json_path = dest_json or str(OUTPUT_DIR / GRAPH_JSON_BASENAME)

        data = nx.node_link_data(g.G)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)


        total_sec = time.perf_counter() - t_total
        print(f"MAIN WORKFLOW FINISHED after {total_sec} s...")

    except Exception as e:
        print("Err mian_workflow", e)


def run(prompt: str | None = None, **kwargs) -> dict:
    """Synchronous wrapper for CLI use."""
    if prompt is None:
        prompt = input("Enter biological query: ").strip()
    return asyncio.run(main_workflow(prompt, **kwargs))


if __name__ == "__main__":
    run()