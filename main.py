from __future__ import annotations

from ctlr import run_controller
from designer import Designer, normalize_result_specs
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from dotenv import load_dotenv

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
    organs_override: Sequence[str] | None = None,
) -> dict:
    from query_pipe import run_query_pipe
    from firegraph.graph import GUtils
    from uniprot_kb import UniprotKB
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

    _phys = list(pipe_result.get("filter_physical_compound") or [])

    cfg = {
        "organs": pipe_result.get("organs", []),
        "filter_physical_compound": _phys,
    }

    if organs_override:
        cfg["organs"] = list(organs_override)

    # ── STAGE 2: UNIPROTKB GRAPH BUILD ───────────────────────────────
    _banner("STAGE 2 / 3  —  UNIPROTKB: Building hierarchical knowledge graph …")
    g = GUtils()
    kb = UniprotKB(g)
    t2 = time.perf_counter()

    try:
        # BUILD KB UNFILTERED
        await kb.main(
            cfg["organs"],
            cfg["filter_physical_compound"],
        )
        t2_done = time.perf_counter() - t2
        print(f"graph buildup finished after {t2_done} s")

        # ── STAGE 3: SERIALIZE + SAVE ─────────────────────────────────
        _banner("STAGE 3 / 3  —  SERIALIZE & SAVE")
        t3 = time.perf_counter()
        g.check_serilize(g.G)
        nodes = {nid: dict(nd) for nid, nd in g.G.nodes(data=True)}
        edges = [{"src": u, "trgt": v, **dict(ed)} for u, v, ed in g.G.edges(data=True)]
        _ok("serialise nodes/edges", time.perf_counter() - t3, f"{len(nodes)} nodes, {len(edges)} edges")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # HTML VISUALIZATION
        html_target = dest_html or str(OUTPUT_DIR / GRAPH_HTML_BASENAME)
        t_html = time.perf_counter()
        kb.visualize_graph(dest_path=html_target)
        _ok("HTML visualization", time.perf_counter() - t_html, html_target)

        # JSON GRAPH
        json_path = dest_json or str(OUTPUT_DIR / GRAPH_JSON_BASENAME)
        t_json = time.perf_counter()
        data = nx.node_link_data(g.G)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
        _ok("JSON export", time.perf_counter() - t_json, json_path)

        # CTLR WF (outsrc, functinal
        filtered_mx = await run_controller(
            g=kb.g,
            outsrc_specs=pipe_result.get("outsrc_criteria", []),
            functional_annotations=pipe_result.get("function_annotation", []),
        )

        # SAVE FILTERED
        filtered_tissue_link = nx.node_link_data(filtered_mx)

        # DESIGN THE PRODUCT
        specs = normalize_result_specs(result_specs or [])
        """des = Designer(
            gutils=GUtils(G=filtered_mx),
            result_specs=specs,
        )"""
        total_sec = time.perf_counter() - t_total
        print(f"MAIN WORKFLOW FINISHED after {total_sec} s...")
        return kb.g
    finally:
        await kb.close()


def run(prompt: str | None = None, **kwargs) -> dict:
    """Synchronous wrapper for CLI use."""
    if prompt is None:
        prompt = input("Enter biological query: ").strip()
    return asyncio.run(main_workflow(prompt, **kwargs))


if __name__ == "__main__":
    run()