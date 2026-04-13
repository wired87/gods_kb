"""
main — End-to-end biological knowledge graph builder.

Query → query_pipe (Gemini NLP: organs, functions, outsrc, db routing)
    → UniprotKB.finalize_biological_graph (cfg-driven hierarchical extraction)
    → return full graph plus tissue_hierarchy_map (organ→tissue→…→electron) + optional HTML.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

load_dotenv()

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = Path("output")
_W = 72  # console width


def _banner(title: str, char: str = "=") -> None:
    print(f"\n{char * _W}\n  {title}\n{char * _W}")


def _ok(label: str, elapsed: float, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  [OK] {label}  ({elapsed:.2f}s){suffix}")


def _resolve_db(db_targets: list[str]) -> str:
    """Prefer 'pubchem' when any target mentions it, otherwise 'uniprot'."""
    for t in db_targets:
        if "pubchem" in t.lower() or "chemical" in t.lower():
            return "pubchem"
    return "uniprot"


async def run_graph_pipeline(
    prompt: str,
    scan_path: str | None = None,
    modality_hint: str | None = None,
    dest_html: str | None = None,
) -> dict:
    """
    FULL PIPELINE: user prompt → query_pipe → UniprotKB → graph dict.

    Returns dict with keys: cfg, nodes, edges, stats, html_path, json_path,
    tissue_hierarchy_map (node-link JSON), tissue_hierarchy_json_path.
    """
    from query_pipe import run_query_pipe
    from firegraph.graph import GUtils
    from uniprot_kb import UniprotKB
    import networkx as nx

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "paste_your_key_here":
        raise EnvironmentError("Set GEMINI_API_KEY in .env before running.")

    t_total = time.perf_counter()

    # ── STAGE 1: QUERY PIPE ───────────────────────────────────────────
    _banner("STAGE 1 / 3  —  QUERY PIPE: Analysing prompt …")
    print(f"  Prompt: {prompt[:120]}")
    t1 = time.perf_counter()
    pipe_result = await asyncio.to_thread(run_query_pipe, prompt, api_key)
    t1_done = time.perf_counter() - t1

    cfg = {
        "db": _resolve_db(pipe_result.get("_db_targets", [])),
        "organs": pipe_result.get("organs", []),
        "function_annotation": pipe_result.get("function_annotation", []),
        "outsrc_criteria": pipe_result.get("outsrc_criteria", []),
    }

    _ok("query_pipe", t1_done, f"db={cfg['db']}")
    print(f"  Organs          ({len(cfg['organs'])}): {cfg['organs']}")
    print(f"  Functions       ({len(cfg['function_annotation'])}): {cfg['function_annotation']}")
    print(f"  Outsrc criteria ({len(cfg['outsrc_criteria'])}): {cfg['outsrc_criteria']}")

    # Guard: if NLP returned nothing useful, warn but continue
    if not cfg["organs"] and not cfg["function_annotation"]:
        print("  WARN: query_pipe returned no organs and no function terms — graph may be sparse")

    # ── STAGE 2: UNIPROTKB GRAPH BUILD ───────────────────────────────
    _banner("STAGE 2 / 3  —  UNIPROTKB: Building hierarchical knowledge graph …")
    g = GUtils()
    kb = UniprotKB(g)
    t2 = time.perf_counter()
    build_result: dict | None = None

    try:
        build_result = await kb.finalize_biological_graph(
            cfg, scan_path=scan_path, modality_hint=modality_hint,
        )
        t2_done = time.perf_counter() - t2
        n_nodes = g.G.number_of_nodes()
        n_edges = g.G.number_of_edges()
        _ok("graph build", t2_done, f"{n_nodes}N / {n_edges}E")

        # Guard: flag unexpectedly empty graphs
        if n_nodes == 0:
            print("  WARN: graph has 0 nodes — check seed resolution and API connectivity")
        elif n_edges == 0:
            print("  WARN: graph has 0 edges — enrichment phases may have failed silently")

        # TOP NODE TYPES (quick diagnostic)
        type_dist: dict[str, int] = {}
        for _, d in g.G.nodes(data=True):
            type_dist[d.get("type", "?")] = type_dist.get(d.get("type", "?"), 0) + 1
        top_types = sorted(type_dist.items(), key=lambda x: -x[1])[:8]
        print(f"  Top node types: {', '.join(f'{t}={c}' for t, c in top_types)}")

        # TOP EDGE RELATIONS (quick diagnostic)
        rel_dist: dict[str, int] = {}
        for _, _, d in g.G.edges(data=True):
            rel_dist[d.get("rel", "?")] = rel_dist.get(d.get("rel", "?"), 0) + 1
        top_rels = sorted(rel_dist.items(), key=lambda x: -x[1])[:8]
        print(f"  Top edge rels:  {', '.join(f'{r}={c}' for r, c in top_rels)}")

        # ── STAGE 3: SERIALIZE + SAVE ─────────────────────────────────
        _banner("STAGE 3 / 3  —  SERIALIZE & SAVE")
        t3 = time.perf_counter()
        g.check_serilize(g.G)
        nodes = {nid: dict(nd) for nid, nd in g.G.nodes(data=True)}
        edges = [{"src": u, "dst": v, **dict(ed)} for u, v, ed in g.G.edges(data=True)]
        _ok("serialise nodes/edges", time.perf_counter() - t3, f"{len(nodes)} nodes, {len(edges)} edges")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # HTML VISUALIZATION
        html_target = dest_html or str(OUTPUT_DIR / "graph.html")
        t_html = time.perf_counter()
        kb.visualize_graph(dest_path=html_target)
        _ok("HTML visualization", time.perf_counter() - t_html, html_target)

        # JSON GRAPH
        json_path = str(OUTPUT_DIR / "graph.json")
        t_json = time.perf_counter()
        data = nx.node_link_data(g.G)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
        _ok("JSON export", time.perf_counter() - t_json, json_path)

        tissue_mx = None
        if isinstance(build_result, dict):
            tissue_mx = build_result.get("tissue_hierarchy_map")
        tissue_hierarchy_json_path = str(OUTPUT_DIR / "tissue_hierarchy_map.json")
        tissue_link: dict = {"nodes": [], "links": []}
        if tissue_mx is not None:
            t_tm = time.perf_counter()
            tissue_link = nx.node_link_data(tissue_mx)
            with open(tissue_hierarchy_json_path, "w", encoding="utf-8") as f:
                json.dump(tissue_link, f, ensure_ascii=False, default=str)
            _ok("tissue hierarchy map JSON", time.perf_counter() - t_tm, tissue_hierarchy_json_path)

        result: dict = {
            "cfg": cfg,
            "pipe_result": {k: v for k, v in pipe_result.items() if not k.startswith("_")},
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "nodes": n_nodes, "edges": n_edges,
                "node_types": dict(top_types), "edge_relations": dict(top_rels),
                "tissue_map_nodes": tissue_mx.number_of_nodes() if tissue_mx is not None else 0,
                "tissue_map_edges": tissue_mx.number_of_edges() if tissue_mx is not None else 0,
            },
            "html_path": html_target,
            "json_path": json_path,
            "tissue_hierarchy_map": tissue_link,
            "tissue_hierarchy_json_path": tissue_hierarchy_json_path if tissue_mx is not None else None,
        }

        total_sec = time.perf_counter() - t_total
        print(f"\n{'=' * _W}")
        print(f"  PIPELINE COMPLETE  —  {n_nodes}N / {n_edges}E  in {total_sec:.1f}s")
        print(f"  HTML  →  {html_target}")
        print(f"  JSON  →  {json_path}")
        if result.get("tissue_hierarchy_json_path"):
            print(f"  Tissue map (organ→electron)  →  {result['tissue_hierarchy_json_path']}")
        print(f"{'=' * _W}\n")

        return result

    finally:
        await kb.close()


def run(prompt: str | None = None, **kwargs) -> dict:
    """Synchronous wrapper for CLI use."""
    if prompt is None:
        prompt = input("Enter biological query: ").strip()
    return asyncio.run(run_graph_pipeline(prompt, **kwargs))


if __name__ == "__main__":
    run()
