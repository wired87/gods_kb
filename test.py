"""
FULL PIPELINE VALIDATION — same path as server ``generation`` tool.

Uses ``main.run_graph_pipeline`` (query_pipe -> UniprotKB.finalize_biological_graph -> serialize -> ctlr tissue filter) for each case, then loads the exported graph JSON for structural checks.

Prompts: adapt the test workflow to serve the entire workflow same as in the server route;
run the test.py workflow and store outs in project root.

Optional per-query keys on ``VALIDATION_QUERIES`` entries: ``scan_path``, ``modality_hint``,
``filter_physical_compound`` (same semantics as MCP ``generation`` / ``run_graph_pipeline``).

Artifacts (HTML/JSON, tissue maps, ``validation_report.json``) are written under the project root
(same directory as this file), not a subfolder.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- FORCE UTF-8 on Windows ---
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

import networkx as nx

from main import run_graph_pipeline

# Artifacts land next to this file (repo root), not cwd-relative ``output/``.
_PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = _PROJECT_ROOT
_W = 72  # console width

# --- VALIDATION QUERIES (optional: scan_path, modality_hint, filter_physical_compound) ---
VALIDATION_QUERIES: list[dict] = [
    {
        "name": "brain_neurotransmission",
        "prompt": "Dopamine signaling in the brain and its role in Parkinson disease",
    },
    {
        "name": "cardiac_ion_channels",
        "prompt": "Calcium ion channel regulation in heart muscle contraction and arrhythmia",
    },
    {
        "name": "liver_drug_metabolism",
        "prompt": "Cytochrome P450 enzymes in liver drug metabolism and hepatotoxicity risk",
    },
]

_MIN_NODES = 1
_MIN_EDGES = 1
_EXPECTED_CORE_TYPES = {"PROTEIN", "GENE"}


def _step(idx: int, total: int, label: str) -> float:
    """Print a numbered step header and return the start timestamp."""
    print(f"\n  {'-' * (_W - 4)}")
    print(f"  STEP {idx}/{total}: {label}")
    print(f"  {'-' * (_W - 4)}")
    return time.perf_counter()


def _ok(label: str, elapsed: float, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  [OK] {label}  ({elapsed:.2f}s){suffix}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _type_distribution(G) -> dict[str, int]:
    dist: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        t = attrs.get("type", "UNKNOWN")
        dist[t] = dist.get(t, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


def _edge_rel_distribution(G) -> dict[str, int]:
    dist: dict[str, int] = {}
    for _, _, attrs in G.edges(data=True):
        r = attrs.get("rel", "UNKNOWN")
        dist[r] = dist.get(r, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


def _validate_graph(G, cfg: dict, name: str) -> list[str]:
    """Structural validation with per-check progress prints. Returns issues list."""
    issues: list[str] = []
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes < _MIN_NODES:
        issues.append(f"FAIL: only {n_nodes} nodes (min: {_MIN_NODES})")
        _fail(f"[{name}] node count {n_nodes} below minimum {_MIN_NODES}")
    else:
        print(f"  [CHK] node count: {n_nodes}  ok")

    if n_edges < _MIN_EDGES:
        issues.append(f"FAIL: only {n_edges} edges (min: {_MIN_EDGES})")
        _fail(f"[{name}] edge count {n_edges} below minimum {_MIN_EDGES}")
    else:
        print(f"  [CHK] edge count: {n_edges}  ok")

    node_types = {attrs.get("type", "UNKNOWN") for _, attrs in G.nodes(data=True)}
    missing_types = _EXPECTED_CORE_TYPES - node_types
    if missing_types:
        issues.append(f"WARN: missing expected core types: {missing_types}")
        _warn(f"[{name}] missing core types: {missing_types}")
    else:
        print(f"  [CHK] core types {_EXPECTED_CORE_TYPES}: present  ok")

    untyped = [nid for nid, attrs in G.nodes(data=True) if not attrs.get("type")]
    if untyped:
        issues.append(f"FAIL: {len(untyped)} node(s) missing 'type' attribute")
        _fail(f"[{name}] {len(untyped)} untyped node(s) (first: {untyped[0]})")
    else:
        print(f"  [CHK] all nodes have 'type': ok")

    if n_nodes > 1:
        n_comp = nx.number_connected_components(nx.Graph(G))
        frag_ratio = n_comp / n_nodes
        if frag_ratio > 0.5:
            issues.append(f"WARN: graph is highly fragmented ({n_comp} components / {n_nodes} nodes)")
            _warn(f"[{name}] fragmentation ratio {frag_ratio:.2f} > 0.5")
        else:
            print(f"  [CHK] connectivity: {n_comp} components / {n_nodes} nodes  (ratio {frag_ratio:.2f})  ok")

    organs_lc = {o.lower() for o in cfg.get("organs", [])}
    if organs_lc:
        found = any(
            any(o in (attrs.get("label") or "").lower() for o in organs_lc)
            for _, attrs in G.nodes(data=True)
        )
        if not found:
            issues.append(f"WARN: no node labels reference cfg organs {cfg.get('organs')}")
            _warn(f"[{name}] no nodes reference organs {list(organs_lc)[:3]}")
        else:
            print(f"  [CHK] organ label presence: ok")

    ppi_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("rel") == "INTERACTS_WITH")
    if ppi_edges == 0:
        issues.append("WARN: no INTERACTS_WITH edges (PPI layer empty)")
        _warn(f"[{name}] no protein-protein interaction edges found")
    else:
        print(f"  [CHK] PPI edges (INTERACTS_WITH): {ppi_edges}  ok")

    dis_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("rel") == "ASSOCIATED_WITH_DISEASE")
    if dis_edges == 0:
        issues.append("WARN: no ASSOCIATED_WITH_DISEASE edges (disease layer empty)")
        _warn(f"[{name}] no disease association edges found")
    else:
        print(f"  [CHK] disease edges (ASSOCIATED_WITH_DISEASE): {dis_edges}  ok")

    return issues


def _load_graph_from_pipeline_json(json_path: str):
    """Rebuild NetworkX graph from ``nx.node_link_data`` export (same as pipeline JSON)."""
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    # NX 3.x: kw is ``link`` (default "links"); older docs/examples wrongly say ``edges``.
    return nx.node_link_graph(payload, link="links")


async def run_single_query(entry: dict, idx: int, total: int) -> dict:
    """Execute ``run_graph_pipeline`` (server-equivalent) then validate exported graph."""
    name = entry["name"]
    prompt = entry["prompt"]
    scan_path = entry.get("scan_path")
    modality_hint = entry.get("modality_hint")
    filter_physical = entry.get("filter_physical_compound")

    print(f"\n{'=' * _W}")
    print(f"  QUERY {idx}/{total}  [{name}]")
    print(f"  {prompt}")
    print(f"{'=' * _W}")

    t0 = time.perf_counter()
    error: str | None = None
    pr: dict | None = None
    cfg: dict | None = None
    validation_issues: list[str] = []
    G: nx.Graph | None = None
    n_nodes = 0
    n_edges = 0
    json_path = str(OUTPUT_DIR / f"{name}_graph.json")
    html_path = str(OUTPUT_DIR / f"{name}_graph.html")

    t_step = _step(1, 2, "run_graph_pipeline — query_pipe -> UniprotKB -> ctlr (server route)")
    try:
        pr = await run_graph_pipeline(
            prompt=prompt,
            scan_path=scan_path,
            modality_hint=modality_hint,
            dest_html=html_path,
            dest_json=json_path,
            filter_physical_compound=filter_physical,
        )
        cfg = pr.get("cfg") or {}
        n_nodes = int(pr.get("stats", {}).get("nodes", 0))
        n_edges = int(pr.get("stats", {}).get("edges", 0))
        _ok("run_graph_pipeline", time.perf_counter() - t_step, f"{n_nodes}N / {n_edges}E")
        print(f"  Organs     ({len(cfg.get('organs') or [])}): {cfg.get('organs')}")
        print(f"  Physical filter: {cfg.get('filter_physical_compound') or '(none — full workflow)'}")
        pipe = pr.get("pipe_result") or {}
        print(f"  Functions (ctlr) ({len(pipe.get('function_annotation') or [])}): "
              f"{pipe.get('function_annotation', [])}")
        print(f"  Outsrc (ctlr) ({len(pipe.get('outsrc_criteria') or [])}): "
              f"{pipe.get('outsrc_criteria', [])}")
        st = pr.get("stats") or {}
        print(f"  Tissue map: {st.get('tissue_map_nodes', 0)}N / {st.get('tissue_map_edges', 0)}E | "
              f"filtered (ctlr): {st.get('filtered_tissue_map_nodes', 0)}N / "
              f"{st.get('filtered_tissue_map_edges', 0)}E")
        print(f"  Artifacts: {pr.get('html_path')}  |  {pr.get('json_path')}")

        if not cfg.get("organs"):
            _warn(f"[{name}] no organs extracted — graph may be very sparse")

        if n_nodes == 0:
            _warn(f"[{name}] graph has 0 nodes — seeding or API may have failed")
        if n_edges == 0:
            _warn(f"[{name}] graph has 0 edges — enrichment phases may have returned no data")

    except Exception as exc:
        error = f"run_graph_pipeline failed: {exc}"
        _fail(f"[{name}] {error}")
        return {
            "name": name, "prompt": prompt, "error": error,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "nodes": 0, "edges": 0,
            "validation": ["FAIL: pipeline did not complete"],
            "status": "FAIL",
            "pipeline": None,
        }

    t_step = _step(2, 2, "Graph structural validation (from pipeline JSON)")
    try:
        G = _load_graph_from_pipeline_json(pr["json_path"])
        validation_issues = _validate_graph(G, cfg, name)
    except Exception as exc:
        err_msg = f"validation load failed: {exc}"
        error = error or err_msg
        validation_issues = [f"FAIL: could not load graph JSON: {exc}"]
        _fail(f"[{name}] {validation_issues[0]}")

    status = "PASS" if not validation_issues else (
        "WARN" if all(i.startswith("WARN") for i in validation_issues) else "FAIL"
    )
    _ok("validation", time.perf_counter() - t_step,
        f"status={status}  issues={len(validation_issues)}")

    node_dist: dict[str, int] = {}
    edge_dist: dict[str, int] = {}
    if G is not None:
        node_dist = _type_distribution(G)
        edge_dist = _edge_rel_distribution(G)
    top_types = list(node_dist.items())[:6]
    top_rels = list(edge_dist.items())[:6]

    elapsed = time.perf_counter() - t0
    print(f"\n  [{status}] {name}  {n_nodes}N / {n_edges}E  in {elapsed:.1f}s")
    if top_types:
        print(f"  Top types: {', '.join(f'{t}={c}' for t, c in top_types)}")
    if top_rels:
        print(f"  Top rels:  {', '.join(f'{r}={c}' for r, c in top_rels)}")

    pipeline_keys = (
        "cfg", "html_path", "json_path", "stats",
        "tissue_hierarchy_json_path", "filtered_tissue_hierarchy_json_path",
    )
    pipeline_subset = {k: pr[k] for k in pipeline_keys if k in pr} if pr else None

    return {
        "name": name,
        "prompt": prompt,
        "cfg": cfg,
        "nodes": n_nodes,
        "edges": n_edges,
        "node_types": node_dist,
        "edge_relations": edge_dist,
        "validation": validation_issues,
        "status": status,
        "elapsed_sec": round(elapsed, 2),
        "json_path": pr.get("json_path") if pr and n_nodes > 0 else None,
        "html_path": pr.get("html_path") if pr and n_nodes > 0 else None,
        "tissue_hierarchy_json_path": pr.get("tissue_hierarchy_json_path") if pr else None,
        "filtered_tissue_hierarchy_json_path": pr.get("filtered_tissue_hierarchy_json_path") if pr else None,
        "pipeline_stats": pr.get("stats") if pr else None,
        "error": error,
        "pipeline": pipeline_subset,
    }


async def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "paste_your_key_here":
        raise EnvironmentError("Set GEMINI_API_KEY in .env before running.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_run = time.perf_counter()
    ts_start = datetime.now(timezone.utc).isoformat()

    print(f"{'=' * _W}")
    print(f"  PIPELINE VALIDATION —  {len(VALIDATION_QUERIES)} queries  (server route: run_graph_pipeline)")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"  Started: {ts_start}")
    print(f"{'=' * _W}")

    results: list[dict] = []
    for i, entry in enumerate(VALIDATION_QUERIES, start=1):
        summary = await run_single_query(entry, idx=i, total=len(VALIDATION_QUERIES))
        results.append(summary)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(VALIDATION_QUERIES),
        "total_elapsed_sec": round(time.perf_counter() - t_run, 2),
        "results": results,
    }
    report_path = str(OUTPUT_DIR / "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'=' * _W}")
    print("  VALIDATION SUMMARY")
    print(f"{'-' * _W}")
    pass_count = warn_count = fail_count = 0
    for r in results:
        s = r.get("status", "UNKNOWN")
        if s == "PASS":
            pass_count += 1
        elif s == "WARN":
            warn_count += 1
        else:
            fail_count += 1

        err_tag = f"  ERR={r['error']}" if r.get("error") else ""
        print(f"  [{s:4s}] {r['name']:30s}  {r['nodes']:5d}N  {r['edges']:5d}E  "
              f"{r['elapsed_sec']:6.1f}s{err_tag}")

        top = list(r.get("node_types", {}).items())[:5]
        if top:
            print(f"         types: {', '.join(f'{t}={c}' for t, c in top)}")
        top_r = list(r.get("edge_relations", {}).items())[:5]
        if top_r:
            print(f"         rels:  {', '.join(f'{rel}={c}' for rel, c in top_r)}")
        pst = r.get("pipeline_stats") or {}
        if pst:
            print(
                f"         tissue: {pst.get('tissue_map_nodes', 0)}N -> filtered "
                f"{pst.get('filtered_tissue_map_nodes', 0)}N (ctlr)"
            )
        for issue in r.get("validation", []):
            print(f"         {issue}")

    total_sec = time.perf_counter() - t_run
    print(f"{'-' * _W}")
    print(f"  PASS={pass_count}  WARN={warn_count}  FAIL={fail_count}"
          f"  (total: {total_sec:.1f}s)")
    print(f"  Report saved: {report_path}")
    print(f"{'=' * _W}\n")


if __name__ == "__main__":
    asyncio.run(main())
