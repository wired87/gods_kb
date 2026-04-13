"""
FULL PIPELINE VALIDATION — query_pipe → UniprotKB workflow → graph validation.

Runs 3 distinct biological queries through the entire pipeline:
    1. query_pipe (Gemini NLP: tokenize, organ resolve, function annotate, outsrc)
    2. UniprotKB.finalize_biological_graph (cfg-driven hierarchical extraction)
    3. Graph structure validation (nodes, edges, types, connectivity)

All output written to output/ directory with per-query artifacts + summary report.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── FORCE UTF-8 on Windows ────────────────────────────────────────
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

from firegraph.graph import GUtils
from uniprot_kb import UniprotKB

OUTPUT_DIR = Path("output")
_W = 72  # console width

# ══════════════════════════════════════════════════════════════════
# 3 VALIDATION QUERIES — diverse biological domains
# ══════════════════════════════════════════════════════════════════
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

# ── MINIMUM THRESHOLDS for a valid graph ──────────────────────────
_MIN_NODES = 1
_MIN_EDGES = 1
_EXPECTED_CORE_TYPES = {"PROTEIN", "GENE"}


# ══════════════════════════════════════════════════════════════════
# CONSOLE HELPERS
# ══════════════════════════════════════════════════════════════════

def _step(idx: int, total: int, label: str) -> float:
    """Print a numbered step header and return the start timestamp."""
    print(f"\n  {'─' * (_W - 4)}")
    print(f"  STEP {idx}/{total}: {label}")
    print(f"  {'─' * (_W - 4)}")
    return time.perf_counter()

def _ok(label: str, elapsed: float, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  [OK] {label}  ({elapsed:.2f}s){suffix}")

def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")

def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ══════════════════════════════════════════════════════════════════
# GRAPH ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════

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
    import networkx as nx
    issues: list[str] = []
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # CHECK 1: minimum size
    if n_nodes < _MIN_NODES:
        issues.append(f"FAIL: only {n_nodes} nodes (min: {_MIN_NODES})")
        _fail(f"[{name}] node count {n_nodes} below minimum {_MIN_NODES}")
    else:
        print(f"  [CHK] node count: {n_nodes}  ✓")

    if n_edges < _MIN_EDGES:
        issues.append(f"FAIL: only {n_edges} edges (min: {_MIN_EDGES})")
        _fail(f"[{name}] edge count {n_edges} below minimum {_MIN_EDGES}")
    else:
        print(f"  [CHK] edge count: {n_edges}  ✓")

    # CHECK 2: core node types present
    node_types = {attrs.get("type", "UNKNOWN") for _, attrs in G.nodes(data=True)}
    missing_types = _EXPECTED_CORE_TYPES - node_types
    if missing_types:
        issues.append(f"WARN: missing expected core types: {missing_types}")
        _warn(f"[{name}] missing core types: {missing_types}")
    else:
        print(f"  [CHK] core types {_EXPECTED_CORE_TYPES}: present  ✓")

    # CHECK 3: every node has a 'type' attribute
    untyped = [nid for nid, attrs in G.nodes(data=True) if not attrs.get("type")]
    if untyped:
        issues.append(f"FAIL: {len(untyped)} node(s) missing 'type' attribute")
        _fail(f"[{name}] {len(untyped)} untyped node(s) (first: {untyped[0]})")
    else:
        print(f"  [CHK] all nodes have 'type': ✓")

    # CHECK 4: connectivity
    if n_nodes > 1:
        n_comp = nx.number_connected_components(nx.Graph(G))
        frag_ratio = n_comp / n_nodes
        if frag_ratio > 0.5:
            issues.append(f"WARN: graph is highly fragmented ({n_comp} components / {n_nodes} nodes)")
            _warn(f"[{name}] fragmentation ratio {frag_ratio:.2f} > 0.5")
        else:
            print(f"  [CHK] connectivity: {n_comp} components / {n_nodes} nodes  (ratio {frag_ratio:.2f})  ✓")

    # CHECK 5: organ terms reflected in graph labels
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
            print(f"  [CHK] organ label presence: ✓")

    # CHECK 6: PPI edges present (new — checks INTERACTS_WITH from cc_interaction)
    ppi_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("rel") == "INTERACTS_WITH")
    if ppi_edges == 0:
        issues.append("WARN: no INTERACTS_WITH edges (PPI layer empty)")
        _warn(f"[{name}] no protein-protein interaction edges found")
    else:
        print(f"  [CHK] PPI edges (INTERACTS_WITH): {ppi_edges}  ✓")

    # CHECK 7: disease edges present (new — checks ASSOCIATED_WITH_DISEASE)
    dis_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("rel") == "ASSOCIATED_WITH_DISEASE")
    if dis_edges == 0:
        issues.append("WARN: no ASSOCIATED_WITH_DISEASE edges (disease layer empty)")
        _warn(f"[{name}] no disease association edges found")
    else:
        print(f"  [CHK] disease edges (ASSOCIATED_WITH_DISEASE): {dis_edges}  ✓")

    return issues


def _save_graph_json(g: GUtils, path: str):
    import networkx as nx
    g.check_serilize(g.G)
    data = nx.node_link_data(g.G)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)


def _resolve_db(db_targets: list[str]) -> str:
    for t in db_targets:
        if "pubchem" in t.lower() or "chemical" in t.lower():
            return "pubchem"
    return "uniprot"


# ══════════════════════════════════════════════════════════════════
# SINGLE QUERY RUNNER
# ══════════════════════════════════════════════════════════════════

async def run_single_query(entry: dict, api_key: str, idx: int, total: int) -> dict:
    """Execute one full pipeline: query_pipe → UniprotKB → validate → save."""
    from query_pipe import run_query_pipe

    name = entry["name"]
    prompt = entry["prompt"]

    print(f"\n{'=' * _W}")
    print(f"  QUERY {idx}/{total}  [{name}]")
    print(f"  {prompt}")
    print(f"{'=' * _W}")

    t0 = time.perf_counter()
    error: str | None = None
    pipe_result = None
    cfg = None
    validation_issues: list[str] = []

    # ── STEP 1: QUERY PIPE ───────────────────────────────────────
    t_step = _step(1, 4, "query_pipe — Gemini NLP extraction")
    try:
        pipe_result = await asyncio.to_thread(run_query_pipe, prompt, api_key)
        cfg = {
            "db": _resolve_db(pipe_result.get("_db_targets", [])),
            "organs": pipe_result.get("organs", []),
            "function_annotation": pipe_result.get("function_annotation", []),
            "outsrc_criteria": pipe_result.get("outsrc_criteria", []),
        }
        _ok("query_pipe", time.perf_counter() - t_step, f"db={cfg['db']}")
        print(f"  Organs     ({len(cfg['organs'])}): {cfg['organs']}")
        print(f"  Functions  ({len(cfg['function_annotation'])}): {cfg['function_annotation']}")
        print(f"  Outsrc     ({len(cfg['outsrc_criteria'])}): {cfg['outsrc_criteria']}")

        if not cfg["organs"] and not cfg["function_annotation"]:
            _warn(f"[{name}] no organs/functions extracted — graph may be very sparse")

    except Exception as exc:
        error = f"query_pipe failed: {exc}"
        _fail(f"[{name}] {error}")
        return {
            "name": name, "prompt": prompt, "error": error,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "nodes": 0, "edges": 0,
            "validation": ["FAIL: query_pipe did not complete"],
        }

    # ── STEP 2: GRAPH BUILD ──────────────────────────────────────
    t_step = _step(2, 4, "UniprotKB.finalize_biological_graph")
    g = GUtils()
    kb = UniprotKB(g)
    t_build = time.perf_counter()
    try:
        await kb.finalize_biological_graph(cfg)
        t_build_done = time.perf_counter() - t_build
        n_nodes = g.G.number_of_nodes()
        n_edges = g.G.number_of_edges()
        _ok("graph build", t_build_done, f"{n_nodes}N / {n_edges}E")

        # Guard gates after build
        if n_nodes == 0:
            _warn(f"[{name}] graph has 0 nodes — seeding or API may have failed")
        if n_edges == 0:
            _warn(f"[{name}] graph has 0 edges — enrichment phases may have returned no data")

    except Exception as exc:
        error = f"graph build failed: {exc}"
        _fail(f"[{name}] {error}")
        n_nodes = g.G.number_of_nodes()
        n_edges = g.G.number_of_edges()
    finally:
        await kb.close()

    # ── STEP 3: VALIDATE ─────────────────────────────────────────
    t_step = _step(3, 4, "Graph structural validation")
    if error is None:
        validation_issues = _validate_graph(g.G, cfg, name)
    else:
        validation_issues = [f"SKIPPED (build error): {error}"]
        _warn(f"[{name}] validation skipped due to build error")

    status = "PASS" if not validation_issues else (
        "WARN" if all(i.startswith("WARN") for i in validation_issues) else "FAIL"
    )
    _ok("validation", time.perf_counter() - t_step,
        f"status={status}  issues={len(validation_issues)}")

    # ── STEP 4: SAVE ARTIFACTS ───────────────────────────────────
    t_step = _step(4, 4, "Save artifacts (JSON + HTML)")
    json_path = str(OUTPUT_DIR / f"{name}_graph.json")
    html_path = str(OUTPUT_DIR / f"{name}_graph.html")

    if n_nodes > 0:
        try:
            t_json = time.perf_counter()
            _save_graph_json(g, json_path)
            _ok("JSON saved", time.perf_counter() - t_json, json_path)

            t_html = time.perf_counter()
            kb.visualize_graph(dest_path=html_path)
            _ok("HTML saved", time.perf_counter() - t_html, html_path)
        except Exception as save_exc:
            _fail(f"[{name}] save error: {save_exc}")
    else:
        _warn(f"[{name}] skipping artifact save — empty graph")

    # TOP TYPES / RELATIONS for report
    node_dist = _type_distribution(g.G)
    edge_dist = _edge_rel_distribution(g.G)
    top_types = list(node_dist.items())[:6]
    top_rels  = list(edge_dist.items())[:6]

    elapsed = time.perf_counter() - t0
    print(f"\n  [{status}] {name}  {n_nodes}N / {n_edges}E  in {elapsed:.1f}s")
    print(f"  Top types: {', '.join(f'{t}={c}' for t, c in top_types)}")
    print(f"  Top rels:  {', '.join(f'{r}={c}' for r, c in top_rels)}")

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
        "json_path": json_path if n_nodes > 0 else None,
        "html_path": html_path if n_nodes > 0 else None,
        "error": error,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════

async def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "paste_your_key_here":
        raise EnvironmentError("Set GEMINI_API_KEY in .env before running.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_run = time.perf_counter()
    ts_start = datetime.now(timezone.utc).isoformat()

    print(f"{'=' * _W}")
    print(f"  PIPELINE VALIDATION  —  {len(VALIDATION_QUERIES)} queries")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"  Started: {ts_start}")
    print(f"{'=' * _W}")

    results: list[dict] = []
    for i, entry in enumerate(VALIDATION_QUERIES, start=1):
        summary = await run_single_query(entry, api_key, idx=i, total=len(VALIDATION_QUERIES))
        results.append(summary)

    # ── FINAL REPORT ─────────────────────────────────────────────
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(VALIDATION_QUERIES),
        "total_elapsed_sec": round(time.perf_counter() - t_run, 2),
        "results": results,
    }
    report_path = str(OUTPUT_DIR / "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # ── SUMMARY TABLE ────────────────────────────────────────────
    print(f"\n{'=' * _W}")
    print("  VALIDATION SUMMARY")
    print(f"{'─' * _W}")
    pass_count = warn_count = fail_count = 0
    for r in results:
        s = r.get("status", "UNKNOWN")
        if s == "PASS":   pass_count  += 1
        elif s == "WARN": warn_count  += 1
        else:             fail_count  += 1

        err_tag = f"  ERR={r['error']}" if r.get("error") else ""
        print(f"  [{s:4s}] {r['name']:30s}  {r['nodes']:5d}N  {r['edges']:5d}E  "
              f"{r['elapsed_sec']:6.1f}s{err_tag}")

        top = list(r.get("node_types", {}).items())[:5]
        if top:
            print(f"         types: {', '.join(f'{t}={c}' for t, c in top)}")
        top_r = list(r.get("edge_relations", {}).items())[:5]
        if top_r:
            print(f"         rels:  {', '.join(f'{rel}={c}' for rel, c in top_r)}")
        for issue in r.get("validation", []):
            print(f"         {issue}")

    total_sec = time.perf_counter() - t_run
    print(f"{'─' * _W}")
    print(f"  PASS={pass_count}  WARN={warn_count}  FAIL={fail_count}"
          f"  (total: {total_sec:.1f}s)")
    print(f"  Report saved: {report_path}")
    print(f"{'=' * _W}\n")


if __name__ == "__main__":
    asyncio.run(main())
