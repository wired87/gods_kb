"""
Acid Master — Architecture & Workflow Visualisation + Brain Scan Energy Pipeline.

Generates three publication-ready diagrams:
  1. workflow_phases.png   — the 18-phase UniProtKB enrichment pipeline
  2. graph_layers.png      — node types, edge relations, external APIs
  3. system_architecture.png — server / agent / CLI integration map

BrainScanIntegrator:
  Transforms neuroimaging data into a position-keyed energy stimulation
  protocol via the firegraph knowledge graph.

Run:  python visual.py
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

import google.generativeai as genai
import httpx
import numpy as np

_OUT = Path("docs")

# ── COLOUR PALETTE ──────────────────────────────────────────────────
_C = {
    "bg":        "#0d1117",
    "card":      "#161b22",
    "border":    "#30363d",
    "text":      "#e6edf3",
    "dim":       "#8b949e",
    "blue":      "#58a6ff",
    "green":     "#3fb950",
    "orange":    "#d29922",
    "red":       "#f85149",
    "purple":    "#bc8cff",
    "cyan":      "#39d2c0",
    "pink":      "#f778ba",
    "yellow":    "#e3b341",
}

def _style_ax(ax, title: str):
    ax.set_facecolor(_C["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold",
                 color=_C["text"], pad=14)


def _box(ax, x, y, w, h, text, color, fontsize=7, alpha=0.92, text_color=None):
    """Draw a rounded box with centered label."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor=_C["border"],
        linewidth=0.8, alpha=alpha,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=text_color or _C["text"],
            fontweight="medium", wrap=True)
    return rect


def _arrow(ax, x1, y1, x2, y2, color=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color or _C["dim"],
                                lw=1.0, shrinkA=4, shrinkB=4))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DIAGRAM 1 — 18-phase enrichment pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_PHASES = [
    ("1",    "Protein & Gene\nIngestion",            "UniProt Proteome",      _C["blue"]),
    ("2",    "Deep UniProt\nDetails",                 "Cofactors, Pathways",   _C["blue"]),
    ("3",    "Pharmacology\n(ChEMBL)",                "Drug → Target",         _C["green"]),
    ("4",    "Molecular Mapping\n(SMILES)",            "PubChem / ChEBI",       _C["green"]),
    ("5",    "Nutritional Origin\n(OpenFoodFacts)",    "Food → Nutrient",       _C["orange"]),
    ("6",    "Genomic &\nFunctional",                 "Ensembl + Reactome",    _C["blue"]),
    ("7",    "Pharmacogenomics\n(ClinPGx)",            "Variant → Gene",        _C["purple"]),
    ("8",    "Bioelectric\n(GtoPdb)",                  "Ion channels",          _C["cyan"]),
    ("9",    "Microbiome\n(VMH)",                      "Gut–Brain axis",        _C["pink"]),
    ("10",   "Cell-Type Expr.\n(HPA)",                 "Tissue atlas",          _C["orange"]),
    ("10.5", "Sequence\nHashing",                      "SHA-256 identity",      _C["dim"]),
    ("11",   "Structural\n(AlphaFold)",                "pLDDT confidence",      _C["cyan"]),
    ("12",   "Domain Decomp.\n(InterPro)",             "Pfam / CDD / SMART",   _C["blue"]),
    ("13a",  "GO Semantic\n(QuickGO)",                 "BP / MF / CC terms",    _C["green"]),
    ("13b",  "Subcellular Loc.\n(COMPARTMENTS)",       "Localisation",          _C["green"]),
    ("13c",  "GO-CAM\nActivities",                     "Causal models",         _C["green"]),
    ("14-16","Allergen\nDetection & Impact",           "KW-0020 + CTD + OT",   _C["red"]),
    ("17",   "Cellular Components\n+ ncRNA",           "HPA + Ensembl NC",      _C["purple"]),
    ("18",   "Electron Density\n(RDKit+PySCF)",        "DFT / TD-DFT / ddCOSMO",_C["yellow"]),
]


def render_workflow_phases():
    fig, ax = plt.subplots(figsize=(14, 20), facecolor=_C["bg"])
    _style_ax(ax, "UniProtKB — 18-Phase Enrichment Pipeline")
    ax.set_ylim(-0.5, 20.5)
    ax.set_xlim(-0.5, 10.5)

    box_w, box_h = 3.4, 0.75
    api_w = 3.0
    x_phase = 0.5
    x_api = 6.5

    # HEADER LABELS
    ax.text(x_phase + box_w / 2, 20.0, "ENRICHMENT PHASE",
            ha="center", fontsize=9, color=_C["dim"], fontweight="bold")
    ax.text(x_api + api_w / 2, 20.0, "EXTERNAL API / DATA",
            ha="center", fontsize=9, color=_C["dim"], fontweight="bold")

    for i, (num, label, api, col) in enumerate(_PHASES):
        y = 19.0 - i * 1.02
        # phase box
        _box(ax, x_phase, y, box_w, box_h, f"Phase {num}\n{label}", col, fontsize=6.5)
        # api box
        _box(ax, x_api, y, api_w, box_h, api, _C["card"], fontsize=6.5)
        # connector
        _arrow(ax, x_phase + box_w, y + box_h / 2, x_api, y + box_h / 2, col)
        # vertical flow arrow (except last)
        if i < len(_PHASES) - 1:
            _arrow(ax, x_phase + box_w / 2, y, x_phase + box_w / 2, y - 0.27, _C["dim"])

    fig.tight_layout()
    fig.savefig(_OUT / "workflow_phases.png", dpi=180, facecolor=_C["bg"])
    plt.close(fig)
    print(f"Saved -> {_OUT / 'workflow_phases.png'}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DIAGRAM 2 — Knowledge-graph node layers & edge relations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_NODE_LAYERS = {
    "CORE": {
        "color": _C["blue"],
        "types": ["PROTEIN", "GENE", "SEQUENCE_HASH"],
    },
    "STRUCTURE": {
        "color": _C["cyan"],
        "types": ["ALPHAFOLD_STRUCTURE", "INTERPRO_DOMAIN", "DENSITY_MATRIX"],
    },
    "CHEMISTRY": {
        "color": _C["green"],
        "types": ["MINERAL", "MOLECULE_CHAIN", "ATOMIC_STRUCTURE", "PHARMA_COMPOUND"],
    },
    "FUNCTION": {
        "color": _C["orange"],
        "types": ["GO_TERM", "GOCAM_ACTIVITY", "REACTOME_PATHWAY", "ECO_EVIDENCE"],
    },
    "BIOLOGY": {
        "color": _C["purple"],
        "types": ["CELL_TYPE", "CELLULAR_COMPONENT", "COMPARTMENT", "NC_GENE"],
    },
    "CLINICAL": {
        "color": _C["red"],
        "types": ["CLINICAL_ANNOTATION", "GENETIC_VARIANT", "ALLERGEN", "ALLERGEN_DISEASE"],
    },
    "EXTERNAL": {
        "color": _C["yellow"],
        "types": ["FOOD_SOURCE", "MICROBIOME_METABOLITE", "ELECTRICAL_COMPONENT"],
    },
    "PHYSICS": {
        "color": _C["pink"],
        "types": ["EXCITATION_FREQUENCY"],
    },
}


def render_graph_layers():
    fig, ax = plt.subplots(figsize=(16, 11), facecolor=_C["bg"])
    _style_ax(ax, "UniProtKB — Runtime Knowledge-Graph Layers")
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-0.5, 11)

    col_x = [0.3, 4.2, 8.1, 12.0]
    start_y = 9.5
    row_h = 2.3
    layers = list(_NODE_LAYERS.items())

    for idx, (layer_name, info) in enumerate(layers):
        col = idx % 4
        row = idx // 4
        x = col_x[col]
        y = start_y - row * row_h

        # layer header
        ax.text(x + 1.6, y + 0.15, layer_name,
                fontsize=9, fontweight="bold", color=info["color"], ha="center")

        for j, ntype in enumerate(info["types"]):
            ny = y - 0.45 - j * 0.42
            _box(ax, x, ny, 3.2, 0.36, ntype, _C["card"], fontsize=6.5,
                 text_color=info["color"])

    # EDGE LEGEND (bottom)
    _EDGES = [
        ("ENCODED_BY",       _C["blue"]),
        ("MODULATES_TARGET", _C["green"]),
        ("HAS_DOMAIN",       _C["cyan"]),
        ("ANNOTATED_WITH",   _C["orange"]),
        ("EXPRESSED_IN",     _C["purple"]),
        ("CLINICAL_SIG.",    _C["red"]),
        ("CONTAINS_NUTRIENT",_C["yellow"]),
        ("DENSITY_COMPUTED", _C["pink"]),
    ]
    ax.text(8.0, 1.2, "KEY EDGE RELATIONS", fontsize=9,
            color=_C["dim"], fontweight="bold", ha="center")
    for i, (rel, col) in enumerate(_EDGES):
        cx = 0.5 + (i % 4) * 4.0
        cy = 0.7 if i < 4 else 0.2
        ax.plot(cx, cy, "s", color=col, markersize=7)
        ax.text(cx + 0.3, cy, rel, fontsize=6.5, color=_C["text"], va="center")

    fig.tight_layout()
    fig.savefig(_OUT / "graph_layers.png", dpi=180, facecolor=_C["bg"])
    plt.close(fig)
    print(f"Saved -> {_OUT / 'graph_layers.png'}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DIAGRAM 3 — System architecture (server + agent + CLI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=_C["bg"])
    _style_ax(ax, "Acid Master — System Architecture")
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 9.5)

    # ── AGENT LAYER (top) ───────────────────────────────────────────
    ax.text(7.0, 9.0, "AGENT  LAYER", fontsize=10, ha="center",
            color=_C["purple"], fontweight="bold")

    _box(ax, 0.3, 7.8, 2.6, 0.85, "main.py\nCLI entry", _C["purple"], fontsize=7)
    _box(ax, 3.6, 7.8, 2.6, 0.85, "wf.py\ncompat wrapper", _C["purple"], fontsize=7)
    _box(ax, 6.9, 7.8, 3.2, 0.85, "master.py\nworkflow router\n& env loader", _C["purple"], fontsize=7)
    _box(ax, 10.8, 7.8, 2.6, 0.85, "agent.md\nMCP agent\ndescriptor", _C["dim"], fontsize=7)

    # main -> master
    _arrow(ax, 1.6, 7.8, 1.6, 7.3, _C["purple"])
    ax.text(1.7, 7.5, "calls", fontsize=5.5, color=_C["dim"])
    # wf -> master
    _arrow(ax, 4.9, 7.8, 7.5, 7.3, _C["purple"])
    # main -> wf (skip, both go to master)

    # ── SERVER LAYER (middle) ───────────────────────────────────────
    ax.text(7.0, 7.1, "SERVER  LAYER", fontsize=10, ha="center",
            color=_C["cyan"], fontweight="bold")

    _box(ax, 3.0, 5.6, 7.8, 1.2, "", _C["card"], alpha=0.5)
    ax.text(6.9, 6.55, "server.py  —  FastMCP  (SSE :8000)", fontsize=8,
            ha="center", color=_C["cyan"], fontweight="bold")

    # MCP tool boxes inside server
    tools = [
        "generate_case_fasta",
        "inspect_case",
        "generate_peptide_fasta",
        "generate_acid_fasta",
    ]
    for i, t in enumerate(tools):
        tx = 3.3 + i * 1.9
        _box(ax, tx, 5.75, 1.7, 0.55, t, _C["card"], fontsize=5.5, text_color=_C["cyan"])

    # master -> server
    _arrow(ax, 8.5, 7.8, 6.9, 6.8, _C["cyan"])

    # ── WORKFLOW ENGINE (lower-middle) ──────────────────────────────
    ax.text(7.0, 5.15, "WORKFLOW  ENGINE", fontsize=10, ha="center",
            color=_C["green"], fontweight="bold")

    _box(ax, 1.5, 3.5, 4.5, 1.3,
         "process_master_query()\n\n· query expansion (×5)\n"
         "· workflow routing\n· token split + embed\n"
         "· category scoring",
         _C["card"], fontsize=6, text_color=_C["green"])

    _box(ax, 7.0, 3.5, 5.2, 1.3,
         "UniProtKB  (uniprot_kb.py)\n\n"
         "18-phase enrichment pipeline\n"
         "graph build · embed · export",
         _C["card"], fontsize=6, text_color=_C["green"])

    _arrow(ax, 6.9, 5.6, 3.75, 4.8, _C["green"])
    _arrow(ax, 6.0, 4.15, 7.0, 4.15, _C["green"])

    # ── DATA / OUTPUT LAYER (bottom) ────────────────────────────────
    ax.text(7.0, 2.9, "DATA  &  OUTPUT", fontsize=10, ha="center",
            color=_C["orange"], fontweight="bold")

    _box(ax, 0.3, 1.3, 3.2, 1.2,
         "firegraph\nRuntime KG\n(NetworkX + GUtils)",
         _C["card"], fontsize=6.5, text_color=_C["orange"])

    _box(ax, 4.0, 1.3, 2.5, 1.2,
         "Gemini\nEmbeddings\n(nodes + edges)",
         _C["card"], fontsize=6.5, text_color=_C["orange"])

    _box(ax, 7.0, 1.3, 3.0, 1.2,
         "FASTA + JSON\nartifacts\ndata/<wf>/<case>/",
         _C["card"], fontsize=6.5, text_color=_C["orange"])

    _box(ax, 10.5, 1.3, 3.0, 1.2,
         "execution_cfg.py\nAPI specs per\ncategory slug",
         _C["card"], fontsize=6.5, text_color=_C["dim"])

    # connections from workflow engine to data layer
    _arrow(ax, 3.75, 3.5, 1.9, 2.5, _C["orange"])
    _arrow(ax, 9.6, 3.5, 5.25, 2.5, _C["orange"])
    _arrow(ax, 9.6, 3.5, 8.5, 2.5, _C["orange"])
    _arrow(ax, 9.6, 3.5, 12.0, 2.5, _C["dim"])

    # ── EXTERNAL APIS (right sidebar) ───────────────────────────────
    apis = [
        "UniProt REST", "Ensembl", "Reactome", "ChEMBL",
        "PubChem", "ChEBI", "GtoPdb", "ClinPGx",
        "VMH", "HPA", "AlphaFold", "InterPro",
        "QuickGO", "COMPARTMENTS", "Open Targets",
        "Open Food Facts", "RDKit / PySCF",
    ]
    ax.text(13.5, 5.15, "EXT.\nAPIs",
            fontsize=7, ha="center", color=_C["dim"], fontweight="bold")
    for i, api in enumerate(apis):
        ay = 4.85 - i * 0.27
        ax.text(13.5, ay, api, fontsize=4.8, ha="center",
                color=_C["yellow"], fontstyle="italic")

    _arrow(ax, 12.2, 4.15, 13.0, 4.15, _C["yellow"])

    fig.tight_layout()
    fig.savefig(_OUT / "system_architecture.png", dpi=180, facecolor=_C["bg"])
    plt.close(fig)
    print(f"Saved -> {_OUT / 'system_architecture.png'}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DIAGRAM 4 — Live firegraph snapshot (runtime data)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# TYPE -> LAYER mapping (reverse of _NODE_LAYERS)
_TYPE_TO_LAYER: dict[str, str] = {}
for _layer, _info in _NODE_LAYERS.items():
    for _t in _info["types"]:
        _TYPE_TO_LAYER[_t] = _layer


def render_live_graph(g_utils, out_path: Path | None = None) -> Path:
    """
    Render the populated firegraph in the same visual style as
    graph_layers.png — but with REAL node counts, top labels, and
    edge-relation statistics from the runtime knowledge graph.

    Returns the saved PNG path.
    """
    G = g_utils.G
    out = out_path or (_OUT / "live_graph.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── BUCKET NODES BY LAYER ────────────────────────────────────────
    layer_nodes: dict[str, dict[str, list[str]]] = {
        layer: {t: [] for t in info["types"]}
        for layer, info in _NODE_LAYERS.items()
    }
    uncategorised = 0

    for nid, ndata in G.nodes(data=True):
        ntype = ndata.get("type", "")
        layer = _TYPE_TO_LAYER.get(ntype)
        if layer and ntype in layer_nodes[layer]:
            label = ndata.get("label", str(nid))
            layer_nodes[layer][ntype].append(label[:40])
        else:
            uncategorised += 1

    # ── BUCKET EDGES BY RELATION ─────────────────────────────────────
    edge_counts: dict[str, int] = {}
    for _, _, edata in G.edges(data=True):
        rel = edata.get("rel", "UNKNOWN")
        edge_counts[rel] = edge_counts.get(rel, 0) + 1
    top_edges = sorted(edge_counts.items(), key=lambda x: -x[1])[:12]

    # ── DRAW ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 14), facecolor=_C["bg"])
    _style_ax(ax, f"Firegraph — Live Snapshot  ({G.number_of_nodes()} nodes · {G.number_of_edges()} edges)")
    ax.set_xlim(-0.5, 18)
    ax.set_ylim(-0.5, 14)

    col_x = [0.3, 4.6, 8.9, 13.2]
    start_y = 12.5
    row_h = 5.8
    layers = list(_NODE_LAYERS.items())

    for idx, (layer_name, info) in enumerate(layers):
        col = idx % 4
        row = idx // 4
        x = col_x[col]
        y = start_y - row * row_h

        # LAYER HEADER
        ax.text(x + 1.8, y + 0.2, layer_name,
                fontsize=10, fontweight="bold", color=info["color"], ha="center")

        y_cursor = y - 0.4
        for ntype in info["types"]:
            labels = layer_nodes[layer_name][ntype]
            count = len(labels)
            # TYPE BOX: name + count
            _box(ax, x, y_cursor, 3.6, 0.36,
                 f"{ntype}  ({count})", _C["card"], fontsize=6.5,
                 text_color=info["color"])
            y_cursor -= 0.42

            # TOP 3 LABELS as dim sub-items
            for lbl in labels[:3]:
                ax.text(x + 0.15, y_cursor + 0.08, f"· {lbl}",
                        fontsize=4.5, color=_C["dim"], va="center",
                        clip_on=True)
                y_cursor -= 0.26

            if count > 3:
                ax.text(x + 0.15, y_cursor + 0.08, f"  … +{count - 3} more",
                        fontsize=4.2, color=_C["dim"], va="center",
                        fontstyle="italic")
                y_cursor -= 0.26

    # ── EDGE LEGEND (bottom strip) ───────────────────────────────────
    edge_y = 1.2
    ax.text(9.0, edge_y + 0.5, "TOP EDGE RELATIONS", fontsize=10,
            color=_C["dim"], fontweight="bold", ha="center")

    # COLOUR-CODE edges by known palette, fallback to dim
    _EDGE_COLOR_HINTS = {
        "ENCODED_BY": _C["blue"], "INVOLVED_IN_CHAIN": _C["green"],
        "MODULATES_TARGET": _C["green"], "HAS_DOMAIN": _C["cyan"],
        "ANNOTATED_WITH": _C["orange"], "EXPRESSED_IN": _C["purple"],
        "VALIDATED_BY": _C["orange"], "CONTAINS_NUTRIENT": _C["yellow"],
        "DENSITY_COMPUTED": _C["pink"], "REQUIRES_MINERAL": _C["green"],
        "IN_PATHWAY": _C["blue"], "CLINICAL_SIG": _C["red"],
    }

    cols_per_row = 4
    for i, (rel, cnt) in enumerate(top_edges):
        cx = 0.5 + (i % cols_per_row) * 4.5
        cy = edge_y - (i // cols_per_row) * 0.4
        ec = _EDGE_COLOR_HINTS.get(rel, _C["dim"])
        ax.plot(cx, cy, "s", color=ec, markersize=7)
        ax.text(cx + 0.3, cy, f"{rel}  ({cnt})", fontsize=6,
                color=_C["text"], va="center")

    if uncategorised:
        ax.text(18.0, -0.2, f"{uncategorised} uncategorised nodes",
                fontsize=5, color=_C["dim"], ha="right", va="bottom")

    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=_C["bg"])
    plt.close(fig)
    print(f"VISUAL  Saved -> {out}")
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BRAIN SCAN ENERGY PIPELINE — Region-to-Cell-Type bridge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Maps Allen Brain Atlas region names to cell-type labels that Phase 10
# (HPA enrichment) populates in the knowledge graph.
_REGION_CELL_MAP: dict[str, list[str]] = {
    "hippocampus":      ["Pyramidal cells", "Interneurons", "Granule cells"],
    "CA1":              ["Pyramidal cells", "Interneurons"],
    "CA3":              ["Pyramidal cells", "Interneurons"],
    "cortex":           ["Pyramidal cells", "Interneurons", "Astrocytes", "Oligodendrocytes"],
    "prefrontal":       ["Pyramidal cells", "Interneurons", "Astrocytes"],
    "occipital":        ["Pyramidal cells", "Interneurons"],
    "temporal":         ["Pyramidal cells", "Interneurons"],
    "parietal":         ["Pyramidal cells", "Interneurons"],
    "cerebellum":       ["Purkinje cells", "Granule cells", "Basket cells"],
    "thalamus":         ["Relay neurons", "Interneurons"],
    "hypothalamus":     ["Neuroendocrine cells", "Interneurons"],
    "striatum":         ["Medium spiny neurons", "Cholinergic interneurons"],
    "amygdala":         ["Pyramidal cells", "Interneurons"],
    "substantia nigra": ["Dopaminergic neurons"],
    "basal ganglia":    ["Medium spiny neurons", "Cholinergic interneurons"],
    "cingulate":        ["Pyramidal cells", "Interneurons"],
    "insula":           ["Von Economo neurons", "Pyramidal cells"],
    "brainstem":        ["Motor neurons", "Interneurons"],
    "pons":             ["Motor neurons", "Interneurons"],
    "medulla":          ["Motor neurons", "Interneurons"],
    "midbrain":         ["Dopaminergic neurons", "GABAergic neurons"],
    "white matter":     ["Oligodendrocytes", "Astrocytes"],
    "corpus callosum":  ["Oligodendrocytes"],
}


# ── MODALITY TRANSFER FUNCTIONS ──────────────────────────────────────
# Each maps raw intensities (np.ndarray) to normalised [0, 1000] scale
# representing relative millivolt-equivalent energy proxy.

def _transfer_pet(intensities: np.ndarray) -> np.ndarray:
    """PET (FDG): glucose metabolism ~ ATP ~ electron flux."""
    mn, mx = intensities.min(), intensities.max()
    if mx == mn:
        return np.full_like(intensities, 500, dtype=float)
    return (intensities - mn) / (mx - mn) * 1000


def _transfer_spect(intensities: np.ndarray) -> np.ndarray:
    """SPECT: perfusion-weighted, normalise by global max."""
    mx = intensities.max()
    if mx == 0:
        return np.zeros_like(intensities, dtype=float)
    return intensities / mx * 1000


def _transfer_fmri(intensities: np.ndarray) -> np.ndarray:
    """fMRI BOLD: z-score clipped to [-3, +3] then scaled to [0, 1000]."""
    mu, sigma = intensities.mean(), intensities.std()
    if sigma == 0:
        return np.full_like(intensities, 500, dtype=float)
    z = np.clip((intensities - mu) / sigma, -3.0, 3.0)
    return (z + 3.0) / 6.0 * 1000


def _transfer_mri(intensities: np.ndarray) -> np.ndarray:
    """MRI T1/T2: tissue contrast, linear min-max to [0, 1000]."""
    mn, mx = intensities.min(), intensities.max()
    if mx == mn:
        return np.full_like(intensities, 500, dtype=float)
    return (intensities - mn) / (mx - mn) * 1000


_MODALITY_TRANSFER: dict[str, Any] = {
    "pet":   _transfer_pet,
    "spect": _transfer_spect,
    "fmri":  _transfer_fmri,
    "mri":   _transfer_mri,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BRAIN SCAN INTEGRATOR
#
#  Transforms neuroimaging data into a position-keyed energy stimulation
#  protocol by bridging physical measurements to biological entities
#  through the firegraph knowledge graph.
#
#  Input:  brain scan (any modality) + skill regulation config
#  Output: dict[pos_str, missing_energy_int]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BrainScanIntegrator:

    _EMBED_MODEL = "models/text-embedding-004"
    _EMBED_BATCH = 100
    _SIMILARITY_THRESHOLD = 0.9
    _ALLEN_BASE = "http://api.brain-map.org/api/v2"

    # ── 5-STEP WALK LAYER ORDER ──────────────────────────────────────
    # CELL_TYPE -> GENE -> PROTEIN -> ATOMIC_STRUCTURE -> EXCITATION_FREQUENCY
    #                              -> ELECTRICAL_COMPONENT

    def __init__(self, g_utils):
        self.g = g_utils
        self.client = httpx.AsyncClient(timeout=60.0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  PUBLIC ENTRY POINT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def process_brain_scan(
        self,
        scan_data,
        update_cfg: dict[str, int],
        modality: str = "pet",
    ) -> dict[str, int]:
        """
        SINGLE ENTRY POINT: Scan -> Energy Stimulation Protocol.

        Args
        ----
        scan_data : str | Path | dict
            File path to NIfTI / DICOM, or raw ``{(x,y,z): intensity}`` dict.
        update_cfg : dict[str, int]
            Skill regulation targets, e.g. ``{"focus": 80, "memory": -20}``.
            Values in [-100, +100] percent.
        modality : str
            One of ``pet``, ``spect``, ``fmri``, ``mri``.

        Returns
        -------
        dict[str, int]
            ``{"x,y,z": missing_energy}`` for every position whose
            functional ontology matched a requested skill above the
            similarity threshold.
        """
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

        # ── A: SMART MEASUREMENT INGESTION ───────────────────────────
        print("BRAIN-SCAN [A] Smart Measurement Ingestion ...")
        unified = self._to_unified_format(scan_data)
        print(f"  {len(unified)} voxels loaded")

        # ── B: ENERGETIC PATTERN + ZERO STATE ────────────────────────
        print("BRAIN-SCAN [B] Energetic Pattern + Zero State ...")
        energetic_pattern = self._map_to_electron_matrix(unified, modality)
        zero_state = self._compute_zero_state(unified)

        # ── C: SPATIAL ONTOLOGY (Allen Brain Atlas + KG bridge) ──────
        print("BRAIN-SCAN [C] Spatial Ontology Mapping ...")
        spatial_map = await self._query_spatial_ontology(list(unified.keys()))
        unique_regions = len({
            v["region"] for v in spatial_map.values() if v.get("region")
        })
        print(f"  {unique_regions} unique brain regions resolved")

        # ── D: 5-STEP GRAPH WALK per position ────────────────────────
        print("BRAIN-SCAN [D] 5-Step Graph Walk ...")
        enriched_map: dict[str, dict] = {}
        pathway_summaries: dict[str, str] = {}
        for pos_key, components in spatial_map.items():
            walk = self._walk_graph_5steps(components)
            enriched_map[pos_key] = walk
            pathway_summaries[pos_key] = walk.get("summary", "")

        # ── E: CONTEXT EMBEDDINGS (Gemini batch) ─────────────────────
        print("BRAIN-SCAN [E] Context Embeddings ...")
        embeddings = await self._generate_context_embeddings(pathway_summaries)
        print(f"  {len(embeddings)} position embeddings generated")

        # ── F: FREQUENCY ONTOLOGY ────────────────────────────────────
        print("BRAIN-SCAN [F] Frequency Ontology ...")
        frequency_ontology = self._build_frequency_ontology(enriched_map)
        print(f"  {len(frequency_ontology)} positions with frequency data")

        # ── G: FUNCTIONAL ONTOLOGY ───────────────────────────────────
        print("BRAIN-SCAN [G] Functional Ontology ...")
        functional_ontology = self._build_functional_ontology(enriched_map)
        print(f"  {len(functional_ontology)} positions with functional labels")

        # ── H: SKILL ALIGNMENT (cosine similarity) ───────────────────
        print("BRAIN-SCAN [H] Skill Alignment ...")
        aligned = await self._llm_align_skills(
            functional_ontology, energetic_pattern, embeddings, update_cfg,
        )
        print(
            f"  {len(aligned)} positions matched "
            f"(threshold {self._SIMILARITY_THRESHOLD})"
        )

        # ── I: ENERGY DELTA ──────────────────────────────────────────
        print("BRAIN-SCAN [I] Energy Delta Calculation ...")
        result = self._calc_missing_energy(zero_state, energetic_pattern, aligned)
        active = sum(1 for v in result.values() if v > 0)
        print(f"  {active}/{len(result)} positions require energy input")

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STEP A: SMART MEASUREMENT INGESTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _to_unified_format(
        self, scan_data
    ) -> dict[tuple[int, int, int], float]:
        """
        Accept any neuroimaging input and return a uniform
        ``{(x, y, z): intensity}`` dict in MNI152 standard space.
        """
        if isinstance(scan_data, dict):
            return self._load_raw_dict(scan_data)

        path = Path(str(scan_data))
        suffix = "".join(path.suffixes).lower()

        if suffix in (".nii", ".nii.gz"):
            return self._load_nifti(path)
        if suffix == ".dcm" or path.is_dir():
            return self._load_dicom_series(path)

        raise ValueError(f"Unsupported scan format: {suffix}")

    def _load_nifti(
        self, path: Path
    ) -> dict[tuple[int, int, int], float]:
        """NIfTI (.nii / .nii.gz) -> MNI152 coordinate dict."""
        import nibabel as nib

        img = nib.as_closest_canonical(nib.load(str(path)))
        data = np.asarray(img.dataobj, dtype=np.float64)
        affine = img.affine

        result: dict[tuple[int, int, int], float] = {}
        nonzero = np.argwhere(data > 0)
        for ijk in nonzero:
            mni = affine @ np.append(ijk, 1.0)
            pos = (int(round(mni[0])), int(round(mni[1])), int(round(mni[2])))
            result[pos] = float(data[tuple(ijk)])
        return result

    def _load_dicom_series(
        self, path: Path
    ) -> dict[tuple[int, int, int], float]:
        """DICOM directory -> canonical volume -> MNI coordinate dict."""
        import nibabel as nib

        img = nib.as_closest_canonical(nib.load(str(path)))
        data = np.asarray(img.dataobj, dtype=np.float64)
        affine = img.affine

        result: dict[tuple[int, int, int], float] = {}
        nonzero = np.argwhere(data > 0)
        for ijk in nonzero:
            mni = affine @ np.append(ijk, 1.0)
            pos = (int(round(mni[0])), int(round(mni[1])), int(round(mni[2])))
            result[pos] = float(data[tuple(ijk)])
        return result

    @staticmethod
    def _load_raw_dict(data: dict) -> dict[tuple[int, int, int], float]:
        """Validate raw ``{(x,y,z): intensity}`` or ``{"x,y,z": intensity}``."""
        result: dict[tuple[int, int, int], float] = {}
        for key, val in data.items():
            if isinstance(key, (tuple, list)) and len(key) == 3:
                result[(int(key[0]), int(key[1]), int(key[2]))] = float(val)
            elif isinstance(key, str) and "," in key:
                parts = [int(p.strip()) for p in key.split(",")]
                result[(parts[0], parts[1], parts[2])] = float(val)
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STEP B: ENERGETIC PATTERN + ZERO STATE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _map_to_electron_matrix(
        unified: dict[tuple, float], modality: str,
    ) -> dict[str, int]:
        """
        Apply modality-specific transfer function to produce
        ``ENERGETIC_PATTERN = {pos_str: energy_int}``.
        """
        transfer_fn = _MODALITY_TRANSFER.get(modality.lower(), _transfer_pet)

        positions = list(unified.keys())
        intensities = np.array(
            [unified[p] for p in positions], dtype=np.float64,
        )
        scaled = transfer_fn(intensities)

        return {
            f"{p[0]},{p[1]},{p[2]}": int(round(s))
            for p, s in zip(positions, scaled)
        }

    @staticmethod
    def _compute_zero_state(
        unified: dict[tuple, float],
    ) -> dict[str, int]:
        """
        Baseline homeostasis (zero brain state).
        Uses the global median intensity as the resting-state equilibrium
        for every position.
        """
        vals = np.array(list(unified.values()), dtype=np.float64)
        baseline = int(round(float(np.median(vals))))
        return {
            f"{p[0]},{p[1]},{p[2]}": baseline
            for p in unified
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STEP C: SPATIAL ONTOLOGY MAPPING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _query_spatial_ontology(
        self, positions: list[tuple[int, int, int]],
    ) -> dict[str, dict]:
        """
        Map MNI coordinates -> brain regions (Allen Brain Atlas)
        -> KG entities (CELL_TYPE / GENE / PROTEIN).
        Deduplicates by coarse 2 mm bins to minimise API calls.
        """
        region_cache: dict[tuple, str] = {}

        # GROUP by coarse grid (2 mm bins)
        coarse_bins: dict[tuple, list[tuple]] = {}
        for pos in positions:
            coarse = (pos[0] // 2 * 2, pos[1] // 2 * 2, pos[2] // 2 * 2)
            coarse_bins.setdefault(coarse, []).append(pos)

        # RESOLVE each coarse bin to a brain region
        pos_to_region: dict[str, str] = {}
        for coarse, fine_list in coarse_bins.items():
            if coarse not in region_cache:
                region_cache[coarse] = await self._lookup_allen_region(coarse)
            region = region_cache[coarse]
            for pos in fine_list:
                pos_to_region[f"{pos[0]},{pos[1]},{pos[2]}"] = region

        # BRIDGE each region to KG entities (cached per region)
        kg_cache: dict[str, dict] = {}
        result: dict[str, dict] = {}

        for pos_key, region in pos_to_region.items():
            if region not in kg_cache:
                kg_cache[region] = self._bridge_region_to_kg(region)
            comp = kg_cache[region]
            result[pos_key] = {
                "region":        region,
                "cell_types":    comp.get("cell_types", []),
                "genes":         comp.get("genes", []),
                "proteins":      comp.get("proteins", []),
                "cell_node_ids": comp.get("cell_node_ids", []),
            }

        return result

    async def _lookup_allen_region(
        self, pos: tuple[int, int, int],
    ) -> str:
        """Query Allen Brain Atlas CCF for the structure at an MNI coordinate."""
        try:
            url = (
                f"{self._ALLEN_BASE}/structure_lookup.json"
                f"?mni_coord={pos[0]},{pos[1]},{pos[2]}"
            )
            resp = await self.client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                msg = data.get("msg")
                if isinstance(msg, list) and msg:
                    return msg[0].get("name", "unknown")
                if isinstance(msg, dict):
                    return msg.get("name", "unknown")
        except Exception:
            pass
        return "unknown"

    def _bridge_region_to_kg(self, region: str) -> dict:
        """
        Map a brain region name to KG entities.
        Uses ``_REGION_CELL_MAP`` to find candidate cell-type labels,
        then walks the graph backwards to collect genes and proteins.
        """
        # MATCH region name fragments to known cell types
        cell_labels: list[str] = []
        region_lower = region.lower()
        for pattern, cells in _REGION_CELL_MAP.items():
            if pattern.lower() in region_lower:
                cell_labels.extend(cells)

        if not cell_labels:
            for pattern, cells in _REGION_CELL_MAP.items():
                for word in region_lower.split():
                    if word in pattern.lower():
                        cell_labels.extend(cells)
                        break
        cell_labels = list(set(cell_labels))

        # SEARCH graph for matching CELL_TYPE nodes
        G = self.g.G
        cell_node_ids: list[str] = []
        genes: list[str] = []
        proteins: list[str] = []

        for nid, ndata in G.nodes(data=True):
            if ndata.get("type") != "CELL_TYPE":
                continue
            node_label = (ndata.get("label") or "").lower()
            for cl in cell_labels:
                if cl.lower() in node_label or node_label in cl.lower():
                    cell_node_ids.append(nid)
                    break

        # REVERSE WALK: cell -> gene -> protein
        for cell_id in cell_node_ids:
            for neighbor in G.neighbors(cell_id):
                nb = G.nodes[neighbor]
                if nb.get("type") == "GENE":
                    genes.append(neighbor)
                    for nn in G.neighbors(neighbor):
                        if G.nodes[nn].get("type") == "PROTEIN":
                            proteins.append(nn)

        return {
            "cell_types":    cell_labels,
            "cell_node_ids": list(set(cell_node_ids)),
            "genes":         list(set(genes)),
            "proteins":      list(set(proteins)),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STEP D: 5-STEP GRAPH WALK
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _walk_graph_5steps(self, components: dict) -> dict:
        """
        Walk 5 hops through the KG from spatial-ontology components:
          CELL_TYPE -> GENE -> PROTEIN -> ATOMIC_STRUCTURE -> EXCITATION_FREQUENCY
                                       -> ELECTRICAL_COMPONENT
        Collects all reached nodes and builds a text summary for embedding.
        """
        collected: dict[str, list[dict]] = {
            "cells": [], "genes": [], "proteins": [],
            "structures": [], "frequencies": [], "electricals": [],
        }
        G = self.g.G

        # Steps 1-2 already resolved in spatial ontology
        for cell_id in components.get("cell_node_ids", []):
            cd = G.nodes.get(cell_id, {})
            if cd:
                collected["cells"].append({"id": cell_id, **cd})

        for gene_id in components.get("genes", []):
            gd = G.nodes.get(gene_id, {})
            if gd:
                collected["genes"].append({"id": gene_id, **gd})

        # Steps 3-5: from each protein, reach atomic + excitation + electrical
        for prot_id in components.get("proteins", []):
            pd = G.nodes.get(prot_id, {})
            if pd:
                collected["proteins"].append({"id": prot_id, **pd})

            for neighbor in G.neighbors(prot_id):
                nd = G.nodes[neighbor]
                ntype = nd.get("type", "")

                # STEP 3: PROTEIN -> ATOMIC_STRUCTURE / MOLECULE_CHAIN
                if ntype in ("ATOMIC_STRUCTURE", "MOLECULE_CHAIN"):
                    collected["structures"].append({"id": neighbor, **nd})
                    # STEP 4: ATOMIC -> EXCITATION_FREQUENCY
                    for deep_nb in G.neighbors(neighbor):
                        dnd = G.nodes[deep_nb]
                        if dnd.get("type") == "EXCITATION_FREQUENCY":
                            collected["frequencies"].append(
                                {"id": deep_nb, **dnd}
                            )

                # STEP 5: PROTEIN -> ELECTRICAL_COMPONENT
                elif ntype == "ELECTRICAL_COMPONENT":
                    collected["electricals"].append({"id": neighbor, **nd})

        # BUILD TEXT SUMMARY (compact, for embedding)
        parts: list[str] = []
        for c in collected["cells"][:3]:
            parts.append(f"{c.get('label', c['id'])} (Cell)")
        for g in collected["genes"][:3]:
            parts.append(f"{g.get('label', g['id'])} (Gene)")
        for p in collected["proteins"][:3]:
            parts.append(f"{p.get('label', p['id'])} (Protein)")
        for s in collected["structures"][:2]:
            sm = (s.get("smiles") or "")[:30]
            parts.append(f"{s.get('label', s['id'])} SMILES={sm} (Atomic)")
        for f in collected["frequencies"][:2]:
            parts.append(
                f"{f.get('frequency_hz', 0):.2e}Hz "
                f"\u03bb={f.get('wavelength_nm', 0):.0f}nm (Excitation)"
            )
        for e in collected["electricals"][:2]:
            parts.append(
                f"g={e.get('conductance_pS', '?')}pS "
                f"V\u00bd={e.get('v_half_activation', '?')}mV (Electrical)"
            )

        return {
            "collected": collected,
            "summary": " -> ".join(parts) if parts else "no KG path",
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STEP E: CONTEXT EMBEDDINGS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _generate_context_embeddings(
        self, pathway_summaries: dict[str, str],
    ) -> dict[str, list[float]]:
        """Batch-embed all pathway summary strings via Gemini."""
        positions = list(pathway_summaries.keys())
        texts = [pathway_summaries[p] for p in positions]

        valid_idx = [i for i, t in enumerate(texts) if t and t != "no KG path"]
        valid_texts = [texts[i] for i in valid_idx]
        if not valid_texts:
            return {}

        all_vectors = await self._batch_embed(valid_texts)
        return {
            positions[idx]: vec
            for idx, vec in zip(valid_idx, all_vectors)
        }

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Gemini batch embedding (chunked to stay under API limits)."""
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._EMBED_BATCH):
            chunk = texts[i : i + self._EMBED_BATCH]
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self._EMBED_MODEL,
                content=chunk,
                task_type="RETRIEVAL_DOCUMENT",
            )
            all_vectors.extend(result["embedding"])
        return all_vectors

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STEP F: FREQUENCY ONTOLOGY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _build_frequency_ontology(
        enriched_map: dict[str, dict],
    ) -> dict[str, dict]:
        """
        Aggregate EXCITATION_FREQUENCY + ELECTRICAL_COMPONENT data
        per position into FREQUENCY_ONTOLOGY.
        """
        result: dict[str, dict] = {}

        for pos_key, walk in enriched_map.items():
            collected = walk.get("collected", {})
            freqs = collected.get("frequencies", [])
            electricals = collected.get("electricals", [])
            if not freqs and not electricals:
                continue

            # DOMINANT FREQUENCY: oscillator-strength-weighted average
            spectrum: list[dict] = []
            total_weight = 0.0
            weighted_freq = 0.0
            nir_targetable = False

            for f in freqs:
                fhz = f.get("frequency_hz", 0)
                osc = f.get("oscillator_strength", 0)
                wl = f.get("wavelength_nm", 0)
                nir = f.get("in_nir_window", False)
                if nir:
                    nir_targetable = True
                spectrum.append({
                    "frequency_hz": fhz,
                    "wavelength_nm": wl,
                    "oscillator_strength": osc,
                    "in_nir_window": nir,
                })
                if osc > 0:
                    weighted_freq += fhz * osc
                    total_weight += osc

            dominant = weighted_freq / total_weight if total_weight > 0 else 0.0

            # ION CHANNEL RESONANCE from electrical components (Phase 8)
            conductance = None
            v_half = None
            for e in electricals:
                if e.get("conductance_pS") is not None:
                    conductance = e["conductance_pS"]
                if e.get("v_half_activation") is not None:
                    v_half = e["v_half_activation"]

            result[pos_key] = {
                "dominant_freq_hz":   dominant,
                "nir_targetable":     nir_targetable,
                "spectrum":           spectrum,
                "conductance_pS":     conductance,
                "v_half_activation":  v_half,
            }

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STEP G: FUNCTIONAL ONTOLOGY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_functional_ontology(
        self, enriched_map: dict[str, dict],
    ) -> dict[str, str]:
        """
        Derive a functional task label per position from GO terms
        reachable through the KG proteins at that location.
        Prefers biological_process (BP); picks the most specific label.
        """
        G = self.g.G
        result: dict[str, str] = {}

        for pos_key, walk in enriched_map.items():
            proteins = walk.get("collected", {}).get("proteins", [])
            if not proteins:
                continue

            go_terms: list[dict] = []
            for prot in proteins:
                prot_id = prot.get("id", "")
                if not G.has_node(prot_id):
                    continue
                for neighbor in G.neighbors(prot_id):
                    nd = G.nodes[neighbor]
                    if nd.get("type") == "GO_TERM":
                        go_terms.append({
                            "id":         neighbor,
                            "label":      nd.get("label", ""),
                            "aspect":     nd.get("aspect", ""),
                            "definition": nd.get("definition", ""),
                        })

            if not go_terms:
                # FALLBACK: protein name as functional label
                result[pos_key] = proteins[0].get("label", "unknown function")
                continue

            # PREFER biological_process terms
            bp = [t for t in go_terms if t["aspect"] == "biological_process"]
            pool = bp if bp else go_terms

            # HEURISTIC for specificity: longest label = most specific term
            best = max(pool, key=lambda t: len(t.get("label", "")))
            result[pos_key] = best.get("label") or best["id"]

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STEP H: LLM SKILL ALIGNMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _llm_align_skills(
        self,
        functional_ontology: dict[str, str],
        energetic_pattern: dict[str, int],
        embeddings: dict[str, list[float]],
        update_cfg: dict[str, int],
    ) -> dict[str, dict]:
        """
        Match positions to skills via cosine similarity on embeddings.

        Returns ``{pos: {"skill", "regulation_pct", "similarity"}}``.
        """
        if not update_cfg or not embeddings:
            return {}

        # EMBED each skill name with the same model
        skill_names = list(update_cfg.keys())
        skill_vectors = await self._batch_embed(skill_names)
        skill_embeddings = dict(zip(skill_names, skill_vectors))

        aligned: dict[str, dict] = {}
        for pos_key, pos_vec in embeddings.items():
            best_skill: str | None = None
            best_sim = 0.0

            for skill, skill_vec in skill_embeddings.items():
                sim = self._cosine_sim(pos_vec, skill_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_skill = skill

            if best_sim >= self._SIMILARITY_THRESHOLD and best_skill:
                aligned[pos_key] = {
                    "skill":          best_skill,
                    "regulation_pct": update_cfg[best_skill],
                    "similarity":     round(best_sim, 4),
                }

        return aligned

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  STEP I: ENERGY DELTA CALCULATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _calc_missing_energy(
        zero_state: dict[str, int],
        energetic_pattern: dict[str, int],
        aligned: dict[str, dict],
    ) -> dict[str, int]:
        """
        Compute energy delta for every aligned position.

        ``Missing_Energy = Phi_Target - Phi_Current``
        where ``Phi_Target = Phi_Zero * (1 + regulation_pct / 100)``.

        Only up-regulation (positive %) yields non-zero output;
        down-regulation positions get 0 (no external energy needed).
        """
        output: dict[str, int] = {}
        for pos, match in aligned.items():
            current_e = energetic_pattern.get(pos, 0)
            zero_e = zero_state.get(pos, 0)
            reg = match["regulation_pct"]

            if reg > 0:
                target_e = zero_e * (1 + reg / 100)
                output[pos] = max(0, int(round(target_e - current_e)))
            else:
                output[pos] = 0

        return output

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def close(self):
        await self.client.aclose()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    _OUT.mkdir(parents=True, exist_ok=True)
    render_workflow_phases()
    render_graph_layers()
    render_system_architecture()
    print(f"\nAll diagrams written to {_OUT.resolve()}/")
