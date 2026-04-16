"""
designer.py — graph-driven substrate design + organ/route formulation intents.

Plan / user prompt (primary spec):
    designer.py — create a Designer class that takes the pipeline graph (``GUtils.G`` or the
    same ``networkx`` object the ctlr tissue filter uses), extracts nodes grouped by ``type``,
    uses molecule-specific graph context to combine components into a working-substrate blueprint,
    and applies caller ``result_specs`` to attach organ-targeted lipid / LNP-style formulation
    strategy (intramuscular, intravenous, oral uptake framing) to proteins. Golden-standard /
    pharmacology cues enter only via caller ``cited_guidance`` and route-driven checklist keys —
    no fabricated batch or clinical data. Emit per-node-type JSON artifacts + manifest for
    downstream synthesis / procurement alignment (e.g. ``shop.py`` absorption mapping).

User prompt (artifact shaping — GENE / TISSUE / CELL_TYPE):
    Adapt cell, gene and tissue nodes to just capture id and most important information AND
    link to ontology and neighbor physical compounds (graph-evidence only; compact neighbor refs).

CHAR: formulation labels are strategy enums only; excipient identities come from the graph or
from caller ``allowed_excipient_classes``.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence, TYPE_CHECKING

import networkx as nx

from ctlr import _ONTOLOGY_LIKE_TYPES, _PHYSICAL_TYPES

if TYPE_CHECKING:
    from firegraph.graph import GUtils

ClinicalRoute = Literal["intramuscular", "intravenous", "oral"]

_SMALL_MOL_TYPES = frozenset({
    "PHARMA_COMPOUND", "MINERAL", "VMH_METABOLITE", "VITAMIN", "FATTY_ACID", "COFACTOR",
    "MOLECULE_CHAIN",
})

_ORGAN_LIKE = frozenset({"ORGAN", "TISSUE", "ANATOMY_PART"})

# CHAR: per-type JSON slices — full nx attrs are omitted to keep artifacts small and stable.
_SLIM_BIOLOGY_NODE_TYPES = frozenset({"GENE", "TISSUE", "CELL_TYPE"})


@dataclass
class OrganDeliverySpec:
    """Caller-defined organ targeting + route; no default clinical or SKU data."""

    target_organs: list[str]
    primary_route: ClinicalRoute
    allowed_excipient_classes: list[str] | None = None
    stability_storage_class: str | None = None
    cited_guidance: list[str] | None = None

    def __post_init__(self) -> None:
        toks = [str(x).strip() for x in (self.target_organs or []) if str(x).strip()]
        if not toks:
            raise ValueError("OrganDeliverySpec.target_organs must be non-empty.")
        self.target_organs = toks
        r = str(self.primary_route).strip().lower().replace(" ", "_")
        if r in ("im", "i.m.", "intramuscular"):
            self.primary_route = "intramuscular"
        elif r in ("iv", "i.v.", "intravenous"):
            self.primary_route = "intravenous"
        elif r in ("po", "oral", "per_os"):
            self.primary_route = "oral"
        else:
            raise ValueError(
                f"Invalid primary_route {self.primary_route!r}; "
                "use intramuscular, intravenous, or oral.",
            )
        if self.allowed_excipient_classes is not None:
            self.allowed_excipient_classes = [
                str(x).strip() for x in self.allowed_excipient_classes if str(x).strip()
            ]
        if self.stability_storage_class is not None:
            self.stability_storage_class = str(self.stability_storage_class).strip() or None
        if self.cited_guidance is not None:
            self.cited_guidance = [str(x).strip() for x in self.cited_guidance if str(x).strip()]


def normalize_result_specs(raw: Sequence[OrganDeliverySpec | dict[str, Any]] | None) -> list[OrganDeliverySpec]:
    """Coerce dicts / dataclass instances into validated ``OrganDeliverySpec`` rows."""
    if not raw:
        return []
    out: list[OrganDeliverySpec] = []
    for row in raw:
        if isinstance(row, OrganDeliverySpec):
            out.append(row)
            continue
        if not isinstance(row, dict):
            raise TypeError("Each result_spec must be OrganDeliverySpec or dict.")
        out.append(
            OrganDeliverySpec(
                target_organs=list(row.get("target_organs") or []),
                primary_route=row["primary_route"],  # type: ignore[arg-type]
                allowed_excipient_classes=row.get("allowed_excipient_classes"),
                stability_storage_class=row.get("stability_storage_class"),
                cited_guidance=row.get("cited_guidance"),
            )
        )
    return out


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _shop_absorption(route: ClinicalRoute) -> str:
    """CHAR: ``shop._normalize_absorption`` only knows injection vs oral."""
    if route == "oral":
        return "oral"
    return "injection"


def _route_strategy_labels(route: ClinicalRoute) -> tuple[list[str], list[str]]:
    """
    CHAR: return (lipid_or_carrier_strategy_labels, process_expectation_keys) —
    taxonomy only, no compound IDs.
    """
    if route == "intravenous":
        return (
            ["systemic_exposure_focus", "lipid_nanocarrier_or_aqueous_iv", "sterility_endotoxin_program"],
            ["sterile_manufacturing", "particulate_control", "endotoxin_spec_program"],
        )
    if route == "intramuscular":
        return (
            ["depot_release_tradeoff", "aqueous_buffer_class", "lipid_adjuvant_optional"],
            ["volume_per_site_limits", "particle_size_considerations"],
        )
    return (
        ["gi_barrier_permeability", "enzymatic_stability", "enteric_strategy_optional"],
        ["food_effect_placeholder_program", "regional_absorption_risk_review"],
    )


def _stable_graph_digest(nx_g: nx.Graph | nx.MultiGraph) -> str:
    """CHAR: order-insensitive fingerprint of node ids+types and edges+rel for manifest provenance."""
    node_lines: list[str] = []
    for nid in sorted(nx_g.nodes(), key=lambda x: str(x)):
        t = nx_g.nodes[nid].get("type", "")
        node_lines.append(f"{nid!s}\x1f{t!s}")
    edge_lines: list[str] = []
    if nx_g.is_multigraph():
        for u, v, k, ed in nx_g.edges(keys=True, data=True):
            rel = ed.get("rel", "") if isinstance(ed, dict) else ""
            edge_lines.append(f"{u!s}\x1f{v!s}\x1f{k!s}\x1f{rel!s}")
    else:
        for u, v, ed in nx_g.edges(data=True):
            rel = ed.get("rel", "") if isinstance(ed, dict) else ""
            edge_lines.append(f"{u!s}\x1f{v!s}\x1f{rel!s}")
    payload = "\n".join(node_lines) + "||" + "\n".join(sorted(edge_lines))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_filename_token(node_type: str) -> str:
    raw = re.sub(r"[^\w\-.]+", "_", str(node_type).strip())
    return raw or "UNKNOWN"


def _obo_purl_from_curie(curie: str | None) -> str | None:
    """
    CHAR: OBO PURL from node ids — supports ``CL:0000123``, ``UBERON:0000955``, or OLS-style
    ``UBERON_0000955`` / ``CL_0000123`` short forms already on graph nodes.
    """
    if not curie:
        return None
    s = str(curie).strip()
    if not s:
        return None
    if "_" in s and ":" not in s and s[:2].isalpha():
        up = s.upper()
        if up.startswith(("UBERON_", "CL_", "GO_")):
            return f"http://purl.obolibrary.org/obo/{up}"
    if ":" not in s:
        return None
    prefix, local = s.split(":", 1)
    prefix_u = prefix.strip().upper()
    local = local.strip()
    if not prefix_u or not local:
        return None
    return f"http://purl.obolibrary.org/obo/{prefix_u}_{local}"


def _biology_primary_label(attrs: dict[str, Any]) -> str | None:
    for k in ("label", "name", "gene_symbol"):
        v = attrs.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _biology_essentials(node_type: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """CHAR: type-scoped stable keys only — no bulk graph/embed payload."""
    out: dict[str, Any] = {}
    if node_type == "GENE":
        for k in ("gene_symbol", "ensembl_id", "hgnc_id", "entrez_id", "ncbi_gene_id"):
            v = attrs.get(k)
            if v is not None and str(v).strip():
                out[k] = v
    elif node_type == "TISSUE":
        for k in ("uberon_id", "ontology_prefix", "uberon_resolved"):
            if k in attrs:
                out[k] = attrs[k]
    elif node_type == "CELL_TYPE":
        for k in ("cl_id", "ontology_prefix", "cl_resolved"):
            if k in attrs:
                out[k] = attrs[k]
    return out


def _ontology_iri_for_node(node_type: str, attrs: dict[str, Any]) -> str | None:
    if node_type == "TISSUE":
        return _obo_purl_from_curie(attrs.get("uberon_id"))
    if node_type == "CELL_TYPE":
        return _obo_purl_from_curie(attrs.get("cl_id"))
    return None


def _compact_neighbor_refs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """CHAR: artifact-stable stubs — id/type/label for downstream join to full graph exports."""
    out: list[dict[str, Any]] = []
    for r in rows:
        rid = r.get("id")
        if rid is None or str(rid).strip() == "":
            continue
        out.append({
            "id": str(rid),
            "type": r.get("type"),
            "label": r.get("label"),
        })
    return out


def _prune_empty_values(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


class Designer:
    """
    Build a substrate blueprint from a tissue / ctlr-filtered graph and merge organ-route
    ``result_specs`` onto protein entities for formulation / procurement handoff.
    """

    def __init__(
        self,
        gutils: GUtils,
        result_specs: Sequence[OrganDeliverySpec | dict[str, Any]] | None = None,
        include_peptide_chains: bool = False,
        context_hops: int = 2,
        organ_match_hops: int = 4,
    ) -> None:
        #
        self._nx: nx.Graph | nx.MultiGraph = graph if graph is not None else gutils.G  # type: ignore[union-attr]
        self._result_specs = normalize_result_specs(result_specs)
        self.include_peptide_chains = bool(include_peptide_chains)
        self.context_hops = max(1, int(context_hops))
        self.organ_match_hops = max(1, int(organ_match_hops))

    def group_nodes_by_type(self) -> dict[str, list[dict[str, Any]]]:
        """CHAR: O(N) single pass — bucket nodes by ``type`` (missing → UNKNOWN)."""
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for nid, attrs in self._nx.nodes(data=True):
            nt = str(attrs.get("type") or "UNKNOWN").strip() or "UNKNOWN"
            row: dict[str, Any] = {"id": str(nid), **dict(attrs)}
            buckets[nt].append(row)
        return dict(buckets)

    def neighborhood_molecular_context(self, node_id: str) -> dict[str, Any]:
        """
        Bounded BFS: physical neighbors + ontology neighbors with ``rel`` histograms
        (graph-evidence only).
        """
        root = str(node_id)
        if root not in self._nx:
            return {"error": "unknown_node", "id": root}

        physical: dict[str, dict[str, Any]] = {}
        ontology: dict[str, dict[str, Any]] = {}
        rel_phys: Counter[str] = Counter()
        rel_onto: Counter[str] = Counter()

        q: deque[tuple[str, int]] = deque([(root, 0)])
        seen: set[str] = {root}
        while q:
            cur, d = q.popleft()
            if d >= self.context_hops:
                continue
            for nb in self._nx.neighbors(cur):
                if nb in seen:
                    continue
                seen.add(nb)
                nd = self._nx.nodes[nb]
                nt = nd.get("type")
                # CHAR: gather parallel edges on MultiGraph
                if self._nx.is_multigraph():
                    for k in self._nx[cur][nb]:
                        ed = self._nx.edges[cur, nb, k]
                        rel = str(ed.get("rel", "?"))
                        if nt in _PHYSICAL_TYPES:
                            rel_phys[rel] += 1
                        elif nt in _ONTOLOGY_LIKE_TYPES:
                            rel_onto[rel] += 1
                else:
                    ed = self._nx.edges[cur, nb]
                    rel = str(ed.get("rel", "?"))
                    if nt in _PHYSICAL_TYPES:
                        rel_phys[rel] += 1
                    elif nt in _ONTOLOGY_LIKE_TYPES:
                        rel_onto[rel] += 1

                entry = {
                    "id": str(nb),
                    "type": nt,
                    "label": nd.get("label") or nd.get("name") or nd.get("gene_symbol"),
                }
                if nt in _PHYSICAL_TYPES:
                    physical[str(nb)] = entry
                elif nt in _ONTOLOGY_LIKE_TYPES:
                    ontology[str(nb)] = entry
                q.append((nb, d + 1))

        return {
            "physical_neighbors": list(physical.values()),
            "ontology_neighbors": list(ontology.values()),
            "physical_rel_histogram": dict(rel_phys),
            "ontology_rel_histogram": dict(rel_onto),
        }

    def _slim_biology_node_item(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        CHAR: GENE / TISSUE / CELL_TYPE artifact row — essentials + bounded context:
        ontology neighbors (annotation/anatomy closure) and small-molecule physical neighbors.
        """
        nid = str(raw["id"])
        nt = str(raw.get("type") or "UNKNOWN")
        attrs = {k: v for k, v in raw.items() if k != "id"}
        ctx = self.neighborhood_molecular_context(nid)
        onto_rows = list(ctx.get("ontology_neighbors") or [])
        phys_rows = list(ctx.get("physical_neighbors") or [])
        compound_rows = [p for p in phys_rows if p.get("type") in _SMALL_MOL_TYPES]
        iri = _ontology_iri_for_node(nt, attrs)
        item: dict[str, Any] = {
            "id": nid,
            "type": nt,
            "label": _biology_primary_label(attrs),
            "essentials": _biology_essentials(nt, attrs),
            "ontology_iri": iri,
            "ontology_neighbors": _compact_neighbor_refs(onto_rows),
            "compound_neighbors": _compact_neighbor_refs(compound_rows),
            "ontology_rel_histogram": dict(ctx.get("ontology_rel_histogram") or {}),
        }
        return _prune_empty_values(item)

    def _nearby_organ_tokens(self, node_id: str) -> set[str]:
        """CHAR: collect normalized organ/tissue labels within ``organ_match_hops`` for spec matching."""
        root = str(node_id)
        if root not in self._nx:
            return set()
        tokens: set[str] = set()
        q: deque[tuple[str, int]] = deque([(root, 0)])
        seen: set[str] = {root}
        while q:
            cur, d = q.popleft()
            if d >= self.organ_match_hops:
                continue
            for nb in self._nx.neighbors(cur):
                if nb in seen:
                    continue
                seen.add(nb)
                nd = self._nx.nodes[nb]
                if nd.get("type") in _ORGAN_LIKE:
                    for key in ("label", "name", "uberon_id", "organ_label"):
                        v = nd.get(key)
                        if v:
                            tokens.add(_norm_label(v))
                    tokens.add(_norm_label(nb))
                q.append((nb, d + 1))
        return {t for t in tokens if t}

    def _spec_matches_tokens(self, tokens: set[str], spec: OrganDeliverySpec) -> bool:
        for org in spec.target_organs:
            o = _norm_label(org)
            if not o:
                continue
            if o in tokens:
                return True
            if any(o in t or t in o for t in tokens if t):
                return True
        return False

    def _collect_small_mol_for(self, node_id: str) -> list[dict[str, Any]]:
        ctx = self.neighborhood_molecular_context(node_id)
        out: list[dict[str, Any]] = []
        for row in ctx.get("physical_neighbors", []):
            if row.get("type") in _SMALL_MOL_TYPES:
                out.append(row)
        return out

    def _protein_identity(self, attrs: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "accession", "gene_symbol", "label", "protein_name", "uniprot_id",
            "protein_accession", "entry_name",
        )
        return {k: attrs[k] for k in keys if attrs.get(k) is not None}

    def build_working_substrate_blueprint(self) -> dict[str, Any]:
        """
        Typed bundles: proteins (and optional MOLECULE_CHAIN) with molecular + ontology context
        derived from the graph.
        """
        by_t = self.group_nodes_by_type()
        proteins = by_t.get("PROTEIN", [])
        chains: list[dict[str, Any]] = []
        if self.include_peptide_chains:
            chains = by_t.get("MOLECULE_CHAIN", [])

        backbone: list[dict[str, Any]] = []
        adjuvant_ids: set[str] = set()

        for row in proteins + chains:
            pid = str(row["id"])
            ctx = self.neighborhood_molecular_context(pid)
            small = self._collect_small_mol_for(pid)
            for sm in small:
                adjuvant_ids.add(str(sm["id"]))
            onto_ids = {str(x["id"]) for x in ctx.get("ontology_neighbors", [])}

            entry = {
                "node_id": pid,
                "node_type": row.get("type"),
                "identity": self._protein_identity(row),
                "molecular_context": ctx,
                "adjacent_small_molecules": small,
                "ontology_neighbor_ids": sorted(onto_ids),
            }
            backbone.append(entry)

        adjuvant_nodes: list[dict[str, Any]] = []
        buckets = self.group_nodes_by_type()
        for aid in sorted(adjuvant_ids):
            for nt, rows in buckets.items():
                if nt not in _SMALL_MOL_TYPES:
                    continue
                for r in rows:
                    if str(r["id"]) == aid:
                        adjuvant_nodes.append({"id": aid, "type": nt, **{k: v for k, v in r.items() if k != "id"}})
                        break

        return {
            "physical_backbone": {"proteins_and_chains": backbone},
            "adjuvant_like_small_molecules": adjuvant_nodes,
            "ontology_rationale": {
                "note": "ontology neighbors on each backbone row explain graph-local annotation context",
            },
        }

    def apply_result_specs_to_proteins(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        """Attach ``formulation_intent`` + ``commerce_sidecar``; ``quality_hooks`` only from caller guidance."""
        merged = deepcopy(blueprint)
        rows = merged.get("physical_backbone", {}).get("proteins_and_chains", [])
        if not self._result_specs:
            merged["formulation_note"] = "result_specs empty — no organ-route lipid merge applied"
            return merged

        for row in rows:
            if row.get("node_type") != "PROTEIN" and not (
                self.include_peptide_chains and row.get("node_type") == "MOLECULE_CHAIN"
            ):
                continue
            pid = str(row["node_id"])
            tokens = self._nearby_organ_tokens(pid)
            matched: list[OrganDeliverySpec] = [
                s for s in self._result_specs if self._spec_matches_tokens(tokens, s)
            ]
            if not matched:
                row["formulation_intent"] = {
                    "matched_delivery_specs": [],
                    "lipid_nanoparticle_strategy": [],
                    "process_expectations_checklist": [],
                    "quality_hooks": [],
                    "note": "no organ token overlap between graph neighborhood and result_specs",
                }
                row["commerce_sidecar"] = None
                continue

            lipid_labels: set[str] = set()
            process_keys: set[str] = set()
            quality: list[str] = []
            routes: list[ClinicalRoute] = []
            excipients: list[str] = []
            stability: list[str] = []

            for spec in matched:
                routes.append(spec.primary_route)
                l1, p1 = _route_strategy_labels(spec.primary_route)
                lipid_labels.update(l1)
                process_keys.update(p1)
                if spec.cited_guidance:
                    quality.extend(spec.cited_guidance)
                if spec.allowed_excipient_classes:
                    excipients.extend(spec.allowed_excipient_classes)
                if spec.stability_storage_class:
                    stability.append(spec.stability_storage_class)

            shop_route = "oral" if all(r == "oral" for r in routes) else "injection"

            row["formulation_intent"] = {
                "matched_delivery_specs": [asdict(s) for s in matched],
                "lipid_nanoparticle_strategy": sorted(lipid_labels),
                "process_expectations_checklist": sorted(process_keys),
                "quality_hooks": quality,
                "caller_excipient_classes": sorted(set(excipients)),
                "caller_stability_notes": stability,
            }
            row["commerce_sidecar"] = {"absorption": shop_route, "clinical_routes": routes}

        return merged

    def emit_node_type_artifacts(self, output_dir: Path | str) -> dict[str, str]:
        """
        Per-type JSON files + ``design_manifest.json``; node payloads use ``nx.node_link_data``
        shape slices where helpful for ctlr parity.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        blueprint = self.build_working_substrate_blueprint()
        final_bp = self.apply_result_specs_to_proteins(blueprint)

        by_t = self.group_nodes_by_type()
        formulation_by_id: dict[str, dict[str, Any]] = {}
        for row in final_bp.get("physical_backbone", {}).get("proteins_and_chains", []):
            formulation_by_id[str(row["node_id"])] = {
                "formulation_intent": row.get("formulation_intent"),
                "commerce_sidecar": row.get("commerce_sidecar"),
            }

        written: dict[str, str] = {}
        for nt, nodes in sorted(by_t.items(), key=lambda x: x[0]):
            merged_nodes: list[dict[str, Any]] = []
            for n in nodes:
                if nt in _SLIM_BIOLOGY_NODE_TYPES:
                    item = self._slim_biology_node_item(n)
                else:
                    item = deepcopy(n)
                extra = formulation_by_id.get(str(item["id"]))
                if extra and (extra.get("formulation_intent") is not None or extra.get("commerce_sidecar")):
                    item["designer_overlay"] = {k: v for k, v in extra.items() if v is not None}
                merged_nodes.append(item)
            fname = f"design_{_safe_filename_token(nt)}.json"
            fpath = out / fname
            payload = {"node_type": nt, "count": len(merged_nodes), "nodes": merged_nodes}
            fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            written[nt] = str(fpath.resolve())

        blueprint_path = out / "design_blueprint.json"
        blueprint_path.write_text(
            json.dumps(final_bp, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        written["__blueprint__"] = str(blueprint_path.resolve())

        manifest = {
            "graph_digest": _stable_graph_digest(self._nx),
            "node_count": self._nx.number_of_nodes(),
            "edge_count": self._nx.number_of_edges(),
            "multigraph": bool(self._nx.is_multigraph()),
            "result_specs_count": len(self._result_specs),
            "per_type_files": {k: v for k, v in written.items() if k != "__blueprint__"},
            "blueprint_path": written["__blueprint__"],
            "ctlr_compatibility": "Consumes the same nx.Graph / MultiGraph as TissueGraphController.filter_components output.",
        }
        man_path = out / "design_manifest.json"
        man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        written["__manifest__"] = str(man_path.resolve())
        return written


def run(
    graph: nx.Graph | nx.MultiGraph | None = None,
    *,
    gutils: GUtils | None = None,
    result_specs: Sequence[OrganDeliverySpec | dict[str, Any]] | None = None,
    output_dir: Path | str,
    include_peptide_chains: bool = False,
) -> dict[str, str]:
    """Convenience: construct ``Designer`` and emit artifacts."""
    d = Designer(
        graph=graph,
        gutils=gutils,
        result_specs=result_specs,
        include_peptide_chains=include_peptide_chains,
    )
    return d.emit_node_type_artifacts(output_dir)
