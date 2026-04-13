"""
Deterministic graph node identities and structured phase logging for ``data/`` workflows.

Prompt (user): analyze the workflows of data dir and fix: (1) capture example/placeholder
data-driven sections with production-ready implementation and precise prints; (2) no nested
node ids — create nodes for composite semantics and link with edges; (3) precise docstrings;
(4) efficient process/edge creation; (5) smart graph standards and low overhead.

CHAR: ``canonical_node_id`` builds opaque, stable ids from a sorted JSON payload — never embeds
another node's ``id`` string in the output. Domain keys (accession, go_id, path_key, …) may appear
only inside the hashed ``parts`` dict. GO_TERM nodes keep the conventional ``GO_<curie>`` form
(a single external key, not a graph-node embedding).

Node conventions (minimal):
    - Every node: ``type``, ``label``; optional ``stub: True`` for minimal placeholders pending enrichment; optional ``external_ids`` dict for CURIEs.
    - Edges: ``rel``, ``src_layer``, ``trgt_layer`` in attrs; keep heavy numerics on the node or
      on one primary edge to avoid duplicate payloads.

Migration: exports using ``networkx.node_link_data`` from builds before graph-identity hardening
used composite string ids (e.g. ``SPATIAL_REGION_<scan>_<i>``). Re-run the pipeline; old JSON is
not compatible (clean break).
"""
from __future__ import annotations

import json
import hashlib
import time
from urllib.parse import urlparse


def _normalize_for_json(obj: object):
    """CHAR: floats rounded so tiny numerical drift does not change stable hashes."""
    if isinstance(obj, float):
        return round(obj, 12)
    if isinstance(obj, dict):
        return {k: _normalize_for_json(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_json(x) for x in obj]
    return obj


def canonical_node_id(namespace: str, parts: dict, *, digest_bytes: int = 16) -> str:
    """
    Build ``{namespace}_{hexdigest}`` from BLAKE2b over canonical JSON of ``parts``.

    Parameters
    ----------
    namespace:
        Short uppercase prefix (e.g. ``GOCAMACT``, ``SPATREG``).
    parts:
        JSON-serializable dict; keys are sorted for stability. Use domain keys only — do not pass
        another graph node's string ``id`` as a value unless that value is itself an external
        stable key (e.g. UniProt accession used as PROTEIN node id is acceptable as *data*).
    digest_bytes:
        BLAKE2b digest length (default 16 bytes → 32 hex chars).
    """
    payload = json.dumps(
        _normalize_for_json(parts), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    h = hashlib.blake2b(payload, digest_size=digest_bytes).hexdigest()
    return f"{namespace}_{h}"


def go_term_node_id(go_curie: str) -> str:
    """Single-key GO term node id: ``GO:0008150`` → ``GO_0008150``."""
    return f"GO_{go_curie.replace(':', '_')}"


def phase_log(phase: str, action: str, **kwargs) -> None:
    """Stable key=value logging for scraping; omit None values."""
    bits = [f"phase={phase}", f"action={action}"]
    for k in sorted(kwargs):
        v = kwargs[k]
        if v is None:
            continue
        bits.append(f"{k}={v}")
    print(" | ".join(bits))


def phase_http_log(
    phase: str,
    action: str,
    url: str,
    *,
    status_code: int | None = None,
    elapsed_ms: float | None = None,
    err_class: str | None = None,
) -> None:
    """Log HTTP outcome without full URLs (host + optional path prefix only)."""
    host = ""
    path_prefix = ""
    if url:
        p = urlparse(url)
        host = p.netloc or ""
        path_prefix = (p.path or "")[:48]
    phase_log(
        phase,
        action,
        url_host=host or None,
        path_prefix=path_prefix or None,
        http_status=status_code,
        elapsed_ms=round(elapsed_ms, 2) if elapsed_ms is not None else None,
        error_class=err_class,
    )


def timed_ms() -> float:
    """CHAR: monotonic ms for elapsed logging."""
    return time.perf_counter() * 1000.0
