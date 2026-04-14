"""
Live MCP client for the ``generation`` tool over SSE — matches ``server.py`` entrypoint.

Prompt (user): Test the entire script from a test server/client file; entry point is
always the server file (must be exposed as set in the Dockerfile). Hardcode test cases
to query the generate route of the server.

Prompt (user): Adapt test_mcp generation to the same spec as defined within the route.

Prompt (user): Save the resulting content within the test ``out-dir``.

Prompt (user): Rerun the process and save all logs in test/logs — after a run check them
and adapt problematic sections to finish retries with exit code 0 (0 fails / errs).

CHAR: Logs → ``tests/logs`` (``ACID_MASTER_E2E_LOG_DIR``). Artifacts → ``tests/out-dir``.
Retries: per-case attempts + full run passes (env-tunable) with backoff until all cases
pass or limits are hit. Exit ``0`` only when every case succeeds.

CHAR: Case payloads mirror ``server.generation`` exactly:

- ``prompt: str`` — outcome, constraints, toxicity flags, diseases to avoid.
- ``scan_2d_file: bytes | None`` — optional; with ``scan_2d_filename`` only (both or neither).
- ``scan_2d_filename: str | None`` — basename + extension when scan bytes are sent.
- ``filter_physical_compound: str | list[str] | None`` — comma/semicolon-separated string
  or token list (``ds.PHYSICAL_CATEGORY_ALIASES``).

Expected success payload: pipeline dict plus ``session_id``, ``session_artifact_dir``,
``artifacts_uri_prefix`` (``output/<session_id>``), ``artifacts_zip_path``,
``artifacts_manifest``.

CHAR: ``server.py`` runs ``mcp.run(transport=\"sse\", ...)`` → FastMCP SSE at ``/sse``.
Dockerfile ``EXPOSE 8000`` matches default ``ACID_MASTER_PORT``.

Run (server must already be listening, e.g. ``docker run -p 8000:8000 ...``):

    pip install -r requirements-e2e.txt
    python tests/test_mcp_generation_sse_client.py

Pytest (requires ``ACID_MASTER_E2E=1``):

    ACID_MASTER_E2E=1 pytest tests/test_mcp_generation_sse_client.py -m integration -v
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# CHAR: ``pytest`` is only required when pytest collects this module; CLI runs must not depend on it.
try:
    import pytest
except ImportError:  # pragma: no cover
    class _E2EPytestShim:
        class mark:
            @staticmethod
            def integration(f: Any) -> Any:
                return f

            @staticmethod
            def asyncio(f: Any) -> Any:
                return f

    pytest = _E2EPytestShim()  # type: ignore[misc, assignment]

# CHAR: E2E only — fastmcp is not part of requirements-test.txt (CI unit tests).
try:
    from fastmcp import Client
    from fastmcp.client.transports import SSETransport
except ImportError:  # pragma: no cover
    Client = None  # type: ignore[misc, assignment]
    SSETransport = None  # type: ignore[misc, assignment]

_LOG = logging.getLogger("acid_master.e2e_generation")

_MISSING_FASTMCP_MSG = "Install fastmcp for E2E (pip install -r requirements-e2e.txt). Python 3.10+ required."

# CHAR: 1×1 PNG — valid raster bytes for ``scan_2d_file`` (loader accepts raster per route doc).
_MIN_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
MIN_SCAN_PNG_BYTES = base64.b64decode(_MIN_PNG_B64)

# CHAR: align with Dockerfile EXPOSE 8000 and server default port.
_DEFAULT_HOST = os.environ.get("ACID_MASTER_CLIENT_HOST", "127.0.0.1")
_DEFAULT_PORT = int(os.environ.get("ACID_MASTER_PORT", "8000"))
DEFAULT_SSE_URL = os.environ.get(
    "ACID_MASTER_SSE_URL",
    f"http://{_DEFAULT_HOST}:{_DEFAULT_PORT}/sse",
)


def e2e_output_dir() -> Path:
    """Directory for saved E2E payloads (default: ``tests/out-dir``)."""
    raw = (os.environ.get("ACID_MASTER_E2E_OUT_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent / "out-dir"


def e2e_logs_dir() -> Path:
    """Directory for E2E run logs (default: ``tests/logs``)."""
    raw = (os.environ.get("ACID_MASTER_E2E_LOG_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent / "logs"


def setup_e2e_logging() -> Path:
    """Configure root logging to ``tests/logs/e2e_generation_<utc>.log`` and stderr."""
    log_dir = e2e_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = log_dir / f"e2e_generation_{ts}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    _LOG.info("Log file: %s", path)
    return path


def _sanitize_request_args(args: dict[str, Any]) -> dict[str, Any]:
    """Make ``generation`` call arguments JSON-safe (embed scan bytes as base64 metadata)."""
    out: dict[str, Any] = {}
    for key, val in args.items():
        if key == "scan_2d_file" and isinstance(val, (bytes, bytearray, memoryview)):
            raw = bytes(val)
            out[key] = {
                "__bytes_base64__": base64.b64encode(raw).decode("ascii"),
                "byte_length": len(raw),
            }
        else:
            out[key] = val
    return out


def _write_e2e_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _coerce_tool_result_dict(data: Any) -> dict[str, Any] | None:
    """
    Normalize FastMCP / MCP tool payloads to a flat dict with ``session_id`` when present.

    CHAR: Some transports wrap the server dict under ``result`` or duplicate nesting.
    """
    if isinstance(data, dict) and "session_id" in data:
        return data
    if isinstance(data, dict):
        inner = data.get("result")
        if isinstance(inner, dict) and "session_id" in inner:
            return inner
        # CHAR: unwrap single-key legacy wrappers
        if len(data) == 1:
            only = next(iter(data.values()))
            if isinstance(only, dict) and "session_id" in only:
                return only
    return None


# Hardcoded cases: each ``arguments`` dict is a direct ``generation`` tool call (same keys as the route).
GENERATION_SSE_CASES: list[dict[str, Any]] = [
    {
        "id": "prompt_only",
        "arguments": {
            "prompt": (
                "Human liver glucose transport and metabolism; avoid speculative toxicity; "
                "exclude undiagnosed disease claims; use public KB seeds only for organs and genes."
            ),
        },
    },
    {
        "id": "filter_physical_compound_str",
        "arguments": {
            "prompt": (
                "Human cardiac tissue — enzymes and receptors for energy metabolism; "
                "constraints: no off-label drug claims; diseases to avoid: unspecified cardiomyopathy."
            ),
            # CHAR: route accepts comma/semicolon-separated string.
            "filter_physical_compound": "gene, protein",
        },
    },
    {
        "id": "filter_physical_compound_list",
        "arguments": {
            "prompt": (
                "Brain cortex expression context; outcome: tissue and gene linkage only; "
                "toxicity flags: none asserted; avoid pediatric disease specificity."
            ),
            "filter_physical_compound": ["tissue", "gene"],
        },
    },
    {
        "id": "optional_2d_scan_png",
        "arguments": {
            "prompt": (
                "Cross-modal sanity: same biological seed as unit tests; optional imaging cue only; "
                "no diagnostic inference from raster."
            ),
            "scan_2d_file": MIN_SCAN_PNG_BYTES,
            "scan_2d_filename": "synthetic_slice.png",
        },
    },
]


def _tool_timeout_sec() -> float:
    raw = os.environ.get("ACID_MASTER_TOOL_TIMEOUT_SEC", "900")
    return max(30.0, float(raw))


def _case_attempts() -> int:
    return max(1, int(os.environ.get("ACID_MASTER_E2E_CASE_ATTEMPTS", "4")))


def _run_passes() -> int:
    return max(1, int(os.environ.get("ACID_MASTER_E2E_RUN_PASSES", "3")))


def _retry_base_sec() -> float:
    return max(1.0, float(os.environ.get("ACID_MASTER_E2E_RETRY_BASE_SEC", "15")))


def _assert_generation_route_response(data: dict[str, Any]) -> str | None:
    """
    Validate the ``generation`` tool return contract from ``server.py`` docstring.

    Returns ``None`` if OK, else an error message.
    """
    need = (
        "session_id",
        "session_artifact_dir",
        "artifacts_uri_prefix",
        "artifacts_zip_path",
        "artifacts_manifest",
    )
    missing = [k for k in need if k not in data]
    if missing:
        return f"missing keys {missing}"

    sid = data["session_id"]
    prefix = data["artifacts_uri_prefix"]
    if prefix != f"output/{sid}":
        return f"artifacts_uri_prefix {prefix!r} != output/{sid!r}"

    manifest = data["artifacts_manifest"]
    if not isinstance(manifest, list):
        return f"artifacts_manifest must be list, got {type(manifest)!r}"

    if not data.get("session_artifact_dir"):
        return "session_artifact_dir empty"

    if not data.get("artifacts_zip_path"):
        return "artifacts_zip_path empty"

    return None


async def _sleep_backoff(attempt_idx: int, base_sec: float) -> None:
    delay = base_sec * (attempt_idx + 1)
    _LOG.info("Backoff sleep %.1fs before next attempt", delay)
    await asyncio.sleep(delay)


async def run_generation_sse_cases(
    sse_url: str | None = None,
    *,
    timeout_sec: float | None = None,
    out_dir: Path | None = None,
    case_max_attempts: int | None = None,
    retry_backoff_sec: float | None = None,
) -> list[dict[str, Any]]:
    """
    Connect via SSE, list tools, call ``generation`` for each hardcoded case.

    Per-case transient failures are retried up to ``case_max_attempts`` with backoff.

    Writes one JSON file per case under ``out_dir`` (default ``tests/out-dir``) and
    ``run_summary.json`` for the full run.

    Returns one summary dict per case with keys: ``id``, ``ok``, ``detail``, optional ``saved``.
    """
    if Client is None or SSETransport is None:
        raise RuntimeError(_MISSING_FASTMCP_MSG)

    url = sse_url or DEFAULT_SSE_URL
    dest_root = out_dir if out_dir is not None else e2e_output_dir()
    to = timeout_sec if timeout_sec is not None else _tool_timeout_sec()
    attempts_cap = case_max_attempts if case_max_attempts is not None else _case_attempts()
    backoff = retry_backoff_sec if retry_backoff_sec is not None else _retry_base_sec()

    transport = SSETransport(url=url)
    client = Client(transport)
    summaries: list[dict[str, Any]] = []

    try:
        async with client:
            tools = await client.list_tools()
            names = {t.name for t in tools}
            if "generation" not in names:
                summaries = [
                    {
                        "id": "_session",
                        "ok": False,
                        "detail": f"MCP server at {url!r} has no 'generation' tool; got {sorted(names)}",
                    },
                ]
            else:
                for case in GENERATION_SSE_CASES:
                    cid = str(case["id"])
                    args = dict(case["arguments"])
                    req_safe = _sanitize_request_args(args)

                    def _write_fail(phase: str, detail: str, *, response: Any = None) -> Path:
                        fname = f"{cid}_failed.json"
                        payload: dict[str, Any] = {
                            "case_id": cid,
                            "ok": False,
                            "phase": phase,
                            "detail": detail,
                            "request": req_safe,
                        }
                        if response is not None:
                            payload["raw_response"] = response
                        path = dest_root / fname
                        _write_e2e_json(path, payload)
                        return path

                    case_ok = False
                    last_detail = ""

                    for attempt in range(attempts_cap):
                        _LOG.info("Case %r attempt %s/%s", cid, attempt + 1, attempts_cap)

                        try:
                            result = await client.call_tool(
                                "generation",
                                args,
                                timeout=to,
                                raise_on_error=False,
                            )
                        except Exception as exc:  # noqa: BLE001 — transport / protocol errors
                            last_detail = f"call_tool error: {exc!r}"
                            _LOG.warning("%s", last_detail)
                            if attempt < attempts_cap - 1:
                                await _sleep_backoff(attempt, backoff)
                            continue

                        is_err = bool(getattr(result, "is_error", False) or getattr(result, "isError", False))

                        raw_data = getattr(result, "data", None)
                        if raw_data is None and hasattr(result, "structured_content"):
                            raw_data = result.structured_content  # type: ignore[attr-defined]

                        if is_err:
                            last_detail = f"tool error: {raw_data!r}"
                            _LOG.warning("%s", last_detail)
                            if attempt < attempts_cap - 1:
                                await _sleep_backoff(attempt, backoff)
                            continue

                        data = _coerce_tool_result_dict(raw_data)
                        if data is None:
                            last_detail = f"unexpected payload type/shape: {type(raw_data)!r} {raw_data!r}"
                            _LOG.warning("%s", last_detail)
                            if attempt < attempts_cap - 1:
                                await _sleep_backoff(attempt, backoff)
                            continue

                        err = _assert_generation_route_response(data)
                        if err:
                            last_detail = err
                            _LOG.warning("contract: %s", err)
                            if attempt < attempts_cap - 1:
                                await _sleep_backoff(attempt, backoff)
                            continue

                        sid = str(data["session_id"])
                        out_path = dest_root / f"{cid}_{sid}.json"
                        _write_e2e_json(
                            out_path,
                            {
                                "case_id": cid,
                                "ok": True,
                                "attempt": attempt + 1,
                                "request": req_safe,
                                "response": data,
                            },
                        )
                        summaries.append(
                            {
                                "id": cid,
                                "ok": True,
                                "detail": f"session_id={sid!r}",
                                "saved": str(out_path),
                            },
                        )
                        _LOG.info("Case %r OK (attempt %s)", cid, attempt + 1)
                        case_ok = True
                        break

                    if not case_ok:
                        fail_path = _write_fail("exhausted_retries", last_detail or "unknown failure")
                        summaries.append({"id": cid, "ok": False, "detail": last_detail, "saved": str(fail_path)})

    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, RuntimeError) and "Install fastmcp" in str(exc):
            raise
        _LOG.exception("MCP SSE session failed (connect or protocol); see traceback")
        summaries = [{"id": "_session", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}]

    _write_e2e_json(
        dest_root / "run_summary.json",
        {
            "sse_url": url,
            "out_dir": str(dest_root),
            "summaries": summaries,
            "all_ok": all(s.get("ok") for s in summaries),
        },
    )

    return summaries


async def run_generation_sse_suite_with_run_retries(
    sse_url: str | None = None,
    *,
    timeout_sec: float | None = None,
    out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Run ``run_generation_sse_cases`` up to ``ACID_MASTER_E2E_RUN_PASSES`` times if any case fails.

    CHAR: Fresh MCP session each pass (new TCP connection) helps recover from dropped SSE.
    """
    passes = _run_passes()
    backoff = _retry_base_sec()
    url = sse_url or DEFAULT_SSE_URL
    dest_root = out_dir if out_dir is not None else e2e_output_dir()
    last_summaries: list[dict[str, Any]] = []

    for pass_idx in range(passes):
        _LOG.info("=== E2E run pass %s/%s → %s ===", pass_idx + 1, passes, url)
        try:
            last_summaries = await run_generation_sse_cases(
                url,
                timeout_sec=timeout_sec,
                out_dir=dest_root,
            )
        except RuntimeError as exc:
            if "Install fastmcp" in str(exc):
                raise
            _LOG.exception("Run pass %s unexpected RuntimeError", pass_idx + 1)
            last_summaries = [{"id": "_session", "ok": False, "detail": str(exc)}]
        except Exception:
            _LOG.exception("Run pass %s unexpected error", pass_idx + 1)
            last_summaries = [{"id": "_session", "ok": False, "detail": "unexpected exception (see log)"}]

        if last_summaries and all(s.get("ok") for s in last_summaries):
            _LOG.info("All cases passed on run pass %s", pass_idx + 1)
            return last_summaries

        if pass_idx < passes - 1:
            _LOG.warning("Run pass %s had failures; retrying after backoff", pass_idx + 1)
            await _sleep_backoff(pass_idx, backoff)

    return last_summaries


def main() -> int:
    """CLI entry: run all hardcoded generation cases against the live SSE server (with retries)."""
    setup_e2e_logging()
    if Client is None or SSETransport is None:
        _LOG.error("%s", _MISSING_FASTMCP_MSG)
        print(_MISSING_FASTMCP_MSG, file=sys.stderr)
        return 1
    _LOG.info("Acid Master MCP SSE client → %s", DEFAULT_SSE_URL)
    _LOG.info("E2E artifacts → %s", e2e_output_dir())
    _LOG.info(
        "Retries: case_attempts=%s run_passes=%s base_backoff_sec=%s",
        _case_attempts(),
        _run_passes(),
        _retry_base_sec(),
    )

    try:
        summaries = asyncio.run(run_generation_sse_suite_with_run_retries())
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("FATAL: %s", exc)
        return 1

    fails = [r for r in summaries if not r.get("ok")]
    for row in summaries:
        status = "OK" if row.get("ok") else "FAIL"
        line = f"[{status}] {row['id']}: {row.get('detail', '')}"
        print(line)
        _LOG.info(line)

    if fails:
        _LOG.error("Finished with %s failing case(s)", len(fails))
        return 2

    _LOG.info("Finished with 0 fails — exit 0")
    return 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_tool_via_sse_live_server() -> None:
    if os.environ.get("ACID_MASTER_E2E") != "1":
        pytest.skip("Set ACID_MASTER_E2E=1 with server.py listening (Docker EXPOSE 8000 → host port).")

    if Client is None:
        pytest.skip("fastmcp not installed; pip install -r requirements-e2e.txt")

    setup_e2e_logging()
    summaries = await run_generation_sse_suite_with_run_retries()
    failures = [s for s in summaries if not s.get("ok")]
    assert not failures, f"generation SSE cases failed after retries: {failures}"


if __name__ == "__main__":
    raise SystemExit(main())
