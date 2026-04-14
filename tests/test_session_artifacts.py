"""
Tests for session artifact helpers (scan persistence and zip bundling).

Prompt (user): Make this project a production ready system. Include test files if
needed inside a tests directory.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from session_artifacts import persist_scan_in_session, zip_session_artifacts


def test_persist_scan_writes_file_with_suffix(tmp_path: Path) -> None:
    session = tmp_path / "sess"
    out = persist_scan_in_session(session, b"abc", "study.dcm")
    assert Path(out).is_file()
    assert Path(out).read_bytes() == b"abc"
    assert Path(out).suffix == ".dcm"


def test_persist_scan_rejects_path_segments_in_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path segments"):
        persist_scan_in_session(tmp_path, b"x", r"sub\study.dcm")


def test_persist_scan_requires_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="extension"):
        persist_scan_in_session(tmp_path, b"x", "nofileext")


def test_persist_scan_deduplicates_same_basename(tmp_path: Path) -> None:
    session = tmp_path / "sess"
    p1 = persist_scan_in_session(session, b"a", "study.dcm")
    p2 = persist_scan_in_session(session, b"b", "study.dcm")
    assert p1 != p2
    assert Path(p2).read_bytes() == b"b"


def test_zip_session_skips_missing_files(tmp_path: Path) -> None:
    existing = tmp_path / "a.txt"
    existing.write_text("hello", encoding="utf-8")
    sid = "abc123"
    zpath = zip_session_artifacts(
        sid,
        [str(existing), "/nonexistent/file.json"],
        zip_root=tmp_path / "zips",
    )
    zp = Path(zpath)
    assert zp.is_file()
    with zipfile.ZipFile(zp, "r") as zf:
        names = zf.namelist()
        assert any(n.endswith("a.txt") for n in names)
