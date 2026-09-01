"""cycle.run._baseline_source_guard 단위 테스트 (#278, ADR-042).

격리/remeasure로 raw.pipelines 유효집합이 바뀌었는데 MinIO best_pipeline.py를
재구성하지 않으면 promote 게이트(레지스트리)와 confirm 게이트(MinIO blob)가 다른
baseline을 본다 — blob sha가 최신 유효행의 신뢰 해시와 어긋나면 게이트가 멈춰야 한다.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest

from cycle.run import BaselineSourceMismatchError, _baseline_source_guard

_BLOB = "import polars as pl\n\nclass BestPipeline:\n    pass\n"
_BLOB_SHA = hashlib.sha256(_BLOB.encode()).hexdigest()


def _conn(trusted):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (trusted,) if trusted is not None else None
    return conn


def _updates(conn):
    return [c for c in conn.execute.call_args_list if c.args[0].strip().lower().startswith("update")]


def test_passes_when_blob_missing(monkeypatch):
    monkeypatch.setattr("cycle.run._best_pipeline_download", lambda cid: None)
    conn = _conn(_BLOB_SHA)
    _baseline_source_guard(conn, "s4e11")
    assert _updates(conn) == []


def test_passes_when_blob_matches_trusted(monkeypatch):
    monkeypatch.setattr("cycle.run._best_pipeline_download", lambda cid: _BLOB)
    conn = _conn(_BLOB_SHA)
    _baseline_source_guard(conn, "s4e11")
    assert _updates(conn) == []


def test_passes_when_no_trusted_sha(monkeypatch):
    monkeypatch.setattr("cycle.run._best_pipeline_download", lambda cid: _BLOB)
    conn = _conn(None)
    _baseline_source_guard(conn, "s4e11")
    assert _updates(conn) == []


def test_raises_and_pauses_on_mismatch(monkeypatch):
    monkeypatch.setattr("cycle.run._best_pipeline_download", lambda cid: _BLOB)
    conn = _conn("deadbeef" * 8)  # 다른 sha (격리행이 blob에 남아있는 상황)
    with pytest.raises(BaselineSourceMismatchError):
        _baseline_source_guard(conn, "s4e11")
    pauses = [c for c in conn.execute.call_args_list if "auto_submit_paused_reason" in c.args[0]]
    assert len(pauses) == 1
    assert "coalesce(auto_submit_paused_reason" in pauses[0].args[0]
    assert "rebuild_best_pipeline --competition s4e11" in pauses[0].args[1][0]
