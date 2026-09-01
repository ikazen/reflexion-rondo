"""bin.rebuild_best_pipeline — 재생 + MinIO 업로드 + 신뢰 스냅샷 기록 (#278).

재생 결과를 최신 유효행의 materialized_code/materialized_sha256으로 심어야
_baseline_source_guard와 submit.py가 재구성된 blob을 정본으로 검증한다.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from bin.rebuild_best_pipeline import rebuild

_BEST = "class BestPipeline:\n    pass\n"
_SHA = hashlib.sha256(_BEST.encode()).hexdigest()


def _conn(latest_valid_id):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (
        (latest_valid_id,) if latest_valid_id else None
    )
    return conn


def test_dry_run_does_not_upload_or_write():
    conn = _conn("pipe-9")
    with (
        patch("cycle.materialize.replay_best_pipeline", return_value=(_BEST, None, 2)),
        patch("store.db.connect", return_value=conn),
        patch("store.s3_code.upload_best_pipeline") as up,
    ):
        rebuild("playground-series-s4e11", dry_run=True)
    up.assert_not_called()
    assert [c for c in conn.execute.call_args_list if "update" in c.args[0].lower()] == []


def test_uploads_and_records_trusted_snapshot():
    conn = _conn("pipe-9")
    with (
        patch("cycle.materialize.replay_best_pipeline", return_value=(_BEST, None, 2)),
        patch("store.db.connect", return_value=conn),
        patch("store.s3_code.upload_best_pipeline") as up,
    ):
        rebuild("playground-series-s4e11", dry_run=False)
    up.assert_called_once_with("playground-series-s4e11", _BEST)
    writes = [c for c in conn.execute.call_args_list if "set materialized_code" in c.args[0]]
    assert len(writes) == 1
    assert writes[0].args[1] == [_BEST, _SHA, "pipe-9"]


def test_exits_when_no_promoted_pipeline():
    conn = _conn(None)
    with (
        patch("cycle.materialize.replay_best_pipeline", return_value=(None, None, 0)),
        patch("store.db.connect", return_value=conn),
        patch("store.s3_code.upload_best_pipeline"),
        pytest.raises(SystemExit),
    ):
        rebuild("playground-series-s4e11", dry_run=False)
