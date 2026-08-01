"""cycle/run.py:establish_bootstrap_baseline 단위 테스트 (#100).

bootstrap 배치 종료 시 최고 attempt를 BasePipeline 대비(confirm_and_measure,
best_source=None) 검증해 raw.pipelines에 확정 baseline으로 승격하는 로직.
confirm_and_measure/materialize_best_pipeline/S3 IO/insert_pipeline은 전부
monkeypatch해 순수 배선(조회 → confirm → 승격)만 검증한다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from cycle.promotion import ConfirmResult
from cycle.run import establish_bootstrap_baseline


def _conn_seq(*fetchone_results) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = list(fetchone_results)
    return conn


_COMMON_KW = dict(
    train=pl.DataFrame({"x": [1.0, 2.0], "y": [0.0, 1.0]}),
    target_col="y",
    metric="auc",
    n_splits=3,
    is_classification=True,
)


def test_returns_false_when_confirmed_pipeline_already_exists():
    """이미 확정 파이프라인이 있으면(재부트스트랩 등) 아무 조회도 더 안 하고 스킵."""
    conn = _conn_seq((1,))
    result = establish_bootstrap_baseline(conn, "s4e1", **_COMMON_KW)
    assert result is False
    assert conn.execute.call_count == 1


def test_returns_false_when_no_scored_attempt():
    conn = _conn_seq(None, None)
    result = establish_bootstrap_baseline(conn, "s4e1", **_COMMON_KW)
    assert result is False


def test_returns_false_when_best_attempt_has_no_code_path():
    conn = _conn_seq(None, ("attempt-1", 0.9, None))
    result = establish_bootstrap_baseline(conn, "s4e1", **_COMMON_KW)
    assert result is False


def test_confirmed_promotes_and_returns_true():
    conn = _conn_seq(
        None,                                   # 확정 파이프라인 없음
        ("attempt-1", 0.9, "path/to/code"),      # best attempt
        ({"fingerprint": "x"},),                 # fingerprint row
    )
    with patch("cycle.run._code_download", return_value="code body"), \
         patch("cycle.run.split_audit_holdout", return_value=(_COMMON_KW["train"], _COMMON_KW["train"])), \
         patch(
             "cycle.run.confirm_and_measure",
             return_value=ConfirmResult(confirmed=True, holdout_score=0.85, seed_gains={"7": {}}),
         ), \
         patch("cycle.run.materialize_best_pipeline", return_value="materialized code"), \
         patch("cycle.run.insert_pipeline") as mock_insert, \
         patch("cycle.run._best_pipeline_upload") as mock_upload:
        result = establish_bootstrap_baseline(conn, "s4e1", **_COMMON_KW)

    assert result is True
    mock_insert.assert_called_once()
    assert mock_insert.call_args.kwargs["competition_id"] == "s4e1"
    assert mock_insert.call_args.kwargs["cv_score"] == 0.9
    assert mock_insert.call_args.kwargs["gain_vs_best"] is None
    assert mock_insert.call_args.kwargs["materialized_code"] == "materialized code"
    mock_upload.assert_called_once_with("s4e1", "materialized code")


def test_confirm_and_measure_called_with_best_source_none():
    """BasePipeline 대비 검증이어야 하므로 best_source=None으로 confirm해야 한다."""
    conn = _conn_seq(None, ("attempt-1", 0.9, "path/to/code"), ({},))
    with patch("cycle.run._code_download", return_value="code body"), \
         patch("cycle.run.split_audit_holdout", return_value=(_COMMON_KW["train"], _COMMON_KW["train"])), \
         patch("cycle.run.confirm_and_measure", return_value=ConfirmResult(confirmed=False, holdout_score=None)) as mock_confirm, \
         patch("cycle.run.materialize_best_pipeline"), \
         patch("cycle.run.insert_pipeline"), \
         patch("cycle.run._best_pipeline_upload"):
        establish_bootstrap_baseline(conn, "s4e1", **_COMMON_KW)

    assert mock_confirm.call_args.kwargs["best_source"] is None
    assert mock_confirm.call_args.kwargs["source"] == "code body"


def test_not_confirmed_does_not_promote():
    conn = _conn_seq(None, ("attempt-1", 0.9, "path/to/code"))
    with patch("cycle.run._code_download", return_value="code body"), \
         patch("cycle.run.split_audit_holdout", return_value=(_COMMON_KW["train"], _COMMON_KW["train"])), \
         patch("cycle.run.confirm_and_measure", return_value=ConfirmResult(confirmed=False, holdout_score=None)), \
         patch("cycle.run.insert_pipeline") as mock_insert, \
         patch("cycle.run._best_pipeline_upload") as mock_upload:
        result = establish_bootstrap_baseline(conn, "s4e1", **_COMMON_KW)

    assert result is False
    mock_insert.assert_not_called()
    mock_upload.assert_not_called()


def test_empty_code_after_header_strip_returns_false():
    conn = _conn_seq(None, ("attempt-1", 0.9, "path/to/code"))
    with patch("cycle.run._code_download", return_value=""):
        result = establish_bootstrap_baseline(conn, "s4e1", **_COMMON_KW)
    assert result is False
