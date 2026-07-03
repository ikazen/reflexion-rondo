"""BON-245: bin/submit.py 정확성 버그 4건 회귀 테스트.

(a) 자동 선택 경로는 confirmed 파이프라인(raw.pipelines)만 소스로 써야 한다.
(b) predict_proba 분기는 metric_class == "binary_proba" 기준이어야 한다.
(c) 제출 값 컬럼은 sample_submission.csv의 실제 컬럼명을 따라야 한다.
(d) NaN 중앙값 대치는 train/test 대칭이어야 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.submit import (
    _impute_train_test_median,
    _load_best_code,
    _predict_raw,
    _submission_value_col,
)


# ---------------------------------------------------------------------------
# (a) _load_best_code
# ---------------------------------------------------------------------------

def _conn_with(row) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = row
    return conn


def test_auto_select_queries_pipelines_not_attempts() -> None:
    """attempt_id 미지정 시 raw.pipelines를 조회하고 raw.attempts는 조회하지 않는다."""
    conn = _conn_with(("code text", 0.91, "attempt-123"))
    with patch("store.db.connect", return_value=conn):
        source, cv_score, aid = _load_best_code("s4e1", None)
    sql = conn.execute.call_args.args[0]
    assert "raw.pipelines" in sql
    assert "raw.attempts" not in sql
    assert source == "code text"
    assert cv_score == 0.91
    assert aid == "attempt-123"


def test_auto_select_returns_code_directly_without_s3_download() -> None:
    """raw.pipelines.code는 이미 스트립된 텍스트라 S3 재다운로드가 없어야 한다."""
    conn = _conn_with(("code text", 0.91, "attempt-123"))
    with patch("store.db.connect", return_value=conn), \
         patch("store.s3_code.download") as mock_download:
        _load_best_code("s4e1", None)
    mock_download.assert_not_called()


def test_auto_select_raises_when_no_confirmed_pipeline() -> None:
    """confirmed 파이프라인이 없으면 명확한 에러로 --attempt-id 사용을 안내한다."""
    conn = _conn_with(None)
    with patch("store.db.connect", return_value=conn):
        with pytest.raises(ValueError, match="No confirmed pipeline"):
            _load_best_code("s4e1", None)


def test_explicit_attempt_id_still_uses_attempts_and_s3() -> None:
    """--attempt-id 지정 시 기존처럼 raw.attempts + S3 다운로드 경로를 그대로 쓴다."""
    conn = _conn_with(("path/to/code.py", 0.80, "attempt-999"))
    with patch("store.db.connect", return_value=conn), \
         patch("store.s3_code.download", return_value="header\n" + ("# " + "-" * 60) + "\nsource here") as mock_download:
        source, cv_score, aid = _load_best_code("s4e1", "attempt-999")
    sql = conn.execute.call_args.args[0]
    assert "raw.attempts" in sql
    mock_download.assert_called_once_with("path/to/code.py")
    assert source == "source here"
    assert cv_score == 0.80
    assert aid == "attempt-999"


# ---------------------------------------------------------------------------
# (b) _predict_raw
# ---------------------------------------------------------------------------

def test_predict_raw_binary_proba_calls_predict_proba() -> None:
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
    result = _predict_raw(model, np.zeros((2, 3)), "binary_proba")
    model.predict_proba.assert_called_once()
    model.predict.assert_not_called()
    assert list(result) == [0.9, 0.2]


@pytest.mark.parametrize("metric_class", ["classification", "regression_error"])
def test_predict_raw_non_proba_calls_predict(metric_class: str) -> None:
    model = MagicMock()
    model.predict.return_value = np.array([1, 0])
    result = _predict_raw(model, np.zeros((2, 3)), metric_class)
    model.predict.assert_called_once()
    model.predict_proba.assert_not_called()
    assert list(result) == [1, 0]


# ---------------------------------------------------------------------------
# (c) _submission_value_col
# ---------------------------------------------------------------------------

def test_submission_value_col_uses_sample_second_column() -> None:
    assert _submission_value_col(["id", "target_prob"], "target") == "target_prob"


def test_submission_value_col_falls_back_when_missing() -> None:
    assert _submission_value_col(["id"], "target") == "target"


# ---------------------------------------------------------------------------
# (d) _impute_train_test_median
# ---------------------------------------------------------------------------

def test_impute_train_test_median_fills_both() -> None:
    train_np = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    test_np = np.array([[np.nan, 8.0]])

    train_out, test_out = _impute_train_test_median(train_np, test_np)

    assert not np.isnan(train_out).any()
    assert not np.isnan(test_out).any()
    # train의 NaN은 train 컬럼 중앙값(5.0)으로 대치돼야 한다 (col 1: [nan,4,6] -> median 5)
    assert train_out[0, 1] == 5.0
    # test의 NaN도 동일 train 중앙값 기준 (col 0: [1,3,5] -> median 3)
    assert test_out[0, 0] == 3.0


def test_impute_train_test_median_noop_when_no_nan() -> None:
    train_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    test_np = np.array([[5.0, 6.0]])
    train_out, test_out = _impute_train_test_median(train_np, test_np)
    assert np.array_equal(train_out, train_np)
    assert np.array_equal(test_out, test_np)
