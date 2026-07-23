"""bin/submit.py 정확성 버그 4건 + 무결성 검증 회귀 테스트.

(a) 자동 선택 경로는 confirmed 파이프라인(raw.pipelines)만 소스로 써야 한다.
(b) predict_proba 분기는 metric_class == "binary_proba" 기준이어야 한다.
(c) 제출 값 컬럼은 sample_submission.csv의 실제 컬럼명을 따라야 한다.
(d) NaN 중앙값 대치는 train/test 대칭이어야 한다.
MinIO best_pipeline.py는 raw.pipelines.pipeline_sha256과 대조해야 한다.
attempt_only 재구성은 Patch 인스턴스의 클래스 속성을 보존해야 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.submit import (
    _bagged_predict,
    _dummy_target_value,
    _impute_train_test_median,
    _load_best_code,
    _load_pipeline,
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
    conn = _conn_with(("code text", 0.91, "attempt-123", "abc123sha"))
    with patch("store.db.connect", return_value=conn):
        source, cv_score, aid, sha = _load_best_code("s4e1", None)
    sql = conn.execute.call_args.args[0]
    assert "raw.pipelines" in sql
    assert "raw.attempts" not in sql
    assert source == "code text"
    assert cv_score == 0.91
    assert aid == "attempt-123"
    assert sha == "abc123sha"


def test_auto_select_returns_code_directly_without_s3_download() -> None:
    """raw.pipelines.code는 이미 스트립된 텍스트라 S3 재다운로드가 없어야 한다."""
    conn = _conn_with(("code text", 0.91, "attempt-123", "abc123sha"))
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
        source, cv_score, aid, sha = _load_best_code("s4e1", "attempt-999")
    sql = conn.execute.call_args.args[0]
    assert "raw.attempts" in sql
    mock_download.assert_called_once_with("path/to/code.py")
    assert source == "source here"
    assert cv_score == 0.80
    assert aid == "attempt-999"


def test_explicit_attempt_id_skips_hash_verification() -> None:
    """--attempt-id 명시 경로는 raw.pipelines 대조 해시가 없어 pipeline_sha256=None (의도된 escape hatch)."""
    conn = _conn_with(("path/to/code.py", 0.80, "attempt-999"))
    with patch("store.db.connect", return_value=conn), \
         patch("store.s3_code.download", return_value="src"):
        *_, sha = _load_best_code("s4e1", "attempt-999")
    assert sha is None


# ---------------------------------------------------------------------------
# _load_pipeline sha256 무결성 검증
# ---------------------------------------------------------------------------

def test_load_pipeline_raises_on_sha256_mismatch() -> None:
    """MinIO에서 받은 best_pipeline.py가 신뢰 해시와 다르면 exec 전에 raise한다."""
    with patch("store.s3_code.download_best_pipeline", return_value="class Patch:\n    pass\n"):
        with pytest.raises(RuntimeError, match="integrity check failed"):
            _load_pipeline("s4e1", expected_sha256="0" * 64)


def test_load_pipeline_passes_on_sha256_match() -> None:
    """해시가 일치하면 정상적으로 pipeline 인스턴스를 반환한다."""
    import hashlib
    source = "class Patch:\n    pass\n"
    correct_sha = hashlib.sha256(source.encode()).hexdigest()
    with patch("store.s3_code.download_best_pipeline", return_value=source):
        pipeline = _load_pipeline("s4e1", expected_sha256=correct_sha)
    assert pipeline is not None


def test_load_pipeline_skips_verification_when_sha256_none() -> None:
    """expected_sha256=None(예: --attempt-id 경로)이면 검증을 건너뛰고 정상 로드한다."""
    with patch("store.s3_code.download_best_pipeline", return_value="class Patch:\n    pass\n"):
        pipeline = _load_pipeline("s4e1", expected_sha256=None)
    assert pipeline is not None


# ---------------------------------------------------------------------------
# attempt_only는 MinIO best_pipeline.py를 참조하지 않아야 한다
# ---------------------------------------------------------------------------

def test_load_pipeline_attempt_only_skips_minio_download() -> None:
    """attempt_only=True면 download_best_pipeline을 아예 호출하지 않는다."""
    source = "class Patch:\n    def build_model(self, params, ctx):\n        return 'attempt-model'\n"
    with patch("store.s3_code.download_best_pipeline") as mock_download:
        pipeline = _load_pipeline("s4e1", extra_source=source, attempt_only=True)
    mock_download.assert_not_called()
    assert pipeline.build_model({}, None) == "attempt-model"


def test_load_pipeline_attempt_only_ignores_stale_minio_blob() -> None:
    """고아 MinIO best_pipeline.py가 있어도 attempt_only면 extra_source의 Patch가 이긴다.

    회귀 테스트 — 2026-07-17 s5e5에서 실제로 재현된 버그(orphaned best_pipeline.py가
    --attempt-id로 지정한 attempt 코드를 조용히 덮어써 Kaggle 제출 2건이 파국났음).
    """
    stale_minio_source = (
        "class Patch:\n    def build_model(self, params, ctx):\n        return 'stale-minio-model'\n"
    )
    attempt_source = (
        "class Patch:\n    def build_model(self, params, ctx):\n        return 'attempt-model'\n"
    )
    with patch("store.s3_code.download_best_pipeline", return_value=stale_minio_source) as mock_download:
        pipeline = _load_pipeline("s4e1", extra_source=attempt_source, attempt_only=True)
    mock_download.assert_not_called()
    assert pipeline.build_model({}, None) == "attempt-model"


def test_load_pipeline_attempt_only_without_source_falls_back_to_base() -> None:
    """attempt_only=True인데 extra_source가 없으면(방어적) BasePipeline으로 폴백한다."""
    from evaluator.harness import BasePipeline
    with patch("store.s3_code.download_best_pipeline") as mock_download:
        pipeline = _load_pipeline("s4e1", extra_source=None, attempt_only=True)
    mock_download.assert_not_called()
    assert isinstance(pipeline, BasePipeline)


def test_load_pipeline_attempt_only_preserves_class_attributes() -> None:
    """훅이 참조하는 클래스 속성(예: s6e7의 _ordinal_orders)이 살아있어야 한다.

    이전엔 attempt_only가 훅 메서드만 type(...)으로 새 클래스에 옮겨 붙여 클래스
    속성이 소실됐다 — 평가는 통과(runner.py는 실제 Patch() 인스턴스를 사용)하고
    submit만 AttributeError로 크래시하는 불일치가 있었다(s6e7 실제 프로덕션 실패).
    """
    source = (
        "class Patch:\n"
        "    _ordinal_orders = {'a': ['low', 'high']}\n"
        "    def build_model(self, params, ctx):\n"
        "        return self._ordinal_orders['a']\n"
    )
    with patch("store.s3_code.download_best_pipeline") as mock_download:
        pipeline = _load_pipeline("s4e1", extra_source=source, attempt_only=True)
    mock_download.assert_not_called()
    assert pipeline.build_model({}, None) == ["low", "high"]


# ---------------------------------------------------------------------------
# _bagged_predict seed bagging
# ---------------------------------------------------------------------------

def _bagging_ctx():
    from evaluator.harness import PipelineContext
    return PipelineContext(target_col="y", metric="auc", n_splits=5, seed=42, is_classification=True)


def test_bagged_predict_calls_build_model_once_per_seed() -> None:
    ctx = _bagging_ctx()
    pipeline = MagicMock()
    model = MagicMock()
    model.predict.return_value = np.array([1.0, 2.0])
    pipeline.build_model.return_value = model

    _bagged_predict(
        pipeline, {"a": 1}, np.zeros((2, 1)), np.zeros(2), np.zeros((2, 1)),
        ctx, "regression_error", bag_seeds=[1, 2, 3],
    )
    assert pipeline.build_model.call_count == 3
    used_seeds = [c.args[1].seed for c in pipeline.build_model.call_args_list]
    assert used_seeds == [1, 2, 3]


def test_bagged_predict_averages_predictions() -> None:
    ctx = _bagging_ctx()
    pipeline = MagicMock()
    model = MagicMock()
    model.predict.side_effect = [np.array([1.0, 3.0]), np.array([3.0, 5.0])]
    pipeline.build_model.return_value = model

    result = _bagged_predict(
        pipeline, {}, np.zeros((2, 1)), np.zeros(2), np.zeros((2, 1)),
        ctx, "regression_error", bag_seeds=[1, 2],
    )
    assert list(result) == [2.0, 4.0]


def test_bagged_predict_uses_binary_proba_for_classification_metric() -> None:
    ctx = _bagging_ctx()
    pipeline = MagicMock()
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4]])
    pipeline.build_model.return_value = model

    result = _bagged_predict(
        pipeline, {}, np.zeros((2, 1)), np.zeros(2), np.zeros((2, 1)),
        ctx, "binary_proba", bag_seeds=[1],
    )
    assert list(result) == [0.8, 0.4]


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


# ---------------------------------------------------------------------------
# _dummy_target_value
# ---------------------------------------------------------------------------
# 타입만 맞춘 placeholder(예: 0)는 Patch가 타깃을 exhaustive 매핑(replace_strict without
# default)으로 인코딩할 때 매핑에 없는 값이라 크래시한다(s5e7 실측). 더미값은 반드시
# train에 실재하는 값이어야 어떤 인코딩 로직과도 호환된다.

def test_dummy_target_value_returns_real_train_category() -> None:
    train = pl.DataFrame({"x": [1, 2, 3], "y": ["Extrovert", "Introvert", "Extrovert"]})
    result = _dummy_target_value(train, "y")
    assert result in ("Extrovert", "Introvert")


def test_dummy_target_value_is_not_a_synthetic_placeholder() -> None:
    """실제 카테고리에 '0'이 없는 데이터셋이면 결과도 '0'이면 안 된다 — 과거 버그 재현 방지."""
    train = pl.DataFrame({"x": [1, 2], "y": ["Extrovert", "Introvert"]})
    result = _dummy_target_value(train, "y")
    assert result != "0"
    assert result != 0


def test_dummy_target_value_works_for_numeric_target() -> None:
    train = pl.DataFrame({"x": [1, 2, 3], "y": [10.5, 20.5, 30.5]})
    result = _dummy_target_value(train, "y")
    assert result == 10.5
