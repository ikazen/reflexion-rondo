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
    _fit_full_train,
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
        source, cv_score, aid, sha, run_ts = _load_best_code("s4e1", None)
    sql = conn.execute.call_args.args[0]
    assert "raw.pipelines" in sql
    assert "raw.attempts" not in sql
    assert source == "code text"
    assert cv_score == 0.91
    assert aid == "attempt-123"
    assert sha == "abc123sha"
    assert run_ts is None


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
    import datetime
    run_ts = datetime.datetime(2026, 7, 28, 3, 15, 1)
    conn = _conn_with(("path/to/code.py", 0.80, "attempt-999", run_ts))
    with patch("store.db.connect", return_value=conn), \
         patch("store.s3_code.download", return_value="header\n" + ("# " + "-" * 60) + "\nsource here") as mock_download:
        source, cv_score, aid, sha, resolved_run_ts = _load_best_code("s4e1", "attempt-999")
    sql = conn.execute.call_args.args[0]
    assert "raw.attempts" in sql
    mock_download.assert_called_once_with("path/to/code.py")
    assert source == "source here"
    assert cv_score == 0.80
    assert aid == "attempt-999"
    assert resolved_run_ts == run_ts


def test_explicit_attempt_id_filters_by_competition() -> None:
    """--attempt-id는 competition_id로도 필터해야 한다 — prefix만으로는 다른 대회 attempt와 충돌할 수 있다."""
    import datetime
    conn = _conn_with(("path/to/code.py", 0.80, "attempt-999", datetime.datetime(2026, 7, 28)))
    with patch("store.db.connect", return_value=conn), \
         patch("store.s3_code.download", return_value="src"):
        _load_best_code("s4e1", "attempt-999")
    sql = conn.execute.call_args.args[0]
    params = conn.execute.call_args.args[1]
    assert "competition_id = %s" in sql
    assert "s4e1" in params


def test_explicit_attempt_id_skips_hash_verification() -> None:
    """--attempt-id 명시 경로는 raw.pipelines 대조 해시가 없어 pipeline_sha256=None (의도된 escape hatch)."""
    import datetime
    conn = _conn_with(("path/to/code.py", 0.80, "attempt-999", datetime.datetime(2026, 7, 28)))
    with patch("store.db.connect", return_value=conn), \
         patch("store.s3_code.download", return_value="src"):
        *_, sha, _ = _load_best_code("s4e1", "attempt-999")
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
# attempt_only + base_source — #80 회귀 테스트
#
# param_candidates만 오버라이드하는 attempt(하이퍼파라미터 탐색)를 base 없이
# 제출하면 build_model/preprocess 등 나머지 hook이 BasePipeline 기본값으로
# 떨어져 cv_score와 무관한(대개 훨씬 나쁜) 예측을 낸다 — s4e12 실사고(#80).
# ---------------------------------------------------------------------------

def test_load_pipeline_attempt_only_with_base_source_inherits_unoverridden_hooks() -> None:
    """base_source가 있으면 patch가 오버라이드하지 않은 hook은 base에서 온다."""
    base_source = (
        "class Patch:\n"
        "    def build_model(self, params, ctx):\n"
        "        return 'base-model'\n"
        "    def postprocess_predictions(self, preds, ctx):\n"
        "        return 'base-postprocess'\n"
    )
    attempt_source = (
        "class Patch:\n"
        "    def param_candidates(self, ctx):\n"
        "        return [{'lr': 0.1}]\n"
    )
    with patch("store.s3_code.download_best_pipeline") as mock_download:
        pipeline = _load_pipeline(
            "s4e1", extra_source=attempt_source, attempt_only=True, base_source=base_source,
        )
    mock_download.assert_not_called()  # base_source는 raw.pipelines 재생분 — MinIO 조회는 여전히 없다
    assert pipeline.param_candidates(None) == [{"lr": 0.1}]  # patch가 이긴다
    assert pipeline.build_model({}, None) == "base-model"  # base로 상속
    assert pipeline.postprocess_predictions(None, None) == "base-postprocess"  # base로 상속


def test_load_pipeline_attempt_only_without_base_source_falls_back_to_base_pipeline() -> None:
    """base_source가 없으면(재생 이력 없음) 기존과 동일하게 BasePipeline()에 patch만 적용."""
    from evaluator.harness import BasePipeline
    attempt_source = (
        "class Patch:\n"
        "    def param_candidates(self, ctx):\n"
        "        return [{'lr': 0.1}]\n"
    )
    pipeline = _load_pipeline(
        "s4e1", extra_source=attempt_source, attempt_only=True, base_source=None,
    )
    assert pipeline.param_candidates(None) == [{"lr": 0.1}]
    assert isinstance(pipeline.base, BasePipeline)
    assert type(pipeline.base) is BasePipeline


def test_load_pipeline_attempt_only_base_source_without_patch_falls_back() -> None:
    """base_source에 Patch 클래스가 없으면(방어적) BasePipeline()으로 폴백한다."""
    pipeline = _load_pipeline(
        "s4e1",
        extra_source="class Patch:\n    def param_candidates(self, ctx):\n        return []\n",
        attempt_only=True,
        base_source="x = 1\n",
    )
    from evaluator.harness import BasePipeline
    assert type(pipeline.base) is BasePipeline


# ---------------------------------------------------------------------------
# cycle.materialize.replay_best_pipeline
# ---------------------------------------------------------------------------

def test_replay_best_pipeline_folds_history_in_run_ts_order() -> None:
    from cycle.materialize import replay_best_pipeline

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("pid-1", "2026-07-26T00:00:00", "class Patch:\n    def build_model(self, p, c):\n        return 1\n", "sha1"),
        ("pid-2", "2026-07-27T00:00:00", "class Patch:\n    def preprocess(self, tr, va, t, c):\n        return tr, va\n", "sha2"),
    ]
    best, last_sha, count = replay_best_pipeline(conn, "s4e1")
    assert count == 2
    assert last_sha == "sha2"
    assert "def build_model" in best
    assert "def preprocess" in best


def test_replay_best_pipeline_filters_by_before_run_ts() -> None:
    """before_run_ts를 주면 SQL에 그 조건이 들어가야 한다 — 재생 결과가 attempt 평가
    시점 이후 승격분을 섞으면 안 되므로."""
    from cycle.materialize import replay_best_pipeline

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    import datetime
    cutoff = datetime.datetime(2026, 7, 28)
    replay_best_pipeline(conn, "s4e1", before_run_ts=cutoff)
    sql = conn.execute.call_args.args[0]
    params = conn.execute.call_args.args[1]
    assert "run_ts <" in sql
    assert cutoff in params


def test_replay_best_pipeline_no_history_returns_none() -> None:
    from cycle.materialize import replay_best_pipeline

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    best, last_sha, count = replay_best_pipeline(conn, "s4e1")
    assert (best, last_sha, count) == (None, None, 0)


def test_replay_best_pipeline_strict_sha_raises_on_mismatch() -> None:
    """strict_sha면 재생본이 승격 당시 병합본과 다를 때 진행하지 않는다(#89) —
    평가와 다른 base로 제출하면 크래시(s5e4)하거나 조용히 열화된 예측이
    제출된다(s5e10)."""
    from cycle.materialize import replay_best_pipeline

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("pid-1", "2026-07-26T00:00:00",
         "class Patch:\n    def build_model(self, p, c):\n        return 1\n",
         "not-the-real-sha"),
    ]
    with pytest.raises(RuntimeError, match="sha256"):
        replay_best_pipeline(conn, "s4e1", strict_sha=True)


# ---------------------------------------------------------------------------
# cycle.materialize.load_base_snapshot — #89
# ---------------------------------------------------------------------------

def test_load_base_snapshot_prefers_materialized_code() -> None:
    import hashlib
    from cycle.materialize import load_base_snapshot

    snapshot = "class Patch:\n    def build_model(self, p, c):\n        return 1\n"
    sha = hashlib.sha256(snapshot.encode()).hexdigest()
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (snapshot, sha)
    src, origin = load_base_snapshot(conn, "s4e1")
    assert src == snapshot
    assert "snapshot" in origin


def test_load_base_snapshot_raises_on_corrupt_snapshot() -> None:
    from cycle.materialize import load_base_snapshot

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("class Patch:\n    pass\n", "0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        load_base_snapshot(conn, "s4e1")


def test_load_base_snapshot_no_history_returns_none() -> None:
    from cycle.materialize import load_base_snapshot

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    src, _ = load_base_snapshot(conn, "s4e1")
    assert src is None


def test_load_base_snapshot_falls_back_to_strict_replay_when_snapshot_missing() -> None:
    """스냅샷 없는 과거 이력은 replay 폴백을 쓰되 반드시 strict_sha로 재생한다."""
    from cycle import materialize

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (None, "sha-old")
    with patch.object(
        materialize, "replay_best_pipeline", return_value=("merged", "sha-old", 3)
    ) as mock_replay:
        src, origin = materialize.load_base_snapshot(conn, "s4e1")
    assert src == "merged"
    assert mock_replay.call_args.kwargs["strict_sha"] is True


def test_load_base_snapshot_passes_before_run_ts() -> None:
    """attempt 평가 시점 이후 승격분이 base에 섞이면 안 된다 — SQL 컷오프 확인."""
    import datetime
    from cycle.materialize import load_base_snapshot

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    cutoff = datetime.datetime(2026, 7, 28)
    load_base_snapshot(conn, "s4e1", before_run_ts=cutoff)
    sql = conn.execute.call_args.args[0]
    params = conn.execute.call_args.args[1]
    assert "run_ts <" in sql
    assert cutoff in params


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
# _fit_full_train — 생성자 early-stopping 파라미터 폴백 (#71)
# ---------------------------------------------------------------------------

def test_bagged_predict_retries_without_early_stopping_params() -> None:
    """eval_set 없이 fit이 죽으면 early-stopping 키를 벗긴 params로 재시도한다."""
    ctx = _bagging_ctx()
    pipeline = MagicMock()
    failing_model = MagicMock()
    failing_model.fit.side_effect = ValueError(
        "For early stopping, at least one dataset and eval metric is required for evaluation"
    )
    working_model = MagicMock()
    working_model.predict.return_value = np.array([1.0, 2.0])
    pipeline.build_model.side_effect = [failing_model, working_model]

    params = {"n_estimators": 100, "early_stopping_rounds": 50, "learning_rate": 0.05}
    model = _fit_full_train(pipeline, params, ctx, np.zeros((2, 1)), np.zeros(2))

    assert model is working_model
    assert pipeline.build_model.call_count == 2
    retry_params = pipeline.build_model.call_args_list[1].args[0]
    assert "early_stopping_rounds" not in retry_params
    assert retry_params == {"n_estimators": 100, "learning_rate": 0.05}


def test_bagged_predict_reraises_when_retry_also_fails() -> None:
    """벗긴 params로도 fit이 죽으면 예외를 그대로 전파한다 — 조용한 실패 금지."""
    ctx = _bagging_ctx()
    pipeline = MagicMock()
    model = MagicMock()
    model.fit.side_effect = ValueError("still broken")
    pipeline.build_model.return_value = model

    params = {"early_stopping_rounds": 50}
    with pytest.raises(ValueError, match="still broken"):
        _fit_full_train(pipeline, params, ctx, np.zeros((2, 1)), np.zeros(2))


def test_bagged_predict_does_not_strip_when_fit_succeeds() -> None:
    """정상 fit이면 재시도 없이 원래 params 그대로 build_model 1회만 호출한다

    (HistGradientBoosting/CatBoost처럼 eval_set 없이도 자체 검증 분할로 동작하는
    estimator의 early-stopping 설정을 조용히 바꾸지 않기 위한 보장).
    """
    ctx = _bagging_ctx()
    pipeline = MagicMock()
    model = MagicMock()
    model.predict.return_value = np.array([1.0, 2.0])
    pipeline.build_model.return_value = model

    params = {"early_stopping_rounds": 50, "learning_rate": 0.05}
    result = _fit_full_train(pipeline, params, ctx, np.zeros((2, 1)), np.zeros(2))

    assert result is model
    assert pipeline.build_model.call_count == 1
    assert pipeline.build_model.call_args.args[0] == params


def test_fit_full_train_no_early_stopping_keys_reraises_immediately() -> None:
    """params에 애초에 early-stopping 키가 없으면 재시도할 게 없어 원래 예외를 바로 올린다."""
    ctx = _bagging_ctx()
    pipeline = MagicMock()
    model = MagicMock()
    model.fit.side_effect = RuntimeError("unrelated failure")
    pipeline.build_model.return_value = model

    params = {"learning_rate": 0.05}
    with pytest.raises(RuntimeError, match="unrelated failure"):
        _fit_full_train(pipeline, params, ctx, np.zeros((2, 1)), np.zeros(2))
    assert pipeline.build_model.call_count == 1


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
