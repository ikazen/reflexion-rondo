"""bin/submit.py 정확성 버그 회귀 테스트.

(a) 자동 선택 경로는 confirmed 파이프라인(raw.pipelines)만 소스로 써야 한다.
(b) predict_proba 분기는 metric_class == "binary_proba" 기준이어야 한다.
(c) 제출 값 컬럼은 sample_submission.csv의 실제 컬럼명을 따라야 한다.
(d) model_spec/ensemble_spec의 params에 random_state가 있어도 bag_seed마다 다른
    모델을 학습해야 한다(#307).
MinIO best_pipeline.py는 raw.pipelines.pipeline_sha256과 대조해야 한다.
attempt_only 재구성은 Patch 인스턴스의 클래스 속성을 보존해야 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.submit import (
    _BAG_SEEDS,
    _bagged_predict,
    _dummy_target_value,
    _load_best_code,
    _load_pipeline,
    _submission_value_col,
    SUBMIT_FIT_TIMEOUT_SEC,
    generate_submission_csv,
    generate_submission_csv_isolated,
    main,
)



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


def test_auto_select_query_excludes_invalid_reason() -> None:
    """격리된(GH #96 타깃 누수 등, #99) pipeline은 자동 제출 후보에서 제외돼야 한다 —
    부풀려진 cv_score로 계속 "최고"로 뽑히면 안 된다."""
    conn = _conn_with(("code text", 0.91, "attempt-123", "abc123sha"))
    with patch("store.db.connect", return_value=conn):
        _load_best_code("s4e1", None)
    sql = conn.execute.call_args.args[0]
    assert "invalid_reason" in sql.lower()


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


def test_load_pipeline_default_path_preserves_class_attributes() -> None:
    """attempt_only가 아닌 기본(MinIO best_pipeline.py) 경로도 훅 밖 클래스 속성을
    보존해야 한다.

    회귀 테스트 — s5e10 프로덕션 실제 재현(2026-08-03): ensemble action_type
    attempt가 build_model 안에서 참조하는 nested class(`_EnsembleRegressor`)를
    hook 메서드만 옮겨 붙이는 옛 type(...) 방식이 통째로 소실시켜
    `AttributeError: 'BestPipeline' object has no attribute '_EnsembleRegressor'`로
    Kaggle 제출이 크래시했다. attempt_only 경로는 이미 PatchedPipeline으로 고쳐져
    있었으나 이 기본 경로는 별도 구현이라 같은 버그가 남아 있었다.
    """
    source = (
        "class Patch:\n"
        "    class _EnsembleRegressor:\n"
        "        def predict(self):\n"
        "            return 'ensembled'\n"
        "    def build_model(self, params, ctx):\n"
        "        return self._EnsembleRegressor()\n"
    )
    with patch("store.s3_code.download_best_pipeline", return_value=source):
        pipeline = _load_pipeline("s4e1", expected_sha256=None)
    assert pipeline.build_model({}, None).predict() == "ensembled"



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


# attempt_only + base_source — #80 회귀 테스트
#
# param_candidates만 오버라이드하는 attempt(하이퍼파라미터 탐색)를 base 없이
# 제출하면 build_model/preprocess 등 나머지 hook이 BasePipeline 기본값으로
# 떨어져 cv_score와 무관한(대개 훨씬 나쁜) 예측을 낸다 — s4e12 실사고(#80).

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


def test_replay_best_pipeline_excludes_invalid_reason() -> None:
    """격리된(GH #96 타깃 누수 등, #99) pipeline은 replay에서 건너뛰어야 한다 —
    그래야 bin/rebuild_best_pipeline.py 재구성과 제출 replay 폴백 둘 다 누수를 다시
    섞어넣지 않는다."""
    from cycle.materialize import replay_best_pipeline

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    replay_best_pipeline(conn, "s4e1")
    sql = conn.execute.call_args.args[0]
    assert "invalid_reason IS NULL" in sql


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


def test_replay_best_pipeline_stops_at_pipeline_id() -> None:
    """stop_at_pipeline_id를 주면 그 행까지(포함) 재생하고 멈춘다 — #254 백필이
    특정 승격 행 자신의 병합본을 재현하려면 필요하다(before_run_ts는 strict `<`라 불가)."""
    from cycle.materialize import replay_best_pipeline

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("pid-1", "2026-07-26T00:00:00", "class Patch:\n    def build_model(self, p, c):\n        return 1\n", None),
        ("pid-2", "2026-07-27T00:00:00", "class Patch:\n    def preprocess(self, tr, va, t, c):\n        return tr, va\n", None),
        ("pid-3", "2026-07-28T00:00:00", "class Patch:\n    def postprocess_predictions(self, p, c):\n        return p\n", None),
    ]
    best, _, count = replay_best_pipeline(conn, "s4e1", stop_at_pipeline_id="pid-2")
    assert count == 2
    assert "def build_model" in best and "def preprocess" in best
    assert "def postprocess_predictions" not in best


def test_promotion_chain_does_not_filter_invalid_reason() -> None:
    """백필은 격리된 행도 봐야 하므로 promotion_chain은 invalid_reason 필터를 안 건다."""
    from cycle.materialize import promotion_chain

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    promotion_chain(conn, "s4e1")
    sql = conn.execute.call_args.args[0]
    assert "invalid_reason IS NULL" not in sql  # 필터를 안 건다(컬럼으로는 반환)
    assert "materialized_origin" in sql


# cycle.materialize.load_base_snapshot — #89

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


def test_load_base_snapshot_trusts_backfilled_snapshot_via_materialized_sha256() -> None:
    """#254 백필 행은 materialized_code가 텍스트로는 승격 당시와 달라도(합성 규칙 변경)
    행동 재현이 검증됐다 — coalesce(materialized_sha256, pipeline_sha256)을 신뢰 해시로
    써서 그 스냅샷 자신의 sha와 대조(손상만 잡음)."""
    import hashlib
    from cycle.materialize import load_base_snapshot

    snapshot = "class Patch:\n    def build_model(self, p, c):\n        return 2\n"
    backfilled_sha = hashlib.sha256(snapshot.encode()).hexdigest()
    conn = MagicMock()
    # 쿼리가 coalesce(materialized_sha256, pipeline_sha256)을 두 번째 컬럼으로 반환
    conn.execute.return_value.fetchone.return_value = (snapshot, backfilled_sha)
    src, _ = load_base_snapshot(conn, "s4e1")
    assert src == snapshot
    sql = conn.execute.call_args.args[0]
    assert "coalesce(p.materialized_sha256, p.pipeline_sha256)" in sql


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



def _bagging_ctx():
    from evaluator.harness import PipelineContext
    return PipelineContext(target_col="y", metric="auc", n_splits=5, seed=42, is_classification=True)


def test_bagged_predict_calls_build_model_once_per_seed() -> None:
    ctx = _bagging_ctx()
    pipeline = MagicMock()
    pipeline.ensemble_spec.return_value = None
    pipeline.model_spec.return_value = None
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
    pipeline.ensemble_spec.return_value = None
    pipeline.model_spec.return_value = None
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
    pipeline.ensemble_spec.return_value = None
    pipeline.model_spec.return_value = None
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4]])
    pipeline.build_model.return_value = model

    result = _bagged_predict(
        pipeline, {}, np.zeros((2, 1)), np.zeros(2), np.zeros((2, 1)),
        ctx, "binary_proba", bag_seeds=[1],
    )
    assert list(result) == [0.8, 0.4]


def test_bagged_predict_routes_ensemble_spec_to_declarative_path() -> None:
    """#226: pipeline.ensemble_spec()이 정의돼 있으면 build_model이 아니라
    harness._fit_predict_ensemble로 가야 한다 — 이전엔 이 분기 자체가 없어
    확정된 ensemble이 제출 시점엔 조용히 build_model(params={})로 대체됐다."""
    ctx = _bagging_ctx()
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((60, 3))
    y_train = (X_train[:, 0] > 0).astype(float)
    X_test = rng.standard_normal((10, 3))

    pipeline = MagicMock()
    pipeline.ensemble_spec.return_value = {
        "members": [{"model": "hgb"}, {"model": "random_forest", "params": {"n_estimators": 10}}],
        "method": "weighted_average",
    }

    result = _bagged_predict(
        pipeline, {}, X_train, y_train, X_test, ctx, "binary_proba", bag_seeds=[1, 2],
    )
    assert result.shape == (10,)
    assert np.all(np.isfinite(result))
    # build_model은 ensemble_spec 경로에서 전혀 호출되면 안 된다 — 호출되면
    # harness가 아니라 다시 단일 모델로 새는 것.
    pipeline.build_model.assert_not_called()


def test_submission_value_col_uses_sample_second_column() -> None:
    assert _submission_value_col(["id", "target_prob"], "target") == "target_prob"


def test_submission_value_col_falls_back_when_missing() -> None:
    assert _submission_value_col(["id"], "target") == "target"



def _bagging_reg_ctx():
    from evaluator.harness import PipelineContext
    return PipelineContext(target_col="y", metric="rmse", n_splits=5, seed=42, is_classification=False)


def test_bagged_predict_strips_seed_from_model_spec_params() -> None:
    """#307: model_spec.params에 random_state가 박혀 있으면
    evaluator.models.build_registry_model이 ctx.seed로 채우지 못해 bag_seed가
    무시된다 — 벗기지 않으면 서로 다른 bag_seeds로 불러도 동일 모델이 나온다."""
    ctx = _bagging_reg_ctx()
    pipeline = MagicMock()
    pipeline.ensemble_spec.return_value = None
    pipeline.model_spec.return_value = {
        "model": "random_forest",
        "params": {"n_estimators": 5, "max_features": 0.5, "random_state": 999},
    }
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((60, 4))
    y_train = rng.standard_normal(60)
    X_test = rng.standard_normal((10, 4))

    preds_seed_1 = _bagged_predict(
        pipeline, {}, X_train, y_train, X_test, ctx, "regression_error", bag_seeds=[1],
    )
    preds_seed_2 = _bagged_predict(
        pipeline, {}, X_train, y_train, X_test, ctx, "regression_error", bag_seeds=[2],
    )
    assert not np.allclose(preds_seed_1, preds_seed_2)


def test_bagged_predict_strips_seed_from_ensemble_member_params() -> None:
    """#307과 동일 버그, ensemble_spec 멤버 params 경로."""
    ctx = _bagging_reg_ctx()
    pipeline = MagicMock()
    pipeline.ensemble_spec.return_value = {
        "members": [
            {"model": "random_forest", "params": {"n_estimators": 5, "max_features": 0.5, "random_state": 999}},
        ],
        "method": "weighted_average",
    }
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((60, 4))
    y_train = rng.standard_normal(60)
    X_test = rng.standard_normal((10, 4))

    preds_seed_1 = _bagged_predict(
        pipeline, {}, X_train, y_train, X_test, ctx, "regression_error", bag_seeds=[1],
    )
    preds_seed_2 = _bagged_predict(
        pipeline, {}, X_train, y_train, X_test, ctx, "regression_error", bag_seeds=[2],
    )
    assert not np.allclose(preds_seed_1, preds_seed_2)


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




def _generate(monkeypatch, tmp_path, *, full_data: bool, **comp_extra):
    """generate_submission_csv를 DB/MinIO 없이 돌린다 — load_train과 _bagged_predict 호출 인자를 돌려준다."""
    from evaluator.harness import BasePipeline

    comp = SimpleNamespace(
        COMPETITION_ID="playground-series-fake", TARGET="y", METRIC="rmse", IS_CLASSIFICATION=False,
        DROP_COLS=["id"], **comp_extra,
    )
    monkeypatch.setitem(sys.modules, "config.competitions.fake_sub", comp)
    frames = {
        "test.csv": pl.DataFrame({"id": [100, 101, 102], "x": [1.0, 2.0, 3.0]}),
        "sample_submission.csv": pl.DataFrame({"id": [100, 101, 102], "y": [0.0, 0.0, 0.0]}),
    }
    train = pl.DataFrame({"x": [float(i) for i in range(40)], "y": [float(i % 7) for i in range(40)]})
    load_train = MagicMock(return_value=train)
    bagged = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
    monkeypatch.setattr("bin.submit._load_best_code", lambda cid, aid: ("src", 1.5, "attempt-1234", None, None))
    monkeypatch.setattr("bin.submit._load_pipeline", lambda *a, **k: BasePipeline())
    monkeypatch.setattr("bin.submit._read_csv", lambda c, name: frames[name])
    monkeypatch.setattr("store.train_data.load_train", load_train)
    monkeypatch.setattr("bin.submit._bagged_predict", bagged)
    monkeypatch.setattr("bin.submit.RUNS_DIR", tmp_path)
    out, attempt_id, cv = generate_submission_csv("fake_sub", full_data=full_data)
    return load_train, bagged, out


@pytest.mark.parametrize("full_data", [False, True])
def test_generate_submission_csv_full_data_controls_the_row_cap(monkeypatch, tmp_path, full_data) -> None:
    """#355: full_data=True만 MAX_TRAIN_ROWS 축소를 끈다. 기본(False)은 attempt 평가와 같은 학습셋."""
    load_train, _, out = _generate(monkeypatch, tmp_path, full_data=full_data)
    assert load_train.call_args.kwargs == {"apply_row_cap": not full_data}
    assert pl.read_csv(out).to_dict(as_series=False) == {"id": [100, 101, 102], "y": [1.0, 2.0, 3.0]}


def test_generate_submission_csv_bag_seeds_come_from_the_competition_config(monkeypatch, tmp_path, capsys) -> None:
    _, bagged, _ = _generate(monkeypatch, tmp_path, full_data=True, SUBMIT_BAG_SEEDS=[7, 8])
    assert bagged.call_args.kwargs["bag_seeds"] == [7, 8]
    assert "submission fit: rows=40 full_data=True seeds=2 elapsed=" in capsys.readouterr().out


def test_generate_submission_csv_defaults_to_the_five_seed_bag(monkeypatch, tmp_path) -> None:
    _, bagged, _ = _generate(monkeypatch, tmp_path, full_data=False)
    assert bagged.call_args.kwargs["bag_seeds"] == _BAG_SEEDS


@pytest.mark.parametrize(("argv", "expected"), [
    (["--competition", "s5e4"], False),
    (["--competition", "s5e4", "--full-data"], True),
])
def test_cli_full_data_flag(monkeypatch, argv, expected) -> None:
    generate = MagicMock(return_value=(Path("out.csv"), "attempt-1234", 1.5))
    monkeypatch.setattr("bin.submit.generate_submission_csv", generate)
    monkeypatch.setattr(sys, "argv", ["bin.submit", *argv])
    main()
    assert generate.call_args.kwargs == {"full_data": expected}


def _completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


_FIT_OUTPUT = (
    "best attempt: a8e23c79  cv=13.03735\n"
    "submission fit: rows=797105 full_data=True seeds=1 elapsed=179s peak_rss=1502MB\n"
    "submission saved: /app/runs/submission_s5e4_20260925_045726.csv\n"
)


@pytest.mark.parametrize(("full_data", "flag_present"), [(True, True), (False, False)])
def test_isolated_generation_runs_bin_submit_in_a_bounded_subprocess(full_data, flag_present, capsys) -> None:
    """#355: fit은 promote task 프로세스 안이 아니라 wall 상한을 건 별도 프로세스에서 돈다."""
    with patch("bin.api._run_in_pgroup", return_value=_completed(stdout=_FIT_OUTPUT)) as run:
        path = generate_submission_csv_isolated("s5e4", "a8e23c79-full-id", full_data=full_data)
    cmd = run.call_args.args[0]
    assert cmd[:3] == [sys.executable, "-m", "bin.submit"]
    assert cmd[cmd.index("--competition") + 1] == "s5e4"
    assert cmd[cmd.index("--attempt-id") + 1] == "a8e23c79-full-id"
    assert ("--full-data" in cmd) is flag_present
    assert "--submit" not in cmd
    assert run.call_args.kwargs["timeout"] == SUBMIT_FIT_TIMEOUT_SEC
    assert path == Path("/app/runs/submission_s5e4_20260925_045726.csv")
    assert "submission fit: rows=797105" in capsys.readouterr().out


def test_isolated_generation_wall_limit_fits_inside_the_promote_task_timeout() -> None:
    """airflow-stack DAG의 promote execution_timeout(180분)보다 짧아야 promote가 상한에 죽지 않는다."""
    assert SUBMIT_FIT_TIMEOUT_SEC < 180 * 60


def test_isolated_generation_reports_the_stderr_tail_on_failure() -> None:
    failed = _completed(returncode=1, stderr="x" * 3000 + "ValueError: no confirmed pipeline")
    with patch("bin.api._run_in_pgroup", return_value=failed):
        with pytest.raises(RuntimeError, match=r"rc=1.*ValueError: no confirmed pipeline"):
            generate_submission_csv_isolated("s5e4", "a8e23c79", full_data=False)


def test_isolated_generation_requires_the_saved_path_line() -> None:
    with patch("bin.api._run_in_pgroup", return_value=_completed(stdout="best attempt: a8e23c79\n")):
        with pytest.raises(RuntimeError, match="no 'submission saved:' line"):
            generate_submission_csv_isolated("s5e4", "a8e23c79", full_data=False)


def test_isolated_generation_propagates_the_wall_timeout() -> None:
    import subprocess

    with patch("bin.api._run_in_pgroup", side_effect=subprocess.TimeoutExpired(cmd="bin.submit", timeout=9000)):
        with pytest.raises(subprocess.TimeoutExpired):
            generate_submission_csv_isolated("s5e4", "a8e23c79", full_data=True)
