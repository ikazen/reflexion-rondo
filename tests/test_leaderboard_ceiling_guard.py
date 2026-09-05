"""불가능 점수(cv > 세계 1위 LB) 즉시 격리 가드 (#288).

CV가 test 정확도의 불편추정치라면 리더보드 스냅샷(raw.leaderboard_snapshot)의 세계
1위 점수를 넘는 건 산술적으로 불가능하다 — s4e11 twin 중복(#228/#287) 실사고를
훨씬 싸고 빠르게 잡는 신호다. 두 지점에서 쓰인다: (1) run_attempt_core가 attempt
평가 직후 label='error'로 즉시 격리, (2) cycle/promotion.confirm_and_measure가
confirm 게이트에서 동일 검사로 확정을 막는다(1이 놓친 경우의 방어선).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl

from agents.strategist import StrategyDecision
from cycle.promotion import ConfirmResult, leaderboard_ceiling_violation
from cycle.run import CycleConfig, run_attempt_core
from cycle.stagnation import StagnationSignal
from runtime.isolate import IsolatedResult


def _snapshot_conn(scores: list[float], metric_sign: int = 1) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (scores, metric_sign)
    return conn


def test_ceiling_violation_flags_cv_above_world_best():
    conn = _snapshot_conn([0.90, 0.92, 0.94488], metric_sign=1)
    reason = leaderboard_ceiling_violation(conn, "playground-series-s4e11", 0.96789)
    assert reason is not None
    assert reason.startswith("cv_exceeds_world_best")


def test_ceiling_violation_none_when_within_range():
    conn = _snapshot_conn([0.90, 0.92, 0.94488], metric_sign=1)
    assert leaderboard_ceiling_violation(conn, "playground-series-s4e11", 0.93) is None


def test_ceiling_violation_at_exact_world_best_is_not_a_violation():
    """세계 1위와 정확히 같은 점수(그 팀 자신을 재현하는 등)는 초과가 아니다."""
    conn = _snapshot_conn([0.90, 0.94488], metric_sign=1)
    assert leaderboard_ceiling_violation(conn, "playground-series-s4e11", 0.94488) is None


def test_ceiling_violation_handles_negative_metric_sign():
    """rmse처럼 낮을수록 좋은 지표(metric_sign=-1)는 세계 1위가 min(scores)다."""
    conn = _snapshot_conn([11.5, 12.0, 13.0], metric_sign=-1)
    assert leaderboard_ceiling_violation(conn, "playground-series-s5e4", 11.0) is not None
    assert leaderboard_ceiling_violation(conn, "playground-series-s5e4", 11.6) is None


def test_ceiling_violation_none_when_no_snapshot():
    """스냅샷 없는 대회는 판정 불가로 None — 미탐지가 오탐(정상 대회를 멈춤)보다 안전."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    assert leaderboard_ceiling_violation(conn, "playground-series-new", 0.99) is None


def test_ceiling_violation_none_when_scores_empty():
    conn = _snapshot_conn([], metric_sign=1)
    assert leaderboard_ceiling_violation(conn, "playground-series-s4e11", 0.99) is None


# --- confirm_and_measure 게이트 (두 번째 방어선) ---

def test_confirm_and_measure_rejects_ceiling_violation_without_cross_seed_eval():
    cache = MagicMock()
    cache.get_memo.return_value = None
    conn = _snapshot_conn([0.90, 0.94488], metric_sign=1)

    with patch("cycle.promotion._cross_seed_confirm") as mock_cross_seed:
        from cycle.promotion import confirm_and_measure

        result = confirm_and_measure(
            source="class Patch:\n    pass\n",
            best_source=None,
            train90=pl.DataFrame({"x": [1, 2], "y": [0, 1]}),
            holdout10=None,
            target_col="y",
            metric="accuracy",
            n_splits=5,
            seed=42,
            is_classification=True,
            confirm_seeds=[7, 101, 137],
            cache=cache,
            competition_id="playground-series-s4e11",
            candidate_cv=0.96789,
            candidate_fold_scores=[0.967, 0.968, 0.969],
            conn=conn,
        )

    mock_cross_seed.assert_not_called()
    assert result.confirmed is False
    cache.put_memo.assert_called_once()


def test_confirm_and_measure_ignores_ceiling_check_when_conn_not_given():
    """conn을 안 주면(cache 전용 duck-type 테스트 더블 등) 리더보드 조회 자체를
    시도하지 않는다 — cache는 4개 메서드(get/put memo/baseline)만 구현하면 되고
    conn 속성을 가질 필요가 없다는 기존 계약(_FakeCache)을 그대로 유지한다."""
    cache = MagicMock(spec=["get_memo", "put_memo", "get_baseline", "put_baseline"])
    cache.get_memo.return_value = None

    with patch("cycle.promotion._cross_seed_confirm", return_value=(True, {})) as mock_cross_seed:
        from cycle.promotion import confirm_and_measure

        result = confirm_and_measure(
            source="class Patch:\n    pass\n",
            best_source=None,
            train90=pl.DataFrame({"x": [1, 2], "y": [0, 1]}),
            holdout10=None,
            target_col="y",
            metric="accuracy",
            n_splits=5,
            seed=42,
            is_classification=True,
            confirm_seeds=[7, 101, 137],
            cache=cache,
            competition_id="playground-series-s4e11",
            candidate_cv=0.96789,
            candidate_fold_scores=[0.967, 0.968, 0.969],
        )

    mock_cross_seed.assert_called_once()
    assert result.confirmed is True


def test_confirm_and_measure_proceeds_normally_when_within_ceiling():
    cache = MagicMock()
    cache.get_memo.return_value = None
    cache.get_baseline.return_value = None
    conn = _snapshot_conn([0.90, 0.94488], metric_sign=1)

    with patch(
        "cycle.promotion._cross_seed_confirm", return_value=(True, {}),
    ) as mock_cross_seed:
        from cycle.promotion import confirm_and_measure

        result = confirm_and_measure(
            source="class Patch:\n    pass\n",
            best_source=None,
            train90=pl.DataFrame({"x": [1, 2], "y": [0, 1]}),
            holdout10=None,
            target_col="y",
            metric="accuracy",
            n_splits=5,
            seed=42,
            is_classification=True,
            confirm_seeds=[7, 101, 137],
            cache=cache,
            competition_id="playground-series-s4e11",
            candidate_cv=0.93,
            candidate_fold_scores=[0.929, 0.930, 0.931],
            conn=conn,
        )

    mock_cross_seed.assert_called_once()
    assert result.confirmed is True


# --- run_attempt_core 즉시 격리 (첫 번째 방어선) ---

def _config() -> CycleConfig:
    return CycleConfig(
        competition_id="playground-series-s4e11",
        train=pl.DataFrame({"f": [1.0, 2.0, 3.0], "target": [0, 1, 0]}),
        target_col="target",
        metric="accuracy",
        stage="reflexion",
        eda_card="n_rows=3",
    )


def test_run_attempt_core_isolates_cv_exceeding_world_best():
    iso = IsolatedResult(
        cv_score=0.96789,
        cv_fold_var=0.0005,
        fold_scores=[0.967, 0.968, 0.969],
        label="jump",
        gain_vs_best=0.03,
        error_trace=None,
    )
    conn = MagicMock()
    with (
        patch("cycle.run.detect_stagnation",
              return_value=StagnationSignal(False, 0, (), 0)),
        patch("cycle.run.get_action_prior", return_value={}),
        patch("cycle.run.strategize", return_value=StrategyDecision(
            hypothesis="h", action_type="model_swap", reflection_ids=[])),
        patch("cycle.run.top_error_pitfalls", return_value=[]),
        patch("cycle.run.generate_code", return_value="source"),
        patch("cycle.run.validate_patch", return_value=[]),
        patch("cycle.run.eval_isolated", return_value=iso),
        patch("cycle.run.leaderboard_ceiling_violation",
              return_value="cv_exceeds_world_best: cv=0.967890 > world_best=0.944880 (metric_sign=1)"),
        patch("cycle.run.is_significant_gain") as mock_sig,
        patch("cycle.run._dynamic_eda_context", return_value=""),
        patch("cycle.run._load_best_pipeline", return_value="prev code"),
        patch("cycle.run._prev_best_params", return_value=None),
        patch("cycle.run._prev_best_fold_scores", return_value=None),
        patch("cycle.run._save_code", return_value="s3://code"),
        patch("cycle.run.insert_attempt") as mock_insert,
        patch("cycle.run.update_bandit") as mock_bandit,
    ):
        data = run_attempt_core(
            conn, _config(), lessons=[], prev_best_cv=0.89,
            defer_promotion=True,
        )

    assert data.label == "error"
    row = mock_insert.call_args[0][1]
    assert row["label"] == "error"
    assert row["error_signature"] == "cv_exceeds_world_best: cv=<val> > world_best=<val> (metric_sign=<val>)"
    mock_sig.assert_not_called()
    assert mock_bandit.call_args.kwargs["label"] == "error"


def test_run_attempt_core_untouched_when_within_ceiling():
    iso = IsolatedResult(
        cv_score=0.93, cv_fold_var=0.0005, fold_scores=[0.929, 0.930, 0.931],
        label="neutral", gain_vs_best=0.005, error_trace=None,
    )
    conn = MagicMock()
    with (
        patch("cycle.run.detect_stagnation",
              return_value=StagnationSignal(False, 0, (), 0)),
        patch("cycle.run.get_action_prior", return_value={}),
        patch("cycle.run.strategize", return_value=StrategyDecision(
            hypothesis="h", action_type="model_swap", reflection_ids=[])),
        patch("cycle.run.top_error_pitfalls", return_value=[]),
        patch("cycle.run.generate_code", return_value="source"),
        patch("cycle.run.validate_patch", return_value=[]),
        patch("cycle.run.eval_isolated", return_value=iso),
        patch("cycle.run.leaderboard_ceiling_violation", return_value=None) as mock_ceiling,
        patch("cycle.run.is_significant_gain", return_value=False),
        patch("cycle.run._dynamic_eda_context", return_value=""),
        patch("cycle.run._load_best_pipeline", return_value="prev code"),
        patch("cycle.run._prev_best_params", return_value=None),
        patch("cycle.run._prev_best_fold_scores", return_value=None),
        patch("cycle.run._save_code", return_value="s3://code"),
        patch("cycle.run.insert_attempt") as mock_insert,
        patch("cycle.run.update_bandit"),
    ):
        data = run_attempt_core(
            conn, _config(), lessons=[], prev_best_cv=0.89,
            defer_promotion=True,
        )

    mock_ceiling.assert_called_once()
    assert data.label == "neutral"
    assert mock_insert.call_args[0][1]["label"] == "neutral"
