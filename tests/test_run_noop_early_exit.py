"""run_attempt_core의 fold-1 조기 중단(#339) 재시도 로직을 검증한다.

evaluator/harness.py의 evaluate_pipeline이 fold-1 점수가 confirmed baseline과
비트 단위로 같으면(is_noop_tie) 나머지 fold를 건너뛰고 IsolatedResult.noop_early_exit=True로
반환한다. run_attempt_core는 예산이 남아 있으면 그 시점에 다른 후보로 1회 재시도한다 —
결과가 이미 유효한 tie이므로 재시도가 실패해도 잃을 게 없다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl

from agents.strategist import StrategyDecision
from cycle.run import CycleConfig, run_attempt_core
from cycle.stagnation import StagnationSignal
from runtime.isolate import DEFAULT_CPU_BUDGET_SECS, IsolatedResult


def _config() -> CycleConfig:
    return CycleConfig(
        competition_id="s4e1",
        train=pl.DataFrame({"f": [1.0, 2.0, 3.0], "target": [0, 1, 0]}),
        target_col="target",
        metric="auc",
        stage="reflexion",
        eda_card="n_rows=3",
    )


def _noop_tie(peak_cpu_sec: float, cv_score: float = 0.9) -> IsolatedResult:
    return IsolatedResult(
        cv_score=cv_score, cv_fold_var=None, fold_scores=None, label="neutral",
        gain_vs_best=0.0, gain_vs_best_relative=0.0, error_trace=None,
        is_noop_tie=True, noop_early_exit=True, peak_cpu_sec=peak_cpu_sec,
    )


def _run(eval_side_effect, generate_code_mock=None, validate_patch_mock=None):
    conn = MagicMock()
    generate_code_mock = generate_code_mock or MagicMock(return_value="source")
    validate_patch_mock = validate_patch_mock or MagicMock(return_value=[])
    with (
        patch("cycle.run.detect_stagnation",
              return_value=StagnationSignal(False, 0, (), 0)),
        patch("cycle.run.get_action_prior", return_value={}),
        patch("cycle.run.strategize", return_value=StrategyDecision(
            hypothesis="h", action_type="hyperparam_search", reflection_ids=[])),
        patch("cycle.run.top_error_pitfalls", return_value=[]),
        patch("cycle.run.generate_code", generate_code_mock),
        patch("cycle.run.validate_patch", validate_patch_mock),
        patch("cycle.run.eval_isolated", side_effect=eval_side_effect) as mock_eval,
        patch("cycle.run.leaderboard_ceiling_violation", return_value=None),
        patch("cycle.run.is_significant_gain", return_value=False),
        patch("cycle.run._dynamic_eda_context", return_value=""),
        patch("cycle.run._load_best_pipeline", return_value="prev code"),
        patch("cycle.run._prev_best_params", return_value=None),
        patch("cycle.run._prev_best_fold_scores", return_value=[0.88, 0.89, 0.90]),
        patch("cycle.run._save_code", return_value="s3://code"),
        patch("cycle.run.insert_attempt") as mock_insert,
        patch("cycle.run.update_bandit"),
    ):
        data = run_attempt_core(
            conn, _config(), lessons=[], prev_best_cv=0.9,
            defer_promotion=True,
        )
    return data, mock_insert, mock_eval, generate_code_mock


def test_prev_best_fold_scores_passed_to_eval_isolated():
    """attempt당 1회 조회한 baseline fold_scores가 eval_isolated에 전달된다."""
    ok = IsolatedResult(
        cv_score=0.9, cv_fold_var=0.001, fold_scores=[0.89, 0.9, 0.91],
        label="neutral", gain_vs_best=0.01, error_trace=None, peak_cpu_sec=120.0,
    )
    _data, _mock_insert, mock_eval, _ = _run(eval_side_effect=[ok])
    assert mock_eval.call_args_list[0].kwargs["prev_best_fold_scores"] == [0.88, 0.89, 0.90]


def test_retry_on_noop_early_exit_when_budget_remains():
    """fold-1 조기 tie + 예산 잔여 → 다른 후보로 1회 재시도하고, 성공하면 그
    결과를 채택한다."""
    ok = IsolatedResult(
        cv_score=0.95, cv_fold_var=0.001, fold_scores=[0.94, 0.95, 0.96],
        label="neutral", gain_vs_best=0.05, error_trace=None, peak_cpu_sec=120.0,
    )
    generate_code_mock = MagicMock(return_value="source")
    data, _mock_insert, mock_eval, _ = _run(
        eval_side_effect=[_noop_tie(peak_cpu_sec=50.0), ok],
        generate_code_mock=generate_code_mock,
    )

    assert mock_eval.call_count == 2
    # 최초 codegen 1회 + noop-tie 재생성 1회 = 2회
    assert generate_code_mock.call_count == 2
    feedback = generate_code_mock.call_args_list[1].kwargs["error_feedback"]
    assert "fold-1" in feedback
    assert data.cv_score == 0.95
    assert data.is_noop_tie is False


def test_retry_uses_remaining_budget():
    """재시도 eval_isolated 호출의 cpu_budget_sec은 1회차가 쓰고 남은 만큼이다."""
    ok = IsolatedResult(
        cv_score=0.95, cv_fold_var=0.001, fold_scores=[0.94, 0.95, 0.96],
        label="jump", gain_vs_best=0.05, error_trace=None, peak_cpu_sec=120.0,
    )
    _data, _mock_insert, mock_eval, _ = _run(
        eval_side_effect=[_noop_tie(peak_cpu_sec=100.0), ok],
    )
    assert mock_eval.call_args_list[1].kwargs["cpu_budget_sec"] == DEFAULT_CPU_BUDGET_SECS - 100.0


def test_noop_early_exit_retry_also_ties_is_accepted():
    """재시도도 fold-1 tie면 그대로 채택한다(무한 재시도 없음, range(2) 상한)."""
    data, mock_insert, mock_eval, _ = _run(
        eval_side_effect=[_noop_tie(peak_cpu_sec=50.0), _noop_tie(peak_cpu_sec=40.0)],
    )
    assert mock_eval.call_count == 2
    assert data.is_noop_tie is True
    assert data.cv_score == 0.9
    row = mock_insert.call_args[0][1]
    assert row["cv_fold_var"] == 0.0  # None → or 0.0 (기존 error-attempt 관례와 동일)
    assert row["fold_scores"] is None


def test_no_retry_when_cpu_budget_exhausted():
    """1회차가 예산을 전부 태우고 tie였으면 재시도 없이 그대로 채택한다."""
    generate_code_mock = MagicMock(return_value="source")
    data, _mock_insert, mock_eval, _ = _run(
        eval_side_effect=[_noop_tie(peak_cpu_sec=DEFAULT_CPU_BUDGET_SECS)],
        generate_code_mock=generate_code_mock,
    )
    assert mock_eval.call_count == 1
    assert generate_code_mock.call_count == 1  # 재생성 없음
    assert data.is_noop_tie is True


def test_retry_static_validation_exhausted_keeps_tie_result():
    """#273과 동일한 정적검사 재시도 폭을 쓰되, 전부 실패해도 이미 확보한
    (유효한) tie 결과를 그대로 채택한다 — error로 격하하지 않는다."""
    validate_patch_mock = MagicMock(side_effect=[
        [],  # 최초 codegen 정적검사 통과
        ["pandas-only API (not on polars): groupby()"],
        ["pandas-only API (not on polars): iterrows()"],
        ["pandas-only API (not on polars): apply()"],
    ])
    data, _mock_insert, mock_eval, generate_code_mock = _run(
        eval_side_effect=[_noop_tie(peak_cpu_sec=50.0)],
        validate_patch_mock=validate_patch_mock,
    )
    assert mock_eval.call_count == 1  # 2회차 eval 없음(재생성이 정적검사를 못 넘김)
    assert data.is_noop_tie is True
    assert data.cv_score == 0.9
    assert data.label == "neutral"
