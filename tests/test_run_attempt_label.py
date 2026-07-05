"""BON-267: run_attempt_core의 label 확정이 promotion과 동일한 paired 유의성
(is_significant_gain) 기준을 따르는지 검증한다.

harness(evaluator/harness.py)가 붙인 절대-마진 label은 잠정값일 뿐이며, run_attempt_core가
is_significant_gain 결과로 최종 jump/neutral을 재확정한다. 그 확정된 label이 insert_attempt
row와 update_bandit 호출에 그대로 반영되는지가 핵심 회귀 지점이다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl

from agents.strategist import StrategyDecision
from cycle.run import CycleConfig, run_attempt_core
from cycle.stagnation import StagnationSignal
from runtime.isolate import IsolatedResult


def _config() -> CycleConfig:
    return CycleConfig(
        competition_id="s4e1",
        train=pl.DataFrame({"f": [1.0, 2.0, 3.0], "target": [0, 1, 0]}),
        target_col="target",
        metric="auc",
        stage="reflexion",
        eda_card="n_rows=3",
    )


def _iso(
    label: str = "neutral",
    gain_vs_best: float | None = 0.01,
    cv_score: float | None = 0.9,
    error_trace: str | None = None,
) -> IsolatedResult:
    return IsolatedResult(
        cv_score=cv_score,
        cv_fold_var=0.0005,
        fold_scores=[0.89, 0.90, 0.91],
        label=label,
        gain_vs_best=gain_vs_best,
        error_trace=error_trace,
    )


def _run(iso_result: IsolatedResult, significant: bool):
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
        patch("cycle.run.eval_isolated", return_value=iso_result),
        patch("cycle.run.is_significant_gain", return_value=significant) as mock_sig,
        patch("cycle.run._dynamic_eda_context", return_value=""),
        patch("cycle.run._load_best_pipeline", return_value="prev code"),
        patch("cycle.run._prev_best_params", return_value=None),
        patch("cycle.run._prev_best_fold_scores", return_value=[0.88, 0.89, 0.90]),
        patch("cycle.run._save_code", return_value="s3://code"),
        patch("cycle.run.insert_attempt") as mock_insert,
        patch("cycle.run.update_bandit") as mock_bandit,
    ):
        data = run_attempt_core(
            conn, _config(), lessons=[], prev_best_cv=0.89,
            defer_promotion=True,
        )
    return data, mock_insert, mock_bandit, mock_sig


def test_significant_gain_promotes_label_to_jump():
    """harness가 neutral로 봤어도 paired 유의성(is_significant_gain)이 True면 jump로 확정."""
    data, mock_insert, mock_bandit, _ = _run(_iso(label="neutral", gain_vs_best=0.02), significant=True)

    assert data.label == "jump"
    row = mock_insert.call_args[0][1]
    assert row["label"] == "jump"
    assert mock_bandit.call_args.kwargs["label"] == "jump"


def test_harness_jump_is_demoted_when_not_significant():
    """harness 절대-마진 기준은 통과(label='jump')했지만 paired 검정은 미달 → neutral로 강등."""
    data, mock_insert, mock_bandit, _ = _run(_iso(label="jump", gain_vs_best=0.03), significant=False)

    assert data.label == "neutral"
    assert mock_insert.call_args[0][1]["label"] == "neutral"
    assert mock_bandit.call_args.kwargs["label"] == "neutral"


def test_non_significant_positive_gain_stays_neutral():
    data, mock_insert, _, _ = _run(_iso(label="neutral", gain_vs_best=0.005), significant=False)

    assert data.label == "neutral"
    assert mock_insert.call_args[0][1]["label"] == "neutral"


def test_regression_label_preserved_regardless_of_significance():
    """regression은 유의성 판정과 무관하게 그대로 유지된다 — jump 판정만 재확정 대상."""
    data, mock_insert, mock_bandit, _ = _run(_iso(label="regression", gain_vs_best=-0.05), significant=False)

    assert data.label == "regression"
    assert mock_insert.call_args[0][1]["label"] == "regression"
    assert mock_bandit.call_args.kwargs["label"] == "regression"


def test_error_trace_forces_error_label_and_skips_significance_check():
    """error_trace가 있으면 is_significant_gain을 호출하지 않고 label='error'로 확정."""
    data, mock_insert, mock_bandit, mock_sig = _run(
        _iso(label="regression", gain_vs_best=None, cv_score=None, error_trace="NameError: x"),
        significant=False,
    )

    assert data.label == "error"
    assert mock_insert.call_args[0][1]["label"] == "error"
    assert mock_bandit.call_args.kwargs["label"] == "error"
    mock_sig.assert_not_called()
