"""run_attempt_core의 label 확정이 promotion과 동일한 paired 유의성
(is_significant_gain) 기준을 따르는지 검증한다.

harness(evaluator/harness.py)가 붙인 절대-마진 label은 잠정값일 뿐이며, run_attempt_core가
is_significant_gain 결과로 최종 jump/neutral을 재확정한다. 그 확정된 label이 insert_attempt
row와 update_bandit 호출에 그대로 반영되는지가 핵심 회귀 지점이다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl

from agents.strategist import StrategyDecision
from cycle.promotion import ConfirmResult
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
        patch("cycle.run.leaderboard_ceiling_violation", return_value=None),
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


# confirm 결과를 bandit/lesson 보상에 반영 (#164)
#
# defer_promotion=False(직접모드)에서만 run_attempt_core 자신이 confirm_and_measure를
# 돌린다. 로컬 유의성(is_significant_gain)만으로 확정된 label="jump"가 나중에
# confirm(cross-seed 재현/holdout)에서 거부되면, 그 사실이 update_bandit과
# _AttemptData.reward_label(→ _do_reflect의 lesson)에 반영돼야 자기강화 루프가
# 안 생긴다. raw.attempts.label(insert_attempt row)은 attempt 시점의 잠정 판정을
# 그대로 보존한다 — 여기서 다운그레이드하는 건 하류 학습 신호뿐이다.

def _run_deferred(iso_result: IsolatedResult, confirm_result: ConfirmResult | None):
    conn = MagicMock()
    # confirm.confirmed=True 경로는 fingerprint를 조회해 raw.pipelines에 insert한다
    # (cycle/run.py:~710) — dict로 고정해 json.loads(MagicMock) TypeError를 피한다.
    conn.execute.return_value.fetchone.return_value = ({},)
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
        patch("cycle.run.leaderboard_ceiling_violation", return_value=None),
        patch("cycle.run.is_significant_gain", return_value=True),
        patch("cycle.run.confirm_and_measure", return_value=confirm_result),
        patch("cycle.run._dynamic_eda_context", return_value=""),
        patch("cycle.run._load_best_pipeline", return_value="prev code"),
        patch("cycle.run._prev_best_params", return_value=None),
        patch("cycle.run._prev_best_fold_scores", return_value=None),
        patch("cycle.run._save_code", return_value="s3://code"),
        patch("cycle.run.insert_attempt") as mock_insert,
        patch("cycle.run.update_bandit") as mock_bandit,
        patch("cycle.run.materialize_best_pipeline", return_value="materialized"),
        patch("cycle.run.insert_pipeline"),
        patch("cycle.run._best_pipeline_upload"),
    ):
        data = run_attempt_core(
            conn, _config(), lessons=[], prev_best_cv=0.89,
            defer_promotion=False,
        )
    return data, mock_insert, mock_bandit


def test_confirm_rejected_jump_downgrades_bandit_reward():
    """confirm이 holdout 악화로 거부하면, raw.attempts.label은 jump로 남지만
    update_bandit엔 regression으로 전달돼야 한다 — 안 그러면 confirm이 거부해도
    같은 action_type이 다음 cycle에 계속 높은 확률로 재선택된다(#164 실측: s6e1
    preprocessing 후보 32회 재생성)."""
    confirm = ConfirmResult(
        confirmed=False, holdout_score=0.8, seed_gains={"7": {}}, holdout_regressed=True,
    )
    data, mock_insert, mock_bandit = _run_deferred(
        _iso(label="jump", gain_vs_best=0.02), confirm_result=confirm,
    )

    assert data.label == "jump"  # attempt 시점 판정은 그대로 보존
    assert mock_insert.call_args[0][1]["label"] == "jump"  # DB row도 마찬가지
    assert data.reward_label == "regression"  # 하류 학습 신호만 다운그레이드
    assert mock_bandit.call_args.kwargs["label"] == "regression"


def test_confirm_confirmed_jump_keeps_bandit_reward():
    """confirm이 통과하면 기존과 동일하게 label="jump"로 보상(회귀 방지)."""
    confirm = ConfirmResult(confirmed=True, holdout_score=0.9, seed_gains={"7": {}})
    data, mock_insert, mock_bandit = _run_deferred(
        _iso(label="jump", gain_vs_best=0.02), confirm_result=confirm,
    )

    assert data.reward_label == "jump"
    assert mock_bandit.call_args.kwargs["label"] == "jump"
