"""run_attempt_core의 attempt 단위 CPU 예산 집행을 검증한다.

과거엔 eval 회차마다 독립적으로 900초 CPU 예산을 줘서, 1회차가 rc=-9(CPU
워치독 kill)로 죽으면 그 무의미한 원문 에러를 그대로 재생성 피드백으로 넘겨
2회차도 같은 자리에서 또 900초를 태웠다(2026-08 실측: CPU kill attempt
113건 전부가 이 경로, attempt당 최대 ~1800초). 지금은 예산을 attempt 전체
기준으로 집행해 1회차가 다 쓰면 2회차를 아예 돌리지 않고, 재생성 피드백도
원문 대신 실행 가능한 지시로 바꾼다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl

from agents.strategist import StrategyDecision
from cycle.run import CycleConfig, run_attempt_core
from cycle.stagnation import StagnationSignal
from runtime.isolate import DEFAULT_CPU_BUDGET_SECS, IsolatedResult


def _config(cpu_budget_secs: float | None = None) -> CycleConfig:
    return CycleConfig(
        competition_id="s4e1",
        train=pl.DataFrame({"f": [1.0, 2.0, 3.0], "target": [0, 1, 0]}),
        target_col="target",
        metric="auc",
        stage="reflexion",
        eda_card="n_rows=3",
        cpu_budget_secs=cpu_budget_secs,
    )


def _cpu_kill(peak_cpu_sec: float, budget: float = DEFAULT_CPU_BUDGET_SECS) -> IsolatedResult:
    return IsolatedResult(
        cv_score=None, cv_fold_var=None, fold_scores=None, label=None,
        gain_vs_best=None,
        error_trace=f"cpu budget exceeded: {peak_cpu_sec:.0f}s CPU used (limit {budget:.0f}s)",
        peak_cpu_sec=peak_cpu_sec,
    )


def _run(eval_side_effect, generate_code_mock=None, cpu_budget_secs=None):
    conn = MagicMock()
    generate_code_mock = generate_code_mock or MagicMock(return_value="source")
    with (
        patch("cycle.run.detect_stagnation",
              return_value=StagnationSignal(False, 0, (), 0)),
        patch("cycle.run.get_action_prior", return_value={}),
        patch("cycle.run.strategize", return_value=StrategyDecision(
            hypothesis="h", action_type="hyperparam_search", reflection_ids=[])),
        patch("cycle.run.top_error_pitfalls", return_value=[]),
        patch("cycle.run.generate_code", generate_code_mock),
        patch("cycle.run.validate_patch", return_value=[]),
        patch("cycle.run.eval_isolated", side_effect=eval_side_effect) as mock_eval,
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
            conn, _config(cpu_budget_secs), lessons=[], prev_best_cv=0.89,
            defer_promotion=True,
        )
    return data, mock_insert, mock_eval, generate_code_mock


def test_retry_skipped_when_first_eval_exhausts_cpu_budget():
    """1회차가 예산 전부를 태우고 죽으면 2회차 eval_isolated를 호출하지
    않는다 — attempt당 최대 소모를 절반으로 자르는 핵심 동작."""
    data, mock_insert, mock_eval, generate_code_mock = _run(
        eval_side_effect=[_cpu_kill(peak_cpu_sec=DEFAULT_CPU_BUDGET_SECS)],
    )

    assert mock_eval.call_count == 1
    assert data.label == "error"
    row = mock_insert.call_args[0][1]
    assert "cpu budget exceeded" in row["error_trace"]
    assert row["peak_cpu_sec"] == DEFAULT_CPU_BUDGET_SECS
    # 재시도가 스킵됐으니 재생성도 일어나지 않는다(codegen 단계의 최초 1회만 호출됨).
    assert generate_code_mock.call_count == 1


def test_retry_uses_remaining_budget_when_first_eval_partially_spends():
    """1회차가 예산을 다 쓰지 않고 실패하면(static 검증 실패 등과 무관한 일반
    에러) 남은 예산으로 2회차를 시도한다."""
    ok = IsolatedResult(
        cv_score=0.9, cv_fold_var=0.001, fold_scores=[0.89, 0.9, 0.91],
        label="neutral", gain_vs_best=0.01, error_trace=None,
        peak_cpu_sec=120.0,
    )
    data, mock_insert, mock_eval, generate_code_mock = _run(
        eval_side_effect=[
            IsolatedResult(
                cv_score=None, cv_fold_var=None, fold_scores=None, label=None,
                gain_vs_best=None, error_trace="ValueError: bad column",
                peak_cpu_sec=100.0,
            ),
            ok,
        ],
    )

    assert mock_eval.call_count == 2
    kwargs = mock_eval.call_args_list[1].kwargs
    assert kwargs["cpu_budget_sec"] == DEFAULT_CPU_BUDGET_SECS - 100.0
    assert data.label == "neutral"


def test_config_cpu_budget_overrides_env_default():
    """comp.CPU_BUDGET_SECS(config.cpu_budget_secs)가 설정되면 env/DEFAULT_CPU_BUDGET_SECS
    보다 우선한다(#176) — s6e8처럼 900s 벽에서 성공 attempt가 계속 죽는 대회를
    대회별로 넉넉히 풀어주기 위함."""
    data, mock_insert, mock_eval, generate_code_mock = _run(
        eval_side_effect=[_cpu_kill(peak_cpu_sec=3600.0, budget=3600)],
        cpu_budget_secs=3600.0,
    )

    assert mock_eval.call_count == 1
    assert mock_eval.call_args_list[0].kwargs["cpu_budget_sec"] == 3600.0
    row = mock_insert.call_args[0][1]
    assert row["peak_cpu_sec"] == 3600.0


def test_regenerate_feedback_is_actionable_not_raw_rc_message():
    """CPU kill 시 재생성 피드백은 원문(rc=-9 등) 대신 실행 가능한 지시로
    바뀐다 — 원문은 LLM이 이유를 알 수 없어 비슷하게 비싼 코드를 다시 써서
    2회차도 같은 자리에서 죽는 낭비를 낳았다."""
    ok = IsolatedResult(
        cv_score=0.9, cv_fold_var=0.001, fold_scores=[0.89, 0.9, 0.91],
        label="neutral", gain_vs_best=0.01, error_trace=None,
        peak_cpu_sec=50.0,
    )
    generate_code_mock = MagicMock(return_value="source")
    _run(
        eval_side_effect=[_cpu_kill(peak_cpu_sec=300.0, budget=900), ok],
        generate_code_mock=generate_code_mock,
    )

    # generate_code(source 최초 생성) + generate_code(error_feedback=...) 재생성 = 2회
    assert generate_code_mock.call_count == 2
    feedback = generate_code_mock.call_args_list[1].kwargs["error_feedback"]
    assert "cpu budget exceeded" not in feedback
    assert "rc=" not in feedback
    assert "CPU 예산" in feedback
