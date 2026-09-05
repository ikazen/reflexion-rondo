"""super-cycle(defer_promotion) 경로에서 was_promoted가 영구 NULL로 남지 않는지 검증한다 (#205).

`store/schema.sql`의 `reflection_impact` 뷰가 `was_promoted IS NOT FALSE`로 NULL을
"legacy(승격됨)"로 취급하는 기존 관례 때문에, promote가 아직 안 뒤집은 attempt를 NULL로
두면(특히 attempt_gate 도입 후 늦게 도착하는 attempt) 승자로 잘못 집계된다. defer_promotion
경로는 False로 시작해 promote가 winner만 나중에 True로 뒤집어야 한다.
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


def _iso() -> IsolatedResult:
    return IsolatedResult(
        cv_score=0.9, cv_fold_var=0.0005, fold_scores=[0.89, 0.90, 0.91],
        label="neutral", gain_vs_best=0.01, error_trace=None,
    )


def _run(**kwargs):
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
        patch("cycle.run.eval_isolated", return_value=_iso()),
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
        run_attempt_core(conn, _config(), lessons=[], prev_best_cv=0.89, **kwargs)
    return mock_insert.call_args[0][1]


def test_super_cycle_defer_promotion_defaults_was_promoted_false():
    row = _run(super_cycle_id="sc-1", defer_promotion=True)
    assert row["was_promoted"] is False


def test_explicit_was_promoted_not_overridden():
    row = _run(super_cycle_id="sc-1", defer_promotion=True, was_promoted=True)
    assert row["was_promoted"] is True


def test_non_super_cycle_path_omits_was_promoted_key():
    """super_cycle_id가 없는(직접모드) 경로는 기존 동작 그대로 — key 자체를 안 넣는다."""
    row = _run(super_cycle_id=None, defer_promotion=False)
    assert "was_promoted" not in row
