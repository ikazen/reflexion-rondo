"""bin/run_attempt_task.py — Airflow attempt task 진입점이 run_attempt_core를 올바른 시그니처로 부르는지 검증한다.

run_attempt_core를 autospec으로 patch하므로 제거된 인자를 넘기면 TypeError로 실패한다 — 시그니처가 바뀐 뒤
호출부가 낡은 인자를 넘겨 운영 attempt가 전부 죽는 것을 막는 회귀 가드다.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import polars as pl

from agents.strategist import StrategyDecision
from bin.run_attempt_task import main
from config.competitions import s5e8
from cycle.run import _AttemptData


def test_main_calls_run_attempt_core_with_the_current_signature():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("sc-12345678", s5e8.COMPETITION_ID, 0.97, [{"lesson": 1}], ["model_swap"])
    train = pl.DataFrame({"f": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
    data = _AttemptData(
        attempt_id="attempt-12345678",
        decision=StrategyDecision(hypothesis="h", action_type="model_swap", reflection_ids=[]),
        cv_score=0.97, label="neutral", gain_vs_best=0.0, retries=0,
    )
    argv = ["x", "-c", "s5e8", "-s", "reflexion", "--queue-id", "q", "--run-id", "r", "--attempt-index", "0"]
    with (
        patch.object(sys, "argv", argv),
        patch("store.db.connect", return_value=conn),
        patch("store.train_data.load_train", return_value=train),
        patch("evaluator.harness.split_audit_holdout", return_value=(train, train)),
        patch("cycle.run.run_attempt_core", autospec=True, return_value=data) as mock_core,
    ):
        main()

    kwargs = mock_core.call_args.kwargs
    assert kwargs["super_cycle_id"] == "sc-12345678"
    assert kwargs["attempt_index"] == 0
    assert kwargs["forced_action"] == "model_swap"
    config = mock_core.call_args.args[1]
    assert config.holdout is not None
    assert config.slug == "s5e8"
    conn.close.assert_called_once()
