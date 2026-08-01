"""bin/run_daemon.py — bootstrap 배치 종료 시 establish_bootstrap_baseline 호출 배선(#100).

establish_bootstrap_baseline 자체의 동작(confirm/승격 로직)은
tests/test_establish_bootstrap_baseline.py가 커버한다 — 여기서는 daemon이 그 함수를
"stage=bootstrap이고 cycles_done>0일 때만" 올바른 인자로 호출하는지만 검증한다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl

from bin.api import DaemonState
from bin.run_daemon import OllamaPacer, _process


def _queue_item(stage: str, n_cycles: int = 1) -> dict:
    return {"queue_id": "q1", "competition": "s4e1", "stage": stage, "n_cycles": n_cycles}


def _disabled_pacer() -> OllamaPacer:
    return OllamaPacer(session_hours=5.0, session_cycles=0, weekly_cycles=0)


def _run_airflow_mode(stage: str, n_cycles: int = 1):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("attempt1", 0.9, "neutral")

    with (
        patch("bin.run_daemon._is_cancelled", return_value=False),
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_dag_run", return_value="run1"),
        patch("bin.run_daemon.airflow_client.wait_for_dag_run", return_value="success"),
        patch("bin.run_daemon._set_status"),
        patch("bin.run_daemon.load_train", return_value=pl.DataFrame({"x": [1.0], "y": [0.0]})),
        patch("bin.run_daemon.establish_bootstrap_baseline") as mock_establish,
    ):
        _process(conn, _queue_item(stage, n_cycles), _disabled_pacer(), DaemonState())
    return mock_establish


def test_bootstrap_stage_success_triggers_baseline_establishment():
    mock_establish = _run_airflow_mode("bootstrap", n_cycles=1)
    mock_establish.assert_called_once()
    assert mock_establish.call_args.kwargs["competition_id"] == "playground-series-s4e1"


def test_reflexion_stage_does_not_trigger_baseline_establishment():
    """bootstrap이 아닌 stage는 이미 확정 baseline이 있는 게 전제라 매번 부를 필요 없다."""
    mock_establish = _run_airflow_mode("reflexion", n_cycles=1)
    mock_establish.assert_not_called()


def test_bootstrap_establishment_failure_does_not_crash_daemon():
    """establish_bootstrap_baseline이 예외를 던져도 daemon _process 자체는 정상 반환해야 한다."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("attempt1", 0.9, "neutral")

    with (
        patch("bin.run_daemon._is_cancelled", return_value=False),
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_dag_run", return_value="run1"),
        patch("bin.run_daemon.airflow_client.wait_for_dag_run", return_value="success"),
        patch("bin.run_daemon._set_status"),
        patch("bin.run_daemon.load_train", return_value=pl.DataFrame({"x": [1.0], "y": [0.0]})),
        patch("bin.run_daemon.establish_bootstrap_baseline", side_effect=RuntimeError("boom")),
    ):
        _process(conn, _queue_item("bootstrap", n_cycles=1), _disabled_pacer(), DaemonState())
    # 예외가 여기까지 전파되지 않았으면(테스트 함수가 끝까지 실행됐으면) 통과.
