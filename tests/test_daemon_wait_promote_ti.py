"""bin/run_daemon.py — daemon 대기 지점을 DAG-run 전체에서 promote task로 전환 (#204).

attempt_gate(#203)만 배포해선 daemon이 여전히 DAG run 전체(straggler 포함)를 기다려
시간이 안 줄어든다 — `_WAIT_ON_PROMOTE_TI` 플래그로 대기 지점 자체를 바꾼다. 기본값
off이므로 플래그를 건드리지 않는 기존 테스트(test_daemon_lease.py 등)는 영향 없다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from bin.api import DaemonState
from bin.run_daemon import OllamaPacer, _process


def _queue_item() -> dict:
    return {
        "queue_id": "q1", "competition": "s4e1", "stage": "reflexion",
        "n_cycles": 1, "cycles_done": 0, "latest_score": None,
    }


def _disabled_pacer() -> OllamaPacer:
    return OllamaPacer(session_hours=5.0, session_cycles=0, weekly_cycles=0)


def test_flag_off_waits_on_dag_run_not_task_instance():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("attempt1", 0.9, "neutral")
    with (
        patch("bin.run_daemon._WAIT_ON_PROMOTE_TI", False),
        patch("bin.run_daemon._is_cancelled", return_value=False),
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_dag_run", return_value="run1"),
        patch("bin.run_daemon.airflow_client.wait_for_dag_run", return_value="success") as mock_wait_run,
        patch("bin.run_daemon.airflow_client.wait_for_task_instance") as mock_wait_ti,
        patch("bin.run_daemon._set_status"),
        patch("bin.run_daemon.DAEMON_CYCLES_PER_LEASE", 1),
    ):
        _process(conn, _queue_item(), _disabled_pacer(), DaemonState())
    mock_wait_run.assert_called_once_with("run1")
    mock_wait_ti.assert_not_called()


def test_flag_on_waits_on_promote_task_instance():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("attempt1", 0.9, "neutral")
    with (
        patch("bin.run_daemon._WAIT_ON_PROMOTE_TI", True),
        patch("bin.run_daemon._is_cancelled", return_value=False),
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_dag_run", return_value="run1"),
        patch("bin.run_daemon.airflow_client.wait_for_dag_run") as mock_wait_run,
        patch("bin.run_daemon.airflow_client.wait_for_task_instance", return_value="success") as mock_wait_ti,
        patch("bin.run_daemon._set_status"),
        patch("bin.run_daemon.DAEMON_CYCLES_PER_LEASE", 1),
    ):
        _process(conn, _queue_item(), _disabled_pacer(), DaemonState())
    mock_wait_ti.assert_called_once_with("run1", "promote")
    mock_wait_run.assert_not_called()


def test_flag_on_treats_failed_promote_as_cycle_failure():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    with (
        patch("bin.run_daemon._WAIT_ON_PROMOTE_TI", True),
        patch("bin.run_daemon._is_cancelled", return_value=False),
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_dag_run", return_value="run1"),
        patch("bin.run_daemon.airflow_client.wait_for_task_instance", return_value="failed"),
        patch("bin.run_daemon._set_status") as mock_status,
        patch("bin.run_daemon.DAEMON_CYCLES_PER_LEASE", 1),
    ):
        _process(conn, _queue_item(), _disabled_pacer(), DaemonState())
    # 최종 상태가 성공(done)이 아니라 실패로 기록돼야 한다 — _set_status(conn, queue_id,
    # status, **extra) 마지막 호출의 status 위치 인자(args[2]) 확인.
    final_status = mock_status.call_args_list[-1].args[2]
    assert final_status == "failed"
