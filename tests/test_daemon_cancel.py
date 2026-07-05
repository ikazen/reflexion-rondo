"""BON-274: 사이클 성공 직후 취소 감지가 무조건적 status='running' 덮어쓰기보다
먼저 일어나는지 검증한다.

실제 운영에서 관찰된 버그: `_process`의 성공 분기가 조건 없이 status를 "running"으로
재기록해서, 그 사이클이 진행되는 동안(대부분의 시간) 걸린 외부 PATCH cancelled 요청이
조용히 지워지고 다음 반복의 취소 체크는 이미 복구된 "running"만 보게 됐다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from bin.api import DaemonState
from bin.run_daemon import OllamaPacer, _process


def _queue_item(n_cycles: int = 5) -> dict:
    return {"queue_id": "q1", "competition": "s4e1", "stage": "reflexion", "n_cycles": n_cycles}


def _disabled_pacer() -> OllamaPacer:
    return OllamaPacer(session_hours=5.0, session_cycles=0, weekly_cycles=0)


def test_cancellation_after_successful_cycle_stops_loop_and_persists():
    """사이클 1이 성공한 직후 취소가 감지되면, 이후 사이클을 트리거하지 않고
    status='cancelled'로 기록한 뒤 즉시 반환해야 한다(status='running'으로
    되돌아가면 안 된다)."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("attempt1", 0.9, "neutral")

    with (
        patch("bin.run_daemon._is_cancelled", side_effect=[False, True]) as mock_cancelled,
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_dag_run", return_value="run1"),
        patch("bin.run_daemon.airflow_client.wait_for_dag_run", return_value="success"),
        patch("bin.run_daemon._set_status") as mock_set_status,
    ):
        _process(conn, _queue_item(), _disabled_pacer(), DaemonState())

    # top-of-loop(cycle 1)=False, post-cycle-success(cycle 1)=True — 두 번째 사이클은
    # 시작조차 하지 않아야 하므로 _is_cancelled는 정확히 2번만 호출된다.
    assert mock_cancelled.call_count == 2

    # 초기 "running" 설정(1회) + 취소 감지 시 "cancelled" 설정(1회) = 2번.
    # 성공 분기의 무조건적 "running" 재기록은 일어나지 않아야 한다.
    statuses = [call.args[2] for call in mock_set_status.call_args_list]
    assert statuses == ["running", "cancelled"]

    last_call = mock_set_status.call_args_list[-1]
    assert last_call.kwargs["cycles_done"] == 1
    assert last_call.kwargs["latest_score"] == 0.9


def test_no_cancellation_runs_all_cycles_normally():
    """취소가 전혀 없으면 기존처럼 n_cycles만큼 전부 실행되고 마지막에 'done'으로
    끝나야 한다 — 회귀 확인."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("attempt1", 0.9, "neutral")

    with (
        patch("bin.run_daemon._is_cancelled", return_value=False),
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_dag_run", return_value="run1"),
        patch("bin.run_daemon.airflow_client.wait_for_dag_run", return_value="success"),
        patch("bin.run_daemon._set_status") as mock_set_status,
    ):
        _process(conn, _queue_item(n_cycles=2), _disabled_pacer(), DaemonState())

    statuses = [call.args[2] for call in mock_set_status.call_args_list]
    # initial running, running(post-cycle1), running(post-cycle2), done(final)
    assert statuses == ["running", "running", "running", "done"]
