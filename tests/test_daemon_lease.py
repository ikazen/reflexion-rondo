"""bin/run_daemon.py — 리스 기반 큐 라운드로빈(#133) 검증.

2026-08 처리량 진단: `big` 큐 슬롯이 함대 전체 3개뿐이라(worker-vm-big 1 +
mac-server-big 2) 한 사이클의 attempt_0/1/2가 이미 다 써버려 daemon 병렬화는
처리량 이득이 없다. 대신 큐 항목 하나(30~100 cycle)가 daemon을 통째로 붙잡아
다른 대회가 며칠씩 실행 기회를 못 얻던 문제를, DAEMON_CYCLES_PER_LEASE마다
pending으로 되돌리는 라운드로빈으로 해결한다 — 동시성 없음, 레이스 없음.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from bin.api import DaemonState
from bin.run_daemon import OllamaPacer, _pop_pending, _process


def _queue_item(n_cycles: int, cycles_done: int = 0, latest_score=None) -> dict:
    return {
        "queue_id": "q1", "competition": "s4e1", "stage": "reflexion",
        "n_cycles": n_cycles, "cycles_done": cycles_done, "latest_score": latest_score,
    }


def _disabled_pacer() -> OllamaPacer:
    return OllamaPacer(session_hours=5.0, session_cycles=0, weekly_cycles=0)


def _run(item: dict, lease: int, mock_set_status=None):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("attempt1", 0.9, "neutral")
    with (
        patch("bin.run_daemon.DAEMON_CYCLES_PER_LEASE", lease),
        patch("bin.run_daemon._is_cancelled", return_value=False),
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_dag_run", return_value="run1"),
        patch("bin.run_daemon.airflow_client.wait_for_dag_run", return_value="success"),
        patch("bin.run_daemon._set_status") as mock_status,
    ):
        _process(conn, item, _disabled_pacer(), DaemonState())
    return mock_status


def test_lease_exhausted_requeues_as_pending_with_progress():
    """n_cycles(8) > lease(2)면 리스 소진 즉시 pending으로 되돌리고, 이번 리스에서
    소비한 만큼만 cycles_done을 진행시켜야 한다 — 예산이 남았으므로 done이면 안 된다."""
    mock_status = _run(_queue_item(n_cycles=8), lease=2)

    statuses = [call.args[2] for call in mock_status.call_args_list]
    # initial running, running(post-cycle1), running(post-cycle2), pending(리스 소진)
    assert statuses == ["running", "running", "running", "pending"]
    last_call = mock_status.call_args_list[-1]
    assert last_call.kwargs["cycles_done"] == 2
    assert "ended_at" not in last_call.kwargs


def test_second_lease_resumes_from_persisted_progress():
    """이전 리스가 남긴 cycles_done부터 재개해 n_cycles를 정확히 채우고 끝나야 한다."""
    mock_status = _run(_queue_item(n_cycles=3, cycles_done=2, latest_score=0.5), lease=2)

    statuses = [call.args[2] for call in mock_status.call_args_list]
    # 남은 예산 1 cycle만 이번 리스에서 소비 — initial running, running(post-cycle), done.
    assert statuses == ["running", "running", "done"]
    last_call = mock_status.call_args_list[-1]
    assert last_call.kwargs["cycles_done"] == 3


def test_resume_does_not_reset_started_at():
    """리스 재개(cycles_done>0으로 시작)에서는 started_at을 다시 세팅하지 않는다 —
    dashboard/api가 '최초 시작'으로 읽으므로. last_leased_at은 매 리스마다 갱신한다."""
    mock_status = _run(_queue_item(n_cycles=3, cycles_done=1), lease=5)

    first_call = mock_status.call_args_list[0]
    assert first_call.args[2] == "running"
    assert "started_at" not in first_call.kwargs
    assert "last_leased_at" in first_call.kwargs


def test_fresh_item_sets_started_at():
    mock_status = _run(_queue_item(n_cycles=1), lease=5)

    first_call = mock_status.call_args_list[0]
    assert "started_at" in first_call.kwargs
    assert "last_leased_at" in first_call.kwargs


def test_small_item_completes_within_single_lease():
    """n_cycles가 lease보다 작거나 같으면 기존과 동일하게 한 번에 끝까지 돈다
    (회귀 확인 — 대부분의 bootstrap 큐가 여기 해당)."""
    mock_status = _run(_queue_item(n_cycles=2), lease=5)

    statuses = [call.args[2] for call in mock_status.call_args_list]
    assert statuses == ["running", "running", "running", "done"]


def test_pop_pending_orders_by_last_leased_at_with_created_at_fallback():
    """라운드로빈 정렬 — last_leased_at 오름차순(NULL은 created_at로 폴백)이
    쿼리에 실제로 들어가는지 확인. 방금 리스를 마친 항목이 뒤로 밀리는 근거."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None

    _pop_pending(conn)

    sql = conn.execute.call_args.args[0]
    assert "coalesce(last_leased_at, created_at) asc" in sql
    assert "cycles_done" in sql
    assert "latest_score" in sql
