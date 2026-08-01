"""LB 자동 재폴링 (#103) 단위 테스트.

bin/run_daemon.py의 백오프 로직 + 스윕 배선, bin/api.py의 refresh_submission_row
(엔드포인트와 daemon 스윕이 공유하는 핵심 로직) 둘 다 커버한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import bin.run_daemon as run_daemon
from bin.api import refresh_submission_row
from bin.run_daemon import _submission_refresh_due, _sweep_stale_submissions


def _now() -> datetime:
    return datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)


# --- _submission_refresh_due 백오프 ---

def test_refresh_due_when_never_checked():
    assert _submission_refresh_due(_now(), None, _now()) is True


def test_refresh_not_due_within_first_window():
    now = _now()
    assert _submission_refresh_due(
        now - timedelta(minutes=5), now - timedelta(seconds=30), now
    ) is False


def test_refresh_due_after_first_window_interval_elapsed():
    now = _now()
    assert _submission_refresh_due(
        now - timedelta(minutes=5), now - timedelta(minutes=3), now
    ) is True


def test_refresh_backoff_widens_for_older_submissions():
    """제출 후 1시간 넘으면 10분 간격 — 5분 전 확인이면 아직 아니다."""
    now = _now()
    assert _submission_refresh_due(
        now - timedelta(hours=2), now - timedelta(minutes=5), now
    ) is False


def test_refresh_backoff_very_old_uses_widest_interval():
    now = _now()
    submitted_at = now - timedelta(hours=10)
    assert _submission_refresh_due(submitted_at, now - timedelta(hours=1), now) is False
    assert _submission_refresh_due(submitted_at, now - timedelta(hours=3), now) is True


# --- _sweep_stale_submissions ---

def test_sweep_respects_module_level_rate_gate(monkeypatch):
    """직전 스윕 이후 SWEEP_INTERVAL이 안 지났으면 DB 조회 자체를 안 한다."""
    monkeypatch.setattr(run_daemon, "_last_submission_sweep", __import__("time").monotonic())
    conn = MagicMock()
    _sweep_stale_submissions(conn)
    conn.execute.assert_not_called()


def test_sweep_queries_submitted_and_pending_statuses(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_submission_sweep", 0.0)
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    _sweep_stale_submissions(conn)
    sql = conn.execute.call_args.args[0]
    assert "submitted" in sql
    assert "pending" in sql


def test_sweep_refreshes_only_due_rows(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_submission_sweep", 0.0)
    conn = MagicMock()
    now = datetime.now(timezone.utc)
    conn.execute.return_value.fetchall.return_value = [
        ("sub-1", now - timedelta(minutes=5), None),  # 한 번도 확인 안 됨 — due
        ("sub-2", now - timedelta(minutes=5), now - timedelta(seconds=10)),  # 방금 확인 — not due
    ]
    with patch("bin.run_daemon.refresh_submission_row", return_value={"status": "submitted"}) as mock_refresh:
        _sweep_stale_submissions(conn)
    mock_refresh.assert_called_once_with(conn, "sub-1")


def test_sweep_does_not_crash_when_refresh_raises(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_submission_sweep", 0.0)
    conn = MagicMock()
    now = datetime.now(timezone.utc)
    conn.execute.return_value.fetchall.return_value = [("sub-1", now - timedelta(minutes=5), None)]
    with patch("bin.run_daemon.refresh_submission_row", side_effect=RuntimeError("boom")):
        _sweep_stale_submissions(conn)  # 예외가 여기까지 전파되면 실패


# --- refresh_submission_row (bin/api.py) ---

def _row_conn(row):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = row
    return conn


def test_refresh_submission_row_not_found_returns_none():
    conn = _row_conn(None)
    assert refresh_submission_row(conn, "missing") is None


def test_refresh_submission_row_terminal_status_short_circuits():
    """이미 종결 상태(complete/error/invalid)면 kaggle을 다시 조회하지 않는다."""
    row = ("sub-1", "s4e1", "attempt-1", _now(), "msg", "path.csv", "complete", 0.9, None, _now())
    conn = _row_conn(row)
    with patch("bin.api._poll_kaggle_once") as mock_poll:
        rec = refresh_submission_row(conn, "sub-1")
    mock_poll.assert_not_called()
    assert rec["status"] == "complete"


def test_refresh_submission_row_pending_updates_checked_at_only():
    row = ("sub-1", "s4e1", "attempt-1", _now(), "msg", None, "submitted", None, None, None)
    conn = _row_conn(row)
    with patch("bin.api._poll_kaggle_once", return_value=("pending", None)):
        rec = refresh_submission_row(conn, "sub-1")
    assert rec["status"] == "submitted"  # status 컬럼 자체는 안 바뀜(pending은 재시도 위임)
    assert rec["checked_at"] is not None


def test_refresh_submission_row_complete_updates_lb_score_and_attempts():
    row = ("sub-1", "s4e1", "attempt-1", _now(), "msg", None, "submitted", None, None, None)
    conn = _row_conn(row)
    with patch("bin.api._poll_kaggle_once", return_value=("complete", 0.987)):
        rec = refresh_submission_row(conn, "sub-1")
    assert rec["status"] == "complete"
    assert rec["lb_score"] == 0.987

    attempts_update_calls = [
        c for c in conn.execute.call_args_list
        if "update raw.attempts" in c.args[0]
    ]
    assert len(attempts_update_calls) == 1
    assert attempts_update_calls[0].args[1][0] == 0.987


def test_refresh_submission_row_error_status_records_error_field():
    row = ("sub-1", "s4e1", "attempt-1", _now(), "msg", None, "submitted", None, None, None)
    conn = _row_conn(row)
    with patch("bin.api._poll_kaggle_once", return_value=("error", None)):
        rec = refresh_submission_row(conn, "sub-1")
    assert rec["status"] == "error"
    assert "kaggle: error" in rec["error"]
