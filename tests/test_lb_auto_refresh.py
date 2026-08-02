"""LB 자동 재폴링 단위 테스트.

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


def test_refresh_due_handles_naive_db_datetimes_against_aware_now():
    """raw.kaggle_submissions.submitted_at/checked_at는 timezone 없는 `timestamp`
    컬럼이라 psycopg2가 naive datetime을 반환하는데, _sweep_stale_submissions는
    aware(datetime.now(timezone.utc))인 now와 비교한다 — DB round-trip을 흉내내
    submitted_at/checked_at을 naive로, now는 aware로 명시적으로 섞는다.
    """
    now_aware = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)
    submitted_at_naive = datetime(2026, 8, 2, 11, 55, 0)  # DB에서 온 그대로 — tzinfo 없음
    checked_at_naive = datetime(2026, 8, 2, 11, 59, 30)
    # TypeError 없이 실행되면 통과 — 첫 10분 윈도우(2분 간격) 안이라 아직 재확인 아님.
    assert _submission_refresh_due(submitted_at_naive, checked_at_naive, now_aware) is False


def test_refresh_due_naive_and_aware_inputs_agree_on_result():
    """naive/aware 어느 조합으로 넣어도 같은 절대시각이면 같은 판정이 나와야 한다."""
    aware_submitted = datetime(2026, 8, 2, 11, 55, 0, tzinfo=timezone.utc)
    naive_submitted = datetime(2026, 8, 2, 11, 55, 0)
    aware_checked = datetime(2026, 8, 2, 11, 57, 0, tzinfo=timezone.utc)
    naive_checked = datetime(2026, 8, 2, 11, 57, 0)
    now = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)

    assert (
        _submission_refresh_due(aware_submitted, aware_checked, now)
        == _submission_refresh_due(naive_submitted, naive_checked, now)
    )


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
    conn = MagicMock()
    # 1) submission row  2) 발산검사 cv_score 조회  3) 직전 완료 제출 조회(없음 → 발산 스킵)
    conn.execute.return_value.fetchone.side_effect = [row, (0.9,), None]
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


# --- cv↔LB 발산 트립와이어 ---
# cv는 개선인데 LB가 악화된 제출을 감지해 원천 pipeline을 격리하고
# 해당 대회 auto-submit을 중단한다.

from bin.api import _apply_cv_lb_divergence_tripwire, _detect_cv_lb_divergence  # noqa: E402


def test_detect_divergence_flags_cv_improved_lb_regressed():
    """rmse(metric_sign=-1)에서 cv는 낮아졌는데(개선) LB는 높아졌으면(악화) 발산."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [
        (0.5,),          # 이번 attempt의 cv_score (직전 1.0보다 개선)
        (1.0, 1.2, -1),  # 직전 완료 제출: prev_cv=1.0, prev_lb=1.2, metric_sign=-1
    ]
    reason = _detect_cv_lb_divergence(conn, "s5e10", "attempt-1", 2.0, _now())
    assert reason is not None
    assert reason.startswith("cv_lb_divergence")


def test_detect_divergence_returns_none_when_both_improve():
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [
        (0.5,),
        (1.0, 1.2, -1),
    ]
    reason = _detect_cv_lb_divergence(conn, "s5e10", "attempt-1", 1.0, _now())  # LB도 개선(1.2→1.0)
    assert reason is None


def test_detect_divergence_returns_none_when_no_previous_submission():
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [(0.5,), None]
    assert _detect_cv_lb_divergence(conn, "s5e10", "attempt-1", 2.0, _now()) is None


def test_detect_divergence_returns_none_when_current_attempt_has_no_cv():
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [None]
    assert _detect_cv_lb_divergence(conn, "s5e10", "attempt-1", 2.0, _now()) is None


def test_apply_tripwire_quarantines_pipeline_and_pauses_competition():
    conn = MagicMock()
    _apply_cv_lb_divergence_tripwire(conn, "s5e10", "attempt-1", "cv_lb_divergence: ...")

    pipeline_call = next(c for c in conn.execute.call_args_list if "raw.pipelines" in c.args[0])
    assert pipeline_call.args[1] == ["cv_lb_divergence: ...", "attempt-1"]

    comp_call = next(c for c in conn.execute.call_args_list if "raw.competitions" in c.args[0])
    assert comp_call.args[1] == ["cv_lb_divergence: ...", "s5e10"]


def test_refresh_submission_row_complete_with_divergence_applies_tripwire():
    row = ("sub-1", "s5e10", "attempt-1", _now(), "msg", None, "submitted", None, None, None)
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [
        row,             # submission row
        (0.5,),          # 발산검사: 현재 cv_score (개선)
        (1.0, 1.2, -1),  # 발산검사: 직전 제출 cv/lb/metric_sign
    ]
    with patch("bin.api._poll_kaggle_once", return_value=("complete", 2.0)):
        rec = refresh_submission_row(conn, "sub-1")

    assert rec["status"] == "complete"
    assert "divergence" in rec
    pipeline_calls = [c for c in conn.execute.call_args_list if "raw.pipelines" in c.args[0]]
    assert len(pipeline_calls) == 1
    comp_calls = [c for c in conn.execute.call_args_list if "auto_submit_paused_reason" in c.args[0]]
    assert len(comp_calls) == 1


def test_refresh_submission_row_complete_without_divergence_does_not_pause():
    row = ("sub-1", "s5e10", "attempt-1", _now(), "msg", None, "submitted", None, None, None)
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [row, (0.9,), None]  # 직전 제출 없음
    with patch("bin.api._poll_kaggle_once", return_value=("complete", 0.85)):
        rec = refresh_submission_row(conn, "sub-1")

    assert "divergence" not in rec
    comp_calls = [c for c in conn.execute.call_args_list if "auto_submit_paused_reason" in c.args[0]]
    assert len(comp_calls) == 0


# --- auto_submit 일시중단 게이트 ---

def test_auto_submit_skips_paused_competition(monkeypatch):
    import bin.api as api_mod
    from bin.api import DaemonState, create_app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_mod, "_competition_id_to_slug", lambda: {"s5e10": "s5e10"})
    monkeypatch.setattr(api_mod, "_best_attempt", lambda conn, cid: ("cand", 0.9))
    monkeypatch.setattr(api_mod, "_start_submission", lambda *a, **k: "sub-1")

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [("s5e10",)]
    conn.execute.return_value.fetchone.return_value = ("cv_lb_divergence: ...",)

    app = create_app(conn, DaemonState())
    resp = TestClient(app).post("/api/submissions/auto", json={"window_hours": 24})
    body = resp.json()

    assert body["submitted"] == []
    assert "auto-submit paused" in body["skipped"][0]["reason"]


# --- GET /api/cv-lb-calibration ---

def test_cv_lb_calibration_endpoint_returns_rows():
    from bin.api import DaemonState, create_app
    from fastapi.testclient import TestClient

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("sub-2", "s5e10", _now(), "attempt-2", 0.02, 0.30, 0.02, -0.24, True),
    ]
    app = create_app(conn, DaemonState())
    resp = TestClient(app).get("/api/cv-lb-calibration", params={"competition": "s5e10"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["diverged"] is True
    sql = conn.execute.call_args_list[-1].args[0]
    assert "cv_lb_calibration" in sql
    assert "competition_id = %s" in sql
