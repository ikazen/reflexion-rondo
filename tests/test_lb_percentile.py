"""LB 백분위 산출·저장 단위 테스트 (#233).

lb_score 원값은 대회마다 metric도 스케일도 달라 fleet 횡단 비교가 불가능해서,
대회 리더보드 분포 안에서의 백분위를 북극성 지표로 쓴다.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from bin.api import (
    DaemonState,
    _backfill_lb_percentiles,
    _lb_percentile,
    _percentile_in,
    create_app,
    refresh_submission_row,
)


def test_percentile_higher_is_better_metric():
    """auc처럼 metric_sign=1인 대회 — 우리보다 낮은 점수 팀 비율."""
    scores = [0.97, 0.96, 0.95, 0.94, 0.90]
    assert _percentile_in(scores, 0.965, 1) == pytest.approx(80.0)
    assert _percentile_in(scores, 0.99, 1) == pytest.approx(100.0)
    assert _percentile_in(scores, 0.80, 1) == pytest.approx(0.0)


def test_percentile_lower_is_better_metric():
    """rmse처럼 metric_sign=-1인 대회 — 우리보다 높은(나쁜) 점수 팀 비율."""
    scores = [10.0, 11.0, 12.0, 13.0]
    assert _percentile_in(scores, 10.5, -1) == pytest.approx(75.0)
    assert _percentile_in(scores, 9.0, -1) == pytest.approx(100.0)
    assert _percentile_in(scores, 14.0, -1) == pytest.approx(0.0)


def test_percentile_ties_do_not_count_as_beaten():
    """동점 팀은 이긴 게 아니다 — 엄격 부등호."""
    assert _percentile_in([0.9, 0.9, 0.8], 0.9, 1) == pytest.approx(100.0 / 3)


def test_lb_percentile_returns_none_without_snapshot():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    assert _lb_percentile(conn, "s4e10", 0.96) is None


def test_lb_percentile_accepts_json_string_scores():
    """psycopg2가 jsonb를 파싱해 list로 주기도 하고 str로 주기도 한다 — 둘 다 받는다."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (json.dumps([0.97, 0.95, 0.90]), 1)
    assert _lb_percentile(conn, "s4e10", 0.96) == pytest.approx(200.0 / 3)


def test_backfill_fills_existing_completed_submissions():
    """스냅샷을 새로 받으면 그 대회의 과거 완료 제출 백분위를 한꺼번에 채운다."""
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [("sub-1", 0.96), ("sub-2", 0.94)]
    conn.execute.return_value.fetchone.return_value = ([0.97, 0.95, 0.90], 1)

    filled = _backfill_lb_percentiles(conn, "s4e10")

    assert filled == 2
    updates = [
        c for c in conn.execute.call_args_list
        if "set lb_percentile" in c.args[0]
    ]
    assert len(updates) == 2


def test_refresh_writes_lb_percentile_on_complete():
    """제출이 complete로 확정되면 lb_score와 함께 백분위도 기록한다."""
    row = ("sub-1", "s4e10", "attempt-1", None, "msg", None, "submitted", None, None, None)
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [
        row,
        ([0.97, 0.95, 0.90], 1),   # _lb_percentile 스냅샷 조회
    ]
    with patch("bin.api._poll_kaggle_once", return_value=("complete", 0.96)), \
         patch("bin.api._detect_cv_lb_divergence", return_value=None):
        rec = refresh_submission_row(conn, "sub-1")

    assert rec["lb_score"] == 0.96
    assert rec["lb_percentile"] == pytest.approx(200.0 / 3)


def test_refresh_survives_missing_snapshot():
    """스냅샷이 없는 대회여도 lb_score 기록은 정상 진행된다 — 백분위만 비워둔다."""
    row = ("sub-1", "s4e10", "attempt-1", None, "msg", None, "submitted", None, None, None)
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [row, None]
    with patch("bin.api._poll_kaggle_once", return_value=("complete", 0.96)), \
         patch("bin.api._detect_cv_lb_divergence", return_value=None):
        rec = refresh_submission_row(conn, "sub-1")

    assert rec["lb_score"] == 0.96
    assert "lb_percentile" not in rec


def test_leaderboard_refresh_skips_fresh_snapshot(monkeypatch):
    """종료된 대회는 최종 LB가 고정이라 매번 다시 받을 이유가 없다."""
    import bin.api as api_mod
    from datetime import datetime, timezone

    monkeypatch.setattr(api_mod, "_active_competition_ids", lambda: {"s4e10"})
    fetch = MagicMock()
    monkeypatch.setattr(api_mod, "_fetch_leaderboard_scores", fetch)

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (datetime.now(timezone.utc),)
    from fastapi.testclient import TestClient

    body = TestClient(create_app(conn, DaemonState())).post(
        "/api/leaderboard/refresh", json={}
    ).json()

    fetch.assert_not_called()
    assert body["skipped"] == [{"competition": "s4e10", "reason": "snapshot fresh"}]
