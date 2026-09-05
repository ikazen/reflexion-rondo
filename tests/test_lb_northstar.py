"""GET /api/lb-northstar: 북극성 지표를 lb_percentile에서 gap_to_p90으로 교체 (#289).

lb_percentile(rank-count 백분위)은 저효과 진입(sample_submission 그대로 제출 등)이
두터운 대회에서 "몇 %를 이겼는가"를 과대평가한다 — s5e4 실측: 백분위 77.99%인데
실제 경쟁 밴드(p10~p50)는 11.5~11.87, 우리 점수 12.6은 그 밴드에 못 미친다. p90(상위
10% 컷오프)까지의 원점수 거리(gap_to_p90, store/schema.sql의 lb_gap_to_p90 뷰)를
대표 지표로 추가한다 — lb_percentile은 진단용으로 응답에 남긴다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from bin.api import DaemonState, create_app


def _lb_northstar_row(
    *,
    last_row=None,
    best_pct=None,
    gap_row=None,
    submissions_today=0,
    backlog=0,
):
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [
        last_row,
        (best_pct,),
        gap_row,
        (submissions_today,),
        (backlog,),
    ]
    return conn


def test_lb_northstar_includes_gap_to_p90(monkeypatch):
    import bin.api as api_mod

    monkeypatch.setattr(api_mod, "_active_competition_ids", lambda: {"playground-series-s4e11"})
    submitted_at = datetime(2026, 7, 26, 21, 0, tzinfo=timezone.utc)
    conn = _lb_northstar_row(
        last_row=(0.94227, 68.59, submitted_at),
        best_pct=68.59,
        gap_row=(0.94371, 0.94227, 0.00144),
    )

    body = TestClient(create_app(conn, DaemonState())).get("/api/lb-northstar").json()

    assert len(body) == 1
    row = body[0]
    assert row["competition_id"] == "playground-series-s4e11"
    assert row["p90_score"] == 0.94371
    assert row["best_lb_score"] == 0.94227
    assert row["gap_to_p90"] == 0.00144
    # 진단용으로 남아있어야 한다 — 대표 지표에서 내려갔을 뿐 삭제 아님.
    assert row["last_lb_percentile"] == 68.59
    assert row["best_lb_percentile"] == 68.59


def test_lb_northstar_gap_fields_none_without_snapshot(monkeypatch):
    """lb_gap_to_p90 뷰에 해당 대회 행이 없으면(스냅샷 없음) gap 필드 전부 None —
    나머지 필드(제출 예산 등)는 정상 응답."""
    import bin.api as api_mod

    monkeypatch.setattr(api_mod, "_active_competition_ids", lambda: {"playground-series-new"})
    conn = _lb_northstar_row(last_row=None, best_pct=None, gap_row=None, submissions_today=1, backlog=2)

    body = TestClient(create_app(conn, DaemonState())).get("/api/lb-northstar").json()

    row = body[0]
    assert row["p90_score"] is None
    assert row["best_lb_score"] is None
    assert row["gap_to_p90"] is None
    assert row["submissions_today"] == 1
    assert row["unsubmitted_confirmed"] == 2
