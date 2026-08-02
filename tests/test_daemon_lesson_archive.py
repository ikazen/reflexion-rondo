"""bin/run_daemon.py — 저효율 교훈 자동 archive 스윕 배선 (#76).

archive_low_gain_lessons 자체의 SQL/필터 로직은 tests/test_archive_lessons.py가
커버한다 — 여기서는 daemon의 하루 주기 게이트 + 호출 배선만 검증한다.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import bin.run_daemon as run_daemon
from bin.run_daemon import _LESSON_ARCHIVE_SWEEP_INTERVAL_SEC, _sweep_low_gain_lessons


def _long_ago() -> float:
    """time.monotonic()의 기준점이 환경마다 달라(프로세스 시작 시점 등일 수 있음)
    단순히 0.0을 쓰면 "24시간 지났다"는 보장이 안 된다 — 현재 시각 기준 상대값으로
    확실히 주기를 넘긴 시점을 만든다."""
    return time.monotonic() - _LESSON_ARCHIVE_SWEEP_INTERVAL_SEC - 1


def test_sweep_respects_daily_rate_gate(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_lesson_archive_sweep", time.monotonic())
    with patch("bin.run_daemon.archive_low_gain_lessons") as mock_archive:
        _sweep_low_gain_lessons(MagicMock())
    mock_archive.assert_not_called()


def test_sweep_calls_archive_when_due(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_lesson_archive_sweep", _long_ago())
    conn = MagicMock()
    with patch("bin.run_daemon.archive_low_gain_lessons", return_value=["r1", "r2"]) as mock_archive:
        _sweep_low_gain_lessons(conn)
    mock_archive.assert_called_once_with(conn)


def test_sweep_does_not_crash_when_archive_raises(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_lesson_archive_sweep", _long_ago())
    with patch("bin.run_daemon.archive_low_gain_lessons", side_effect=RuntimeError("boom")):
        _sweep_low_gain_lessons(MagicMock())  # 예외가 여기까지 전파되면 실패
