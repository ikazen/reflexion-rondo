"""bin/archive_lessons.py 단위 테스트 (#76 자동 배선).

CLI는 기존에 이미 이 로직을 구현하고 있었다 — 이번 변경은 순수 리팩터(로직을
find_archive_candidates/archive_low_gain_lessons로 추출해 daemon과 공유)와
그 함수의 배선 검증이 목적. SQL 필터 조건 자체(times_applied/avg_gain 임계)는
기존 CLI가 이미 검증된 형태로 쓰던 것을 그대로 옮긴 것.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from bin.archive_lessons import archive_low_gain_lessons, find_archive_candidates


def _conn_with(rows) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows
    return conn


def test_find_archive_candidates_passes_thresholds():
    conn = _conn_with([])
    find_archive_candidates(conn, min_applied=5, max_gain=-0.01)
    params = conn.execute.call_args.args[1]
    assert params == [-0.01, 5]


def test_find_archive_candidates_query_filters_archived_and_applies_thresholds():
    conn = _conn_with([])
    find_archive_candidates(conn)
    sql = conn.execute.call_args.args[0]
    assert "avg_gain <= %s" in sql
    assert "times_applied >= %s" in sql
    assert "archived = false" in sql


def test_archive_low_gain_lessons_returns_empty_when_no_candidates():
    conn = _conn_with([])
    result = archive_low_gain_lessons(conn)
    assert result == []
    update_calls = [c for c in conn.execute.call_args_list if "update raw.reflections" in c.args[0]]
    assert update_calls == []


def test_archive_low_gain_lessons_updates_archived_flag():
    rows = [
        ("r1", 5, -0.01, "L2_class", "s4e1"),
        ("r2", 4, -0.02, "L3_general", "s4e1"),
    ]
    conn = _conn_with(rows)
    result = archive_low_gain_lessons(conn)
    assert result == ["r1", "r2"]

    update_calls = [c for c in conn.execute.call_args_list if "update raw.reflections" in c.args[0]]
    assert len(update_calls) == 1
    assert update_calls[0].args[1] == [["r1", "r2"]]
    assert "archived = true" in update_calls[0].args[0]


def test_archive_low_gain_lessons_respects_custom_thresholds():
    conn = _conn_with([])
    archive_low_gain_lessons(conn, min_applied=10, max_gain=-0.005)
    params = conn.execute.call_args.args[1]
    assert params == [-0.005, 10]
