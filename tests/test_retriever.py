"""memory/retriever.py — search_failure_lessons 단위 테스트.

embed()가 필요 없는 순수 SQL 채널이라 mock conn으로 직접 테스트 가능.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from memory.retriever import search_failure_lessons, _global_gain_stats


def _conn(rows: list[tuple]) -> MagicMock:
    mock = MagicMock()
    mock.execute.return_value.fetchall.return_value = rows
    return mock


def _conn_one(row: tuple | None) -> MagicMock:
    mock = MagicMock()
    mock.execute.return_value.fetchone.return_value = row
    return mock


def test_search_failure_lessons_no_embed_call():
    """embed()를 호출하지 않는다 — 순수 SQL 채널."""
    conn = _conn([])
    search_failure_lessons(conn, "s4e1", k=3)
    conn.execute.assert_called_once()


def test_search_failure_lessons_returns_expected_fields():
    rows = [
        ("rid-1", "Geography 인코딩하라", "전체 교훈", "L1_local", -0.02, "failure", 0.01),
    ]
    conn = _conn(rows)
    out = search_failure_lessons(conn, "s4e1", k=3)
    assert len(out) == 1
    assert out[0]["reflection_id"] == "rid-1"
    assert out[0]["lesson_type"] == "failure"
    assert "embedding" not in out[0]
    assert "sim" not in out[0]


def test_search_failure_lessons_query_filters_lesson_type_failure():
    conn = _conn([])
    search_failure_lessons(conn, "s4e1", k=3)
    sql: str = conn.execute.call_args[0][0]
    assert "lesson_type = 'failure'" in sql
    assert "archived = false" in sql


# --- _global_gain_stats (BON-195) ---

def test_global_gain_stats_returns_mean_and_std():
    conn = _conn_one((0.01, 0.02))
    mean, std = _global_gain_stats(conn)
    assert mean == 0.01
    assert std == 0.02


def test_global_gain_stats_empty_table_returns_zeros():
    """reflection_impact이 비어 있으면 avg()가 NULL을 반환한다 — (0.0, 0.0) 폴백."""
    conn = _conn_one((None, None))
    assert _global_gain_stats(conn) == (0.0, 0.0)


def test_global_gain_stats_no_row_returns_zeros():
    conn = _conn_one(None)
    assert _global_gain_stats(conn) == (0.0, 0.0)


def test_global_gain_stats_queries_reflection_impact_view():
    conn = _conn_one((0.0, 0.0))
    _global_gain_stats(conn)
    sql: str = conn.execute.call_args[0][0]
    assert "reflection_impact" in sql
