"""memory/retriever.py — search_failure_lessons 단위 테스트.

embed()가 필요 없는 순수 SQL 채널이라 mock conn으로 직접 테스트 가능.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from memory.retriever import search_failure_lessons


def _conn(rows: list[tuple]) -> MagicMock:
    mock = MagicMock()
    mock.execute.return_value.fetchall.return_value = rows
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
