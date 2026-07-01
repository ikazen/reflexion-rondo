from __future__ import annotations

from unittest.mock import MagicMock, call

from cycle.run import _prev_best


def _conn_seq(*results) -> MagicMock:
    """순서대로 fetchone 결과를 반환하는 mock conn."""
    mock = MagicMock()
    mock.execute.return_value.fetchone.side_effect = list(results)
    return mock


def test_pipelines_row_returns_pipelines_max():
    """raw.pipelines에 행이 있으면 그 값을 반환하고 attempts는 조회하지 않는다."""
    conn = _conn_seq((0.91,))
    result = _prev_best(conn, "s4e1")
    assert result == 0.91
    assert conn.execute.call_count == 1


def test_pipelines_null_falls_back_to_attempts():
    """raw.pipelines가 비어(None) cold-start면 raw.attempts max로 폴백한다."""
    conn = _conn_seq((None,), (0.88,))
    result = _prev_best(conn, "s4e1")
    assert result == 0.88
    assert conn.execute.call_count == 2


def test_both_empty_returns_none():
    """pipelines도 attempts도 비어 있으면 None 반환."""
    conn = _conn_seq((None,), (None,))
    result = _prev_best(conn, "s4e1")
    assert result is None


def test_pipelines_query_uses_pipelines_table():
    """첫 번째 쿼리가 raw.pipelines를 참조한다."""
    conn = _conn_seq((0.9,))
    _prev_best(conn, "s4e1")
    first_sql: str = conn.execute.call_args_list[0][0][0]
    assert "raw.pipelines" in first_sql
    assert "raw.attempts" not in first_sql


def test_fallback_query_uses_attempts_table():
    """폴백 쿼리가 raw.attempts를 참조한다."""
    conn = _conn_seq((None,), (0.85,))
    _prev_best(conn, "s4e1")
    fallback_sql: str = conn.execute.call_args_list[1][0][0]
    assert "raw.attempts" in fallback_sql
