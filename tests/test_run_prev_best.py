from __future__ import annotations

from unittest.mock import MagicMock, call

from cycle.run import _prev_best, _prev_best_fold_scores, _prev_best_params


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


# --- _prev_best_params ---

def test_prev_best_params_returns_dict_row():
    conn = _conn_seq(({"max_depth": 4},))
    result = _prev_best_params(conn, "s4e1")
    assert result == {"max_depth": 4}


def test_prev_best_params_parses_json_string():
    """jsonb가 driver에서 raw str로 온 경우도 dict로 파싱한다."""
    conn = _conn_seq(('{"max_depth": 4}',))
    result = _prev_best_params(conn, "s4e1")
    assert result == {"max_depth": 4}


def test_prev_best_params_no_row_returns_none():
    conn = _conn_seq(None)
    result = _prev_best_params(conn, "s4e1")
    assert result is None


def test_prev_best_params_null_params_returns_none():
    """raw.pipelines 행은 있으나 attempts.params가 null인 경우."""
    conn = _conn_seq((None,))
    result = _prev_best_params(conn, "s4e1")
    assert result is None


def test_prev_best_params_joins_attempts_and_pipelines():
    conn = _conn_seq(({"a": 1},))
    _prev_best_params(conn, "s4e1")
    sql: str = conn.execute.call_args_list[0][0][0]
    assert "raw.pipelines" in sql
    assert "raw.attempts" in sql


# --- _prev_best_fold_scores ---

def test_prev_best_fold_scores_returns_list_row():
    """확정 파이프라인이 있으면 그 값을 쓰고 폴백 쿼리는 안 나간다."""
    conn = _conn_seq(([0.9, 0.91, 0.89],))
    result = _prev_best_fold_scores(conn, "s4e1")
    assert result == [0.9, 0.91, 0.89]
    assert conn.execute.call_count == 1


def test_prev_best_fold_scores_parses_json_string():
    conn = _conn_seq(("[0.9, 0.91, 0.89]",))
    result = _prev_best_fold_scores(conn, "s4e1")
    assert result == [0.9, 0.91, 0.89]


def test_prev_best_fold_scores_no_row_falls_back_to_attempts():
    """확정 파이프라인이 아예 없으면(콜드스타트) attempts 최고 cv_score로 폴백한다(#73).

    폴백이 없으면 paired 유의성 검정이 영원히 비활성화돼 절대-margin 폴백(사실상
    도달 불가)만 남는 자기강화 데드락에 빠진다.
    """
    conn = _conn_seq(None, ([0.8, 0.82, 0.79],))
    result = _prev_best_fold_scores(conn, "s4e1")
    assert result == [0.8, 0.82, 0.79]
    assert conn.execute.call_count == 2


def test_prev_best_fold_scores_null_falls_back_to_attempts():
    """확정 파이프라인 행은 있는데 fold_scores가 null이어도 폴백한다."""
    conn = _conn_seq((None,), ([0.8],))
    result = _prev_best_fold_scores(conn, "s4e1")
    assert result == [0.8]


def test_prev_best_fold_scores_both_empty_returns_none():
    """confirmed도 attempts 폴백도 다 없으면 None."""
    conn = _conn_seq(None, None)
    result = _prev_best_fold_scores(conn, "s4e1")
    assert result is None


def test_prev_best_fold_scores_fallback_query_filters_errors_and_nulls():
    """폴백 쿼리는 error_trace가 없고 fold_scores가 non-null인 attempt만 대상으로 한다."""
    conn = _conn_seq(None, ([0.8],))
    _prev_best_fold_scores(conn, "s4e1")
    fallback_sql: str = conn.execute.call_args_list[1][0][0]
    assert "raw.attempts" in fallback_sql
    assert "error_trace is null" in fallback_sql
    assert "fold_scores is not null" in fallback_sql


def test_prev_best_fold_scores_exclude_attempt_id_filters_self_comparison():
    """exclude_attempt_id를 주면 폴백 쿼리에 그 attempt를 제외하는 조건과 파라미터가 들어간다.

    bin/run_promote_task.py는 이미 DB에 커밋된 winner 자신과 비교할 위험이 있어
    (콜드스타트에서 winner가 곧 역대 최고 attempt인 경우가 흔함) 자기 자신을
    제외해야 한다(#73).
    """
    conn = _conn_seq(None, ([0.8],))
    _prev_best_fold_scores(conn, "s4e1", exclude_attempt_id="winner-123")
    fallback_call = conn.execute.call_args_list[1]
    fallback_sql: str = fallback_call[0][0]
    fallback_params: list = fallback_call[0][1]
    assert "attempt_id !=" in fallback_sql
    assert "winner-123" in fallback_params


def test_prev_best_fold_scores_joins_attempts_and_pipelines():
    conn = _conn_seq(([0.9],))
    _prev_best_fold_scores(conn, "s4e1")
    sql: str = conn.execute.call_args_list[0][0][0]
    assert "raw.pipelines" in sql
    assert "raw.attempts" in sql
