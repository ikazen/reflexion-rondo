"""cycle.run._prev_best/_prev_best_params/_prev_best_fold_scores의 확정 pipeline 조회 단위 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock

from cycle.run import _prev_best, _prev_best_fold_scores, _prev_best_params


def _conn_seq(*results) -> MagicMock:
    """순서대로 fetchone 결과를 반환하는 mock conn."""
    mock = MagicMock()
    mock.execute.return_value.fetchone.side_effect = list(results)
    return mock


def test_pipelines_row_returns_pipelines_max():
    """raw.pipelines에 행이 있으면 그 값을 반환한다."""
    conn = _conn_seq((0.91,))
    result = _prev_best(conn, "s4e1")
    assert result == 0.91
    assert conn.execute.call_count == 1


def test_no_confirmed_pipeline_returns_none():
    """확정 파이프라인이 없으면 None — attempts 폴백은 없다(#102).

    이전엔 raw.attempts 전체 최고 cv_score로 폴백했으나(phantom-max), 재측정 없이
    채택된 값이라 자기강화 데드락의 근본 원인이었다(#73). 콜드스타트 대응은
    establish_bootstrap_baseline(#100)/bin/establish_baseline.py(#101)가
    실제 재검증으로 처리한다 — 여기서 폴백으로 봉합하지 않는다.
    """
    conn = _conn_seq((None,))
    result = _prev_best(conn, "s4e1")
    assert result is None
    assert conn.execute.call_count == 1


def test_pipelines_query_uses_pipelines_table():
    conn = _conn_seq((0.9,))
    _prev_best(conn, "s4e1")
    sql: str = conn.execute.call_args_list[0][0][0]
    assert "raw.pipelines" in sql



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



def test_prev_best_fold_scores_returns_list_row():
    conn = _conn_seq(([0.9, 0.91, 0.89],))
    result = _prev_best_fold_scores(conn, "s4e1")
    assert result == [0.9, 0.91, 0.89]
    assert conn.execute.call_count == 1


def test_prev_best_fold_scores_parses_json_string():
    conn = _conn_seq(("[0.9, 0.91, 0.89]",))
    result = _prev_best_fold_scores(conn, "s4e1")
    assert result == [0.9, 0.91, 0.89]


def test_prev_best_fold_scores_no_confirmed_pipeline_returns_none():
    """확정 파이프라인이 없으면 None — attempts 폴백은 없다(#102, 위 test_no_confirmed_pipeline_returns_none와 동일 원칙)."""
    conn = _conn_seq(None)
    result = _prev_best_fold_scores(conn, "s4e1")
    assert result is None
    assert conn.execute.call_count == 1


def test_prev_best_fold_scores_null_fold_scores_returns_none():
    """확정 파이프라인 행은 있는데 fold_scores가 null이면 None."""
    conn = _conn_seq((None,))
    result = _prev_best_fold_scores(conn, "s4e1")
    assert result is None


def test_prev_best_fold_scores_joins_attempts_and_pipelines():
    conn = _conn_seq(([0.9],))
    _prev_best_fold_scores(conn, "s4e1")
    sql: str = conn.execute.call_args_list[0][0][0]
    assert "raw.pipelines" in sql
    assert "raw.attempts" in sql


# invalid_reason 격리 필터 (#99, GH #96)
# 누수로 격리(bin/quarantine_leaks.py)된 pipeline은 baseline/advisory 어디에도
# 쓰이면 안 된다 — 부풀려진 cv_score가 다시 게이트를 오염시키기 때문.

def test_prev_best_query_excludes_invalid_reason():
    conn = _conn_seq((0.9,))
    _prev_best(conn, "s4e1")
    sql: str = conn.execute.call_args_list[0][0][0]
    assert "invalid_reason" in sql.lower()


def test_prev_best_params_query_excludes_invalid_reason():
    conn = _conn_seq(({"a": 1},))
    _prev_best_params(conn, "s4e1")
    sql: str = conn.execute.call_args_list[0][0][0]
    assert "invalid_reason" in sql.lower()


def test_prev_best_fold_scores_primary_query_excludes_invalid_reason():
    conn = _conn_seq(([0.9],))
    _prev_best_fold_scores(conn, "s4e1")
    sql: str = conn.execute.call_args_list[0][0][0]
    assert "invalid_reason" in sql.lower()
