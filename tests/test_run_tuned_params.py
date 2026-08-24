"""cycle.run._latest_tuned_params(#230) — raw.tuned_params advisory 조회 단위 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock

from cycle.run import _latest_tuned_params


def _conn_with(fetchone_result=None, fetchall_result=None) -> MagicMock:
    conn = MagicMock()
    r1 = MagicMock()
    r1.fetchone.return_value = fetchone_result
    r2 = MagicMock()
    r2.fetchall.return_value = fetchall_result or []
    conn.execute.side_effect = [r1, r2]
    return conn


def test_no_tuning_run_returns_none():
    conn = _conn_with(fetchone_result=None)
    assert _latest_tuned_params(conn, "s4e1") is None
    assert conn.execute.call_count == 1


def test_returns_none_when_nothing_improved():
    """튜닝했지만 원본보다 나은 조합을 못 찾았으면 advisory로 넘길 가치가 없다."""
    conn = _conn_with(
        fetchone_result=("run-1",),
        fetchall_result=[("ridge", None, {"alpha": 1.0}, 0.5, False)],
    )
    assert _latest_tuned_params(conn, "s4e1") is None


def test_returns_entries_when_improved():
    conn = _conn_with(
        fetchone_result=("run-1",),
        fetchall_result=[("ridge", None, {"alpha": 2.0}, 0.6, True)],
    )
    result = _latest_tuned_params(conn, "s4e1")
    assert result == {
        "entries": [
            {"model": "ridge", "member_index": None, "params": {"alpha": 2.0}, "cv_score": 0.6, "improved": True},
        ]
    }


def test_parses_json_string_params():
    """jsonb가 driver에서 raw str로 온 경우도 dict로 파싱한다(cycle.run._prev_best_params와 동일 관례)."""
    conn = _conn_with(
        fetchone_result=("run-1",),
        fetchall_result=[("lgbm", 0, '{"n_estimators": 300}', 0.7, True)],
    )
    result = _latest_tuned_params(conn, "s4e1")
    assert result["entries"][0]["params"] == {"n_estimators": 300}


def test_multi_member_entries_include_both_improved_and_not():
    """일부 멤버만 개선돼도(전부 개선 아니어도) 개선된 항목이 하나라도 있으면
    전체 멤버 목록을 반환한다 — LLM이 어느 게 개선됐는지 improved 필드로 직접 판단."""
    conn = _conn_with(
        fetchone_result=("run-1",),
        fetchall_result=[
            ("ridge", 0, {"alpha": 1.0}, 0.5, False),
            ("lgbm", 1, {"n_estimators": 300}, 0.7, True),
        ],
    )
    result = _latest_tuned_params(conn, "s4e1")
    assert len(result["entries"]) == 2
    assert result["entries"][0]["improved"] is False
    assert result["entries"][1]["improved"] is True


def test_query_uses_latest_tuning_run_id():
    conn = _conn_with(fetchone_result=("run-1",), fetchall_result=[])
    _latest_tuned_params(conn, "s4e1")
    first_sql = conn.execute.call_args_list[0][0][0]
    second_sql = conn.execute.call_args_list[1][0][0]
    assert "order by created_at desc" in first_sql
    assert "tuning_run_id" in second_sql
