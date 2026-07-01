from __future__ import annotations

from unittest.mock import MagicMock

from cycle.stagnation import _STAGNANT_THRESHOLD, detect_stagnation


def _conn(rows: list[tuple]) -> MagicMock:
    mock = MagicMock()
    mock.execute.return_value.fetchall.return_value = rows
    return mock


def test_all_error_attempts_trigger_stagnation():
    """연속 크래시(cv_score=NULL, label='regression') 윈도우가 is_stagnant=True를 반환한다."""
    rows = [("regression", "feature_engineering")] * _STAGNANT_THRESHOLD
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.is_stagnant is True
    assert sig.stagnant_for >= _STAGNANT_THRESHOLD


def test_stagnant_for_counts_error_attempts():
    rows = [("regression", "model_swap")] * 5
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.stagnant_for == 5


def test_jump_resets_stagnant_for():
    """jump이 섞이면 jump 이후 비-jump 횟수만 stagnant_for에 반영된다."""
    rows = [
        ("regression", "feature_engineering"),
        ("regression", "model_swap"),
        ("jump", "hyperparam_search"),
        ("regression", "ensemble"),
    ]
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.stagnant_for == 2


def test_empty_window_not_stagnant():
    sig = detect_stagnation(_conn([]), "s4e1")
    assert sig.is_stagnant is False
    assert sig.stagnant_for == 0


def test_single_jump_not_stagnant():
    rows = [("jump", "feature_engineering")]
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.is_stagnant is False
    assert sig.stagnant_for == 0


def test_neutral_attempts_count_toward_stagnation():
    rows = [("neutral", "feature_engineering")] * _STAGNANT_THRESHOLD
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.is_stagnant is True


def test_query_does_not_filter_cv_score():
    """cv_score is not null 필터가 쿼리에 없어야 한다 — 에러 attempt 포함 확인."""
    conn = _conn([("regression", "model_swap")] * 3)
    detect_stagnation(conn, "s4e1")
    sql: str = conn.execute.call_args[0][0]
    assert "cv_score" not in sql
