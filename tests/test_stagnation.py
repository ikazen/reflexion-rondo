"""cycle.stagnation.detect_stagnation의 정체 판정(연속 에러/jump 리셋) 단위 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock

from cycle.stagnation import _STAGNANT_THRESHOLD, detect_stagnation


def _conn(rows: list[tuple]) -> MagicMock:
    mock = MagicMock()
    mock.execute.return_value.fetchall.return_value = rows
    return mock


def test_all_error_attempts_trigger_stagnation():
    """연속 크래시(cv_score=NULL, label='regression') 윈도우가 is_stagnant=True를 반환한다."""
    rows = [(True, "regression", "feature_engineering")] * _STAGNANT_THRESHOLD
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.is_stagnant is True
    assert sig.stagnant_for >= _STAGNANT_THRESHOLD


def test_stagnant_for_counts_error_attempts():
    rows = [(True, "regression", "model_swap")] * 5
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.stagnant_for == 5


def test_jump_resets_stagnant_for():
    """jump이 섞이면 jump 이후 비-jump 횟수만 stagnant_for에 반영된다."""
    rows = [
        (True, "regression", "feature_engineering"),
        (True, "regression", "model_swap"),
        (True, "jump", "hyperparam_search"),
        (True, "regression", "ensemble"),
    ]
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.stagnant_for == 2


def test_empty_window_not_stagnant():
    sig = detect_stagnation(_conn([]), "s4e1")
    assert sig.is_stagnant is False
    assert sig.stagnant_for == 0


def test_single_jump_not_stagnant():
    rows = [(True, "jump", "feature_engineering")]
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.is_stagnant is False
    assert sig.stagnant_for == 0


def test_neutral_attempts_count_toward_stagnation():
    rows = [(True, "neutral", "feature_engineering")] * _STAGNANT_THRESHOLD
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.is_stagnant is True


def test_query_does_not_filter_cv_score():
    """cv_score is not null 필터가 쿼리에 없어야 한다 — 에러 attempt 포함 확인."""
    conn = _conn([(True, "regression", "model_swap")] * 3)
    detect_stagnation(conn, "s4e1")
    sql: str = conn.execute.call_args[0][0]
    assert "cv_score" not in sql


def test_loser_action_types_count_as_used():
    """패자로 시도된 action_type도 underused_actions에서 제외돼야 한다.

    승자는 매번 hyperparam_search뿐이지만, 패자로 다른 action_type들이 이미
    시도됐다면 그것들을 '저활용'으로 다시 추천하면 안 된다.
    """
    rows = [
        (True, "neutral", "hyperparam_search"),
        (False, "neutral", "feature_engineering"),
        (False, "neutral", "model_swap"),
        (True, "neutral", "hyperparam_search"),
        (False, "neutral", "preprocessing"),
        (False, "neutral", "ensemble"),
    ]
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.underused_actions == ()


def test_loser_jump_does_not_count_as_jump():
    """패자의 jump 라벨은 jumps_in_window/stagnant_for에 반영되지 않는다 — 승자 이력만 본다."""
    rows = [
        (False, "jump", "feature_engineering"),
        (True, "regression", "hyperparam_search"),
        (True, "regression", "hyperparam_search"),
        (True, "regression", "hyperparam_search"),
    ]
    sig = detect_stagnation(_conn(rows), "s4e1")
    assert sig.jumps_in_window == 0
    assert sig.stagnant_for == 3
    assert sig.is_stagnant is True
