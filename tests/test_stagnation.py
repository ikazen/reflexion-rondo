"""cycle.stagnation.detect_stagnation의 정체 판정(연속 에러/jump 리셋) 단위 테스트.

#215: 승자 조회가 raw attempts(window*3)에서 파생되지 않고 `was_promoted=true`를 직접
LIMIT window로 조회한다 — SQL 텍스트로 라우팅하는 fake conn을 쓴다.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from cycle.stagnation import _STAGNANT_THRESHOLD, detect_stagnation


class _Cur:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple]:
        return self._rows


class _Conn:
    """action_type 커버리지 쿼리와 winner 쿼리를 SQL 텍스트로 구분해 각자 다른 행을 준다."""

    def __init__(self, action_rows: list[tuple] = (), winner_rows: list[tuple] = ()) -> None:
        self.action_rows = list(action_rows)
        self.winner_rows = list(winner_rows)
        self.executed: list[str] = []

    def execute(self, sql: str, params=None) -> _Cur:
        s = " ".join(sql.split())
        self.executed.append(s)
        if "was_promoted = true" in s:
            return _Cur(self.winner_rows)
        return _Cur(self.action_rows)


def _conn(winner_rows: list[tuple], action_rows: list[tuple] | None = None) -> _Conn:
    """대부분의 테스트는 승자 이력(jump/stagnant_for)만 신경 쓴다 — action_rows를 안
    주면 더미 한 행으로 채워 "action_type 커버리지 안 비어있음" 조건만 맞춘다."""
    return _Conn(action_rows=action_rows or [("model_swap",)], winner_rows=winner_rows)


def _winner_labels(rows: list[tuple]) -> list[tuple]:
    """(is_winner, label, action_type) 3튜플에서 승자만 골라 (label,) 1튜플로 변환 —
    기존 테스트가 쓰던 3컬럼 표현을 그대로 재사용하기 위한 헬퍼."""
    return [(label,) for was_promoted, label, _action in rows if was_promoted]


def test_all_error_attempts_trigger_stagnation():
    """연속 크래시(cv_score=NULL, label='regression') 윈도우가 is_stagnant=True를 반환한다."""
    rows = [(True, "regression", "feature_engineering")] * _STAGNANT_THRESHOLD
    sig = detect_stagnation(_conn(_winner_labels(rows)), "s4e1")
    assert sig.is_stagnant is True
    assert sig.stagnant_for >= _STAGNANT_THRESHOLD


def test_stagnant_for_counts_error_attempts():
    rows = [(True, "regression", "model_swap")] * 5
    sig = detect_stagnation(_conn(_winner_labels(rows)), "s4e1")
    assert sig.stagnant_for == 5


def test_jump_resets_stagnant_for():
    """jump이 섞이면 jump 이후 비-jump 횟수만 stagnant_for에 반영된다."""
    rows = [
        (True, "regression", "feature_engineering"),
        (True, "regression", "model_swap"),
        (True, "jump", "hyperparam_search"),
        (True, "regression", "ensemble"),
    ]
    sig = detect_stagnation(_conn(_winner_labels(rows)), "s4e1")
    assert sig.stagnant_for == 2


def test_empty_window_not_stagnant():
    sig = detect_stagnation(_Conn(action_rows=[], winner_rows=[]), "s4e1")
    assert sig.is_stagnant is False
    assert sig.stagnant_for == 0


def test_single_jump_not_stagnant():
    rows = [(True, "jump", "feature_engineering")]
    sig = detect_stagnation(_conn(_winner_labels(rows)), "s4e1")
    assert sig.is_stagnant is False
    assert sig.stagnant_for == 0


def test_neutral_attempts_count_toward_stagnation():
    rows = [(True, "neutral", "feature_engineering")] * _STAGNANT_THRESHOLD
    sig = detect_stagnation(_conn(_winner_labels(rows)), "s4e1")
    assert sig.is_stagnant is True


def test_query_does_not_filter_cv_score():
    """cv_score is not null 필터가 쿼리에 없어야 한다 — 에러 attempt 포함 확인."""
    rows = [(True, "regression", "model_swap")] * 3
    conn = _conn(_winner_labels(rows))
    detect_stagnation(conn, "s4e1")
    assert all("cv_score" not in sql for sql in conn.executed)


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
    action_rows = [(action,) for _, _, action in rows]
    conn = _Conn(action_rows=action_rows, winner_rows=_winner_labels(rows))
    sig = detect_stagnation(conn, "s4e1")
    assert sig.underused_actions == ()


def test_loser_jump_does_not_count_as_jump():
    """패자의 jump 라벨은 jumps_in_window/stagnant_for에 반영되지 않는다 — 승자 이력만 본다."""
    rows = [
        (False, "jump", "feature_engineering"),
        (True, "regression", "hyperparam_search"),
        (True, "regression", "hyperparam_search"),
        (True, "regression", "hyperparam_search"),
    ]
    sig = detect_stagnation(_conn(_winner_labels(rows)), "s4e1")
    assert sig.jumps_in_window == 0
    assert sig.stagnant_for == 3
    assert sig.is_stagnant is True


def test_winners_queried_directly_not_derived_from_window_times_three():
    """#215 회귀 방지 — window*3(15)개 raw attempt 안에 승자가 window(5)개보다 적어도
    (사이클당 확정 0~1개인 gate 도입 이후 실제 상황), 승자 쿼리 자체가 LIMIT window로
    독립적으로 도니 stagnant_for가 과소산정되지 않는다."""
    # raw attempts 15개 중 승자는 3개뿐(과거엔 winner_labels가 이 3개로 잘림) —
    # 하지만 실제 DB에는 그 이전에도 승자가 더 있고, winner 쿼리는 그걸 별도로 본다.
    action_rows = [("model_swap",)] * 15
    winner_rows = [
        ("regression",), ("regression",), ("regression",),  # 최근 raw 15개 안의 승자 3개
        ("regression",),  # window*3 밖에 있었을 승자 4번째 — 별도 쿼리라 여전히 잡힘
    ]
    conn = _Conn(action_rows=action_rows, winner_rows=winner_rows)
    sig = detect_stagnation(conn, "s4e1", window=5)
    assert sig.stagnant_for == 4
    assert sig.is_stagnant is True
