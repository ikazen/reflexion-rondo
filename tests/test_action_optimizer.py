"""cycle.action_optimizer 밴딧(assign_super_cycle_actions/update_bandit/get_action_prior) 단위 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config.settings import ACTION_TYPES
from cycle.action_optimizer import (
    _BANDIT_DECAY,
    _HALF_SUCCESS,
    _NEUTRAL_INCREMENT,
    assign_super_cycle_actions,
    get_action_prior,
    update_bandit,
)


def _conn(rows: list[tuple] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.execute.return_value.fetchall.return_value = rows or []
    return mock



def test_assign_returns_n_attempts():
    conn = _conn()
    result = assign_super_cycle_actions(conn, "s4e1", n_attempts=3)
    assert len(result) == 3


def test_assign_elements_are_action_types():
    conn = _conn()
    result = assign_super_cycle_actions(conn, "s4e1", n_attempts=3)
    assert all(a in ACTION_TYPES for a in result)


def test_assign_no_duplicates():
    conn = _conn()
    result = assign_super_cycle_actions(conn, "s4e1", n_attempts=3)
    assert len(result) == len(set(result))


def test_assign_empty_bandit_returns_uniform():
    conn = _conn(rows=[])
    result = assign_super_cycle_actions(conn, "s4e1", n_attempts=3)
    assert len(result) == 3
    assert all(a in ACTION_TYPES for a in result)


def test_assign_exploration_not_deterministic():
    """동률 posterior에서 seed=None이면 랭킹이 매번 같지 않다 — 탐색 복원 핵심 회귀 방지."""
    seen: set[tuple[str, ...]] = set()
    for _ in range(30):
        conn = _conn(rows=[])
        seen.add(tuple(assign_super_cycle_actions(conn, "s4e1", n_attempts=3)))
    assert len(seen) > 1, "탐색성 소멸 — 항상 동일한 랭킹 반환 (고정 시드 버그 재발)"


def test_assign_strong_posterior_dominates():
    """한 action만 강한 posterior면 다수 시행에서 top-1로 안정 선택."""
    top1_counts: dict[str, int] = {a: 0 for a in ACTION_TYPES}
    rows = [
        ("feature_engineering", 100.0, 1.0),  # 강한 posterior
        ("model_swap", 1.0, 1.0),
        ("hyperparam_search", 1.0, 1.0),
        ("preprocessing", 1.0, 1.0),
        ("ensemble", 1.0, 1.0),
    ]
    for _ in range(50):
        conn = _conn(rows=rows)
        top1 = assign_super_cycle_actions(conn, "s4e1", n_attempts=1)[0]
        top1_counts[top1] += 1
    assert top1_counts["feature_engineering"] > 35, "강한 posterior action이 top-1로 자주 안 뽑힘"


def test_assign_seed_makes_deterministic():
    """seed를 고정하면 결과가 항상 동일 (테스트 제어성)."""
    rows = []
    r1 = assign_super_cycle_actions(_conn(rows=rows), "s4e1", n_attempts=3, seed=42)
    r2 = assign_super_cycle_actions(_conn(rows=rows), "s4e1", n_attempts=3, seed=42)
    assert r1 == r2



def _update(label: str, gain: float | None, error: str | None, action: str = "feature_engineering"):
    conn = _conn()
    update_bandit(conn, "s4e1", action, label, gain, error)
    return conn


def test_update_error_trace_increments_beta():
    conn = _update("neutral", None, "SyntaxError: invalid syntax")
    params = conn.execute.call_args[0][1]
    da, db = params[3], params[4]
    assert da == 0.0 and db == 1.0


def test_update_regression_label_increments_beta():
    conn = _update("regression", -0.01, None)
    params = conn.execute.call_args[0][1]
    da, db = params[3], params[4]
    assert da == 0.0 and db == 1.0


def test_update_jump_label_increments_alpha():
    conn = _update("jump", 0.02, None)
    params = conn.execute.call_args[0][1]
    da, db = params[3], params[4]
    assert da == 1.0 and db == 0.0


def test_update_positive_gain_is_half_success():
    """유의미 미달 양수 gain은 full win이 아닌 half-success(_HALF_SUCCESS)."""
    conn = _update("neutral", 0.005, None)
    params = conn.execute.call_args[0][1]
    da, db = params[3], params[4]
    assert da == pytest.approx(_HALF_SUCCESS) and db == pytest.approx(_NEUTRAL_INCREMENT)


def test_update_neutral_increments_both_small():
    conn = _update("neutral", 0.0, None)
    params = conn.execute.call_args[0][1]
    da, db = params[3], params[4]
    assert da == pytest.approx(0.1) and db == pytest.approx(0.1)


def test_update_jump_is_full_win_not_positive_gain():
    """jump만 full win(1.0). gain_vs_best > 0이어도 jump 아니면 half-success."""
    conn_jump = _update("jump", 0.02, None)
    conn_neutral_gain = _update("neutral", 0.02, None)
    params_jump = conn_jump.execute.call_args[0][1]
    params_ng = conn_neutral_gain.execute.call_args[0][1]
    assert params_jump[3] == pytest.approx(1.0) and params_jump[4] == pytest.approx(0.0)
    assert params_ng[3] == pytest.approx(_HALF_SUCCESS) and params_ng[4] == pytest.approx(_NEUTRAL_INCREMENT)


def test_update_decay_in_sql():
    """ON CONFLICT SQL에 decay factor(0.95)가 포함된다."""
    conn = _update("jump", 0.02, None)
    sql: str = conn.execute.call_args[0][0]
    assert "0.95" in sql


def test_update_decay_constant_value():
    assert _BANDIT_DECAY == pytest.approx(0.95)


def test_update_unknown_action_type_skips_db():
    conn = _conn()
    update_bandit(conn, "s4e1", "not_a_real_action", "jump", 0.1, None)
    conn.execute.assert_not_called()



def test_get_action_prior_returns_all_action_types():
    result = get_action_prior(_conn(), "s4e1")
    assert set(result.keys()) == set(ACTION_TYPES)


def test_get_action_prior_values_in_unit_interval():
    result = get_action_prior(_conn(), "s4e1")
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_get_action_prior_empty_bandit_is_uniform_beta():
    """빈 DB → Beta(1,1) draw → 모든 action에 유효한 값."""
    result = get_action_prior(_conn(rows=[]), "s4e1")
    assert len(result) == len(ACTION_TYPES)
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_get_action_prior_seed_deterministic():
    r1 = get_action_prior(_conn(), "s4e1", seed=7)
    r2 = get_action_prior(_conn(), "s4e1", seed=7)
    assert r1 == r2
