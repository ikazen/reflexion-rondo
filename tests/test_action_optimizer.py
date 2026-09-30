"""cycle.action_optimizer 밴딧(assign_super_cycle_actions/update_bandit/get_action_prior) 단위 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config.settings import ACTION_TYPES
from cycle.action_optimizer import (
    _BANDIT_DECAY,
    _DEAD_ACTION_MEAN_THRESHOLD,
    _DEAD_ACTION_MIN_OBSERVED,
    _DEAD_ACTION_REEXPLORE_EVERY,
    _HALF_SUCCESS,
    _NEUTRAL_INCREMENT,
    _NOOP_TIE_PENALTY,
    _is_dead_action,
    _reexplore_window_open,
    assign_super_cycle_actions,
    bandit_deltas,
    get_action_prior,
    update_bandit,
)


def _conn(rows: list[tuple] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.execute.return_value.fetchall.return_value = rows or []
    return mock


def _conn_with_dead_action(action: str, total_attempts: int) -> MagicMock:
    """1번째 execute=bandit 행(해당 액션만 죽음), 2번째 execute=competition 전체 attempt 수."""
    conn = MagicMock()
    bandit_result = MagicMock()
    bandit_result.fetchall.return_value = [(action, 1.0, 20.0)]  # mean=1/21≈0.048 < 0.1, observed=18>=5
    count_result = MagicMock()
    count_result.fetchone.return_value = (total_attempts,)
    conn.execute.side_effect = [bandit_result, count_result]
    return conn



def _conn_with_dead_actions(actions: list[str], total_attempts: int) -> MagicMock:
    """_conn_with_dead_action의 복수 버전 — 여러 액션이 동시에 죽은 상태."""
    conn = MagicMock()
    bandit_result = MagicMock()
    bandit_result.fetchall.return_value = [(a, 1.0, 20.0) for a in actions]
    count_result = MagicMock()
    count_result.fetchone.return_value = (total_attempts,)
    conn.execute.side_effect = [bandit_result, count_result]
    return conn


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


# 죽은 액션 배제 + 주기적 재탐색(#375)

@pytest.mark.parametrize(("alpha", "beta", "expected"), [
    (1.0, 20.0, True),    # mean=0.048<0.1, observed=19>=5 -> dead
    (1.0, 3.0, False),    # mean=0.25(문턱 위) -> 관측과 무관하게 dead 아님
    (0.05, 9.55, True),   # 인위적 값(정상 갱신으로는 거의 안 나오지만 함수 자체의 경계는 확인) mean=0.005<0.1, observed=8.55>=5 -> dead
    (0.05, 0.45, False),  # mean=0.1(문턱 정확히 미만 아님) 전에, observed=-0.5<5 -> dead 아님(둘 다 불충족)
    (1.0, 1.0, False),    # 순수 prior(관측 0) -> dead 아님
])
def test_is_dead_action_boundary(alpha, beta, expected):
    assert _is_dead_action(alpha, beta) is expected


# 죽은 액션이 정확히 1개일 때 n_attempts를 len(ACTION_TYPES)-1로 두면 eligible 후보 수가
# n_attempts와 정확히 같아져(부족분을 못 채워 폴백이 개입할 여지도, 남는 후보라 랭킹 운이
# 끼어들 여지도 없이) 배제 여부만 결정적으로 드러난다.
_N_ATTEMPTS_ONE_DEAD = len(ACTION_TYPES) - 1


def test_dead_action_is_excluded_when_mean_is_low_and_observed_enough():
    conn = _conn_with_dead_action("hyperparam_search", total_attempts=50)
    result = assign_super_cycle_actions(conn, "s6e8", n_attempts=_N_ATTEMPTS_ONE_DEAD, seed=42)
    assert "hyperparam_search" not in result
    assert len(result) == _N_ATTEMPTS_ONE_DEAD


@pytest.mark.parametrize("total_attempts", [0, 1, 99, 103, 199])
def test_dead_action_excluded_outside_the_reexplore_window(total_attempts):
    conn = _conn_with_dead_action("hyperparam_search", total_attempts=total_attempts)
    result = assign_super_cycle_actions(conn, "s6e8", n_attempts=_N_ATTEMPTS_ONE_DEAD, seed=42)
    assert "hyperparam_search" not in result


# 죽은 액션이 3개 이상이면 살아 있는 후보가 n_attempts(3)에 모자란다(#404, s6e8 실측).
_THREE_DEAD = ["ensemble", "hyperparam_search", "model_swap"]


def test_shortfall_is_filled_with_live_actions_not_dead_ones():
    conn = _conn_with_dead_actions(_THREE_DEAD, total_attempts=50)
    result = assign_super_cycle_actions(conn, "s6e8", n_attempts=3, seed=42)
    assert len(result) == 3
    assert not set(result) & set(_THREE_DEAD)
    assert set(result) == {"feature_engineering", "preprocessing"}


def test_shortfall_repeats_the_best_ranked_live_action():
    conn = _conn_with_dead_actions(_THREE_DEAD, total_attempts=50)
    result = assign_super_cycle_actions(conn, "s6e8", n_attempts=3, seed=42)
    assert result[0] != result[1]
    assert result[2] == result[0]


def test_all_dead_falls_back_to_the_original_ranking():
    """살아 있는 후보가 하나도 없으면 배정 자체를 못 하는 것보다 낫도록 원래 랭킹에서 채운다."""
    conn = _conn_with_dead_actions(list(ACTION_TYPES), total_attempts=50)
    result = assign_super_cycle_actions(conn, "s6e8", n_attempts=3, seed=42)
    assert len(result) == 3
    assert len(set(result)) == 3


def test_reexplore_window_restores_dead_actions_even_when_three_are_dead():
    conn = _conn_with_dead_actions(_THREE_DEAD, total_attempts=100)
    result = assign_super_cycle_actions(conn, "s6e8", n_attempts=3, seed=42)
    assert len(set(result)) == 3


def _rng_favoring(action: str):
    """action의 Thompson 표본만 압도적으로 높게 강제한다 — 배제되지 않는 한 반드시 뽑히도록
    랭킹 운을 제거하고 필터 자체의 동작만 본다."""
    class _FakeRng:
        def beta(self, a, b):
            return 0.99 if (a, b) == (1.0, 20.0) else 0.01
    return lambda seed=None: _FakeRng()


@pytest.mark.parametrize("total_attempts", [100, 200])  # n_attempts=1이라 창 폭도 1 — total % 100 == 0인 정확한 그 틱
def test_dead_action_reincluded_inside_the_reexplore_window(monkeypatch, total_attempts):
    """재탐색 창이 열리면 죽은 액션도 다시 후보가 된다 — Thompson 표본을 모의로 압도적으로
    강제해(n_attempts=1) 랭킹 운이 아니라 배제 자체의 해제를 확인한다."""
    monkeypatch.setattr("cycle.action_optimizer.np.random.default_rng", _rng_favoring("hyperparam_search"))
    conn = _conn_with_dead_action("hyperparam_search", total_attempts=total_attempts)
    result = assign_super_cycle_actions(conn, "s6e8", n_attempts=1, seed=42)
    assert result == ["hyperparam_search"]


@pytest.mark.parametrize("total_attempts", [1, 99, 101, 103, 199, 201])
def test_dead_action_excluded_outside_the_reexplore_window_even_with_a_favorable_draw(monkeypatch, total_attempts):
    """창이 닫혀 있으면 표본이 압도적으로 높아도(모의) 배제된다."""
    monkeypatch.setattr("cycle.action_optimizer.np.random.default_rng", _rng_favoring("hyperparam_search"))
    conn = _conn_with_dead_action("hyperparam_search", total_attempts=total_attempts)
    result = assign_super_cycle_actions(conn, "s6e8", n_attempts=1, seed=42)
    assert result != ["hyperparam_search"]


def test_reexplore_window_query_is_skipped_when_nothing_is_dead():
    """죽은 액션이 없으면 count 쿼리 자체를 안 던진다 — 매 배정마다 불필요한 쿼리를 늘리지 않는다."""
    conn = _conn(rows=[])
    assign_super_cycle_actions(conn, "s4e1", n_attempts=3, seed=42)
    assert conn.execute.call_count == 1


def test_reexplore_window_count_query_is_scoped_to_the_competition():
    conn = _conn_with_dead_action("hyperparam_search", total_attempts=50)
    assign_super_cycle_actions(conn, "s6e8", n_attempts=1, seed=42)
    count_call = conn.execute.call_args_list[1]
    assert "competition_id" in count_call.args[0]
    assert count_call.args[1] == ["s6e8"]


def test_reexplore_constants_keep_a_full_window_within_the_period():
    assert _DEAD_ACTION_REEXPLORE_EVERY > 3  # n_attempts(기본 3) 창이 다음 배수 전에 끝나야 한다


@pytest.mark.parametrize(("total", "n_attempts", "expected"), [
    (0, 3, True), (2, 3, True), (3, 3, False), (99, 3, False),
    (100, 3, True), (102, 3, True), (103, 3, False),
    (200, 4, True), (203, 4, True), (204, 4, False),
])
def test_reexplore_window_open_boundaries(total, n_attempts, expected):
    """_DEAD_ACTION_REEXPLORE_EVERY 값 자체와 경계 비교 연산자를 Thompson 샘플링과 분리해서 확인한다."""
    assert _reexplore_window_open(total, n_attempts) is expected


def test_dead_action_is_filtered_out_even_when_its_raw_draw_would_rank_first(monkeypatch):
    """(1.0, 20.0)의 Thompson 표본이 우연히 1등이어도(모의로 강제) 배제 필터가 걸러야 한다 —
    랭킹이 우연히 정답과 같아지는 경우를 배제하고 필터 자체의 존재를 증명한다."""
    class _FakeRng:
        def beta(self, a, b):
            return 0.99 if (a, b) == (1.0, 20.0) else 0.01

    monkeypatch.setattr("cycle.action_optimizer.np.random.default_rng", lambda seed=None: _FakeRng())
    conn = _conn_with_dead_action("hyperparam_search", total_attempts=50)
    result = assign_super_cycle_actions(conn, "s6e8", n_attempts=1, seed=42)
    assert result == ["feature_engineering"]  # 나머지 4개가 동점(0.01)일 때 ACTION_TYPES 선언 순서상 첫 번째



def _update(
    label: str,
    gain: float | None,
    error: str | None,
    action: str = "feature_engineering",
    is_noop_tie: bool = False,
):
    conn = _conn()
    update_bandit(conn, "s4e1", action, label, gain, error, is_noop_tie=is_noop_tie)
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


def test_update_noop_tie_penalizes_beta_weaker_than_failure():
    """no-op tie는 실패(β+=1.0)보다 약하지만 neutral(β+=0.1)보다 뚜렷한 페널티."""
    conn = _update("neutral", 0.0, None, is_noop_tie=True)
    params = conn.execute.call_args[0][1]
    da, db = params[3], params[4]
    assert da == 0.0 and db == pytest.approx(_NOOP_TIE_PENALTY)
    assert db > _NEUTRAL_INCREMENT


def test_update_noop_tie_overrides_jump_label():
    """tie면 label이 어떻게 찍혀 있든(방어적으로) neutral 취급 안 받는다."""
    conn = _update("jump", 0.0, None, is_noop_tie=True)
    params = conn.execute.call_args[0][1]
    da, db = params[3], params[4]
    assert da == 0.0 and db == pytest.approx(_NOOP_TIE_PENALTY)


def test_update_noop_tie_default_is_false():
    """is_noop_tie 생략 시 기존 neutral 동작 그대로(하위 호환)."""
    conn = _update("neutral", 0.0, None)
    params = conn.execute.call_args[0][1]
    da, db = params[3], params[4]
    assert da == pytest.approx(_NEUTRAL_INCREMENT) and db == pytest.approx(_NEUTRAL_INCREMENT)


def test_update_decay_is_passed_as_a_parameter_not_a_sql_literal():
    """ON CONFLICT의 decay는 _BANDIT_DECAY를 SQL 파라미터로 받는다 — 리터럴이 상수와 따로 놀지 않게(#420)."""
    conn = _update("jump", 0.02, None)
    sql, params = conn.execute.call_args[0]
    assert "0.95" not in sql
    assert params[5] == pytest.approx(_BANDIT_DECAY) and params[7] == pytest.approx(_BANDIT_DECAY)
    assert (params[6], params[8]) == (params[3], params[4])


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


@pytest.mark.parametrize(
    ("label", "gain", "error", "tie", "expected"),
    [
        ("neutral", None, "SyntaxError", False, (0.0, 1.0)),
        ("regression", -0.01, None, False, (0.0, 1.0)),
        ("jump", 0.02, None, True, (0.0, _NOOP_TIE_PENALTY)),
        ("jump", 0.02, None, False, (1.0, 0.0)),
        ("neutral", 0.005, None, False, (_HALF_SUCCESS, _NEUTRAL_INCREMENT)),
        ("neutral", 0.0, None, False, (_NEUTRAL_INCREMENT, _NEUTRAL_INCREMENT)),
        ("neutral", 0.0, None, True, (0.0, _NOOP_TIE_PENALTY)),
        ("error", None, "boom", True, (0.0, 1.0)),
    ],
)
def test_bandit_deltas_priority_table(label, gain, error, tie, expected):
    """에러/regression > no-op tie > jump > 양수 gain > 그 외 — update_bandit과 리플레이가 공유하는 표."""
    assert bandit_deltas(label, gain, error, tie) == pytest.approx(expected)
