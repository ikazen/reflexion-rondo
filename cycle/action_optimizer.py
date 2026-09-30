"""Action-type Beta-Bernoulli bandit — persist step에서 결정적 갱신, LLM 없음.

scope='local'(competition_id별) Local 티어만 구현.
Global/Cluster 티어는 대회 누적 후 승격 예정.

역할 분리:
  assign_super_cycle_actions — super_cycle retrieve에서 attempt별 action_type 강제 배정.
  get_action_prior           — reflexion 사이클에서 LLM Strategist에 텍스트 prior로만 제공.
                               최종 action 결정은 LLM (ADR-005/014). regret 보장 없음(advisory).

업데이트 규칙:
  jump                        → α += 1.0  (is_significant_gain paired 게이트 통과,
                                            promotion과 동일 기준의 유의미 이득)
  gain_vs_best > 0 (non-jump) → α += 0.5  (half-success, 유의미 미달 양수 이득)
  regression 또는 error_trace → β += 1.0
  is_noop_tie                 → β += 0.3  (cv_score가 prev_best와 완전 동일 —
                                            patch가 유효 계산을 안 바꿨다는 확정
                                            신호. 실패보다 약하지만 neutral과는
                                            구분 — #330, neutral로 묻히면 밴딧이
                                            무효과 액션을 계속 선호하게 된다)
  neutral                     → α, β 각 0.1 (약한 신호 유지)

ON CONFLICT 시 decay(_BANDIT_DECAY=0.95)로 기존 α/β를 prior(1.0) 쪽으로 수축 후 가산.
초기 운빨 action의 조기 수렴을 방지한다.
"""
from __future__ import annotations

import numpy as np

from config.settings import ACTION_TYPES
from store.db import PgConn

_NEUTRAL_INCREMENT = 0.1
_HALF_SUCCESS = 0.5
_NOOP_TIE_PENALTY = 0.3
_BANDIT_DECAY = 0.95
_SCOPE_LOCAL = "local"


def bandit_deltas(
    label: str | None,
    gain_vs_best: float | None,
    error_trace: str | None,
    is_noop_tie: bool = False,
) -> tuple[float, float]:
    """(alpha, beta) 증가분. update_bandit과 bin/api.py의 posterior 리플레이가 공유한다 — 한쪽만 바꾸면 리플레이가 라이브와 발산한다."""
    if error_trace is not None or label == "regression":
        return 0.0, 1.0
    if is_noop_tie:
        return 0.0, _NOOP_TIE_PENALTY
    if label == "jump":
        return 1.0, 0.0
    if gain_vs_best is not None and gain_vs_best > 0:
        return _HALF_SUCCESS, _NEUTRAL_INCREMENT
    return _NEUTRAL_INCREMENT, _NEUTRAL_INCREMENT


def update_bandit(
    conn: PgConn,
    competition_id: str,
    action_type: str,
    label: str,
    gain_vs_best: float | None,
    error_trace: str | None,
    is_noop_tie: bool = False,
) -> None:
    if action_type not in ACTION_TYPES:
        return

    da, db = bandit_deltas(label, gain_vs_best, error_trace, is_noop_tie)

    conn.execute(
        """
        INSERT INTO raw.action_bandit (scope, scope_key, action_type, alpha, beta, updated_at)
        VALUES (%s, %s, %s, 1.0 + %s, 1.0 + %s, now())
        ON CONFLICT (scope, scope_key, action_type)
        DO UPDATE SET
            alpha      = 1.0 + (raw.action_bandit.alpha - 1.0) * %s + %s,
            beta       = 1.0 + (raw.action_bandit.beta  - 1.0) * %s + %s,
            updated_at = now()
        """,
        [_SCOPE_LOCAL, competition_id, action_type, da, db, _BANDIT_DECAY, da, _BANDIT_DECAY, db],
    )


# 사후 평균이 낮고 관측이 충분히 쌓인 액션은 배정에서 뺀다(#375) — Thompson 표본이 상위
# n_attempts를 뽑는 구조라 posterior mean이 0.08 안팎이어도 매 사이클 15%가량 배정돼 CPU를
# 태운다(s6e8 hyperparam_search 46h 29/29 tie 실측). observed는 decay(_BANDIT_DECAY)가 걸린
# 값이라 절대 관측 횟수의 근사일 뿐이고 상한이 있다(초당 최대 증가폭 1.0을 decay 0.95로 나눈
# 20 근방) — "판정을 신뢰할 만큼은 쌓였는지"용 문턱으로만 쓴다.
_DEAD_ACTION_MEAN_THRESHOLD = 0.1
_DEAD_ACTION_MIN_OBSERVED = 5.0
# 완전히 배제하면 회복 기회가 없다 — 대회 전체 attempt 수(이 action 자신의 배제로 멈추지
# 않는, 항상 진행하는 시계) 기준 매 100건마다 n_attempts틱만큼 죽은 액션도 정상 후보로
# 되돌려 재탐색시킨다.
_DEAD_ACTION_REEXPLORE_EVERY = 100


def _is_dead_action(alpha: float, beta: float) -> bool:
    observed = (alpha - 1.0) + (beta - 1.0)
    return observed >= _DEAD_ACTION_MIN_OBSERVED and alpha / (alpha + beta) < _DEAD_ACTION_MEAN_THRESHOLD


def _reexplore_window_open(total_attempts: int, n_attempts: int) -> bool:
    return total_attempts % _DEAD_ACTION_REEXPLORE_EVERY < n_attempts


def assign_super_cycle_actions(
    conn: PgConn,
    competition_id: str,
    n_attempts: int = 3,
    seed: int | None = None,
) -> list[str]:
    """bandit Thompson sample 1회로 전체 action을 순위 매기고 top-n을 배정한다.

    seed=None(기본)이면 매 사이클 새 엔트로피 → 탐색. 테스트에서만 고정.
    """
    rows = conn.execute(
        """
        SELECT action_type, alpha, beta
        FROM raw.action_bandit
        WHERE scope = %s AND scope_key = %s
        """,
        [_SCOPE_LOCAL, competition_id],
    ).fetchall()

    bandit: dict[str, tuple[float, float]] = {r[0]: (r[1], r[2]) for r in rows}
    rng = np.random.default_rng(seed)
    scores = {
        action: float(rng.beta(*bandit.get(action, (1.0, 1.0))))
        for action in ACTION_TYPES
    }
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)

    dead = {a for a in ACTION_TYPES if _is_dead_action(*bandit.get(a, (1.0, 1.0)))}
    if dead:
        total = conn.execute(
            "SELECT count(*) FROM raw.attempts WHERE competition_id = %s",
            [competition_id],
        ).fetchone()[0]
        if _reexplore_window_open(total, n_attempts):
            dead = set()

    eligible = [a for a in ranked if a not in dead]
    picked = eligible[:n_attempts]
    if len(picked) < n_attempts:
        # dead가 3개 이상이면 후보가 n_attempts에 모자란다 — dead로 채우면 배제한 의미가 사라지므로 살아 있는 액션을
        # 반복 배정하고, 하나도 없을 때만 원래 랭킹에서 채운다(#404, ADR-060).
        pool = eligible or ranked
        picked = [pool[i % len(pool)] for i in range(n_attempts)]
    return picked


def get_action_prior(
    conn: PgConn,
    competition_id: str,
    seed: int | None = None,
) -> dict[str, float]:
    """action별 Beta posterior에서 Thompson 표본을 1회씩 뽑아 반환한다 (advise용, 높을수록 추천).

    posterior mean이 아니라 표본이라 호출마다 값이 달라진다. DB에 데이터 없으면 균일 Beta(1,1) → 모든 action 동등.
    반환값은 LLM Strategist 프롬프트에 텍스트로 주입되며, 최종 결정은 LLM(advisory).
    """
    rows = conn.execute(
        """
        SELECT action_type, alpha, beta
        FROM raw.action_bandit
        WHERE scope = %s AND scope_key = %s
        """,
        [_SCOPE_LOCAL, competition_id],
    ).fetchall()

    rng = np.random.default_rng(seed)
    bandit: dict[str, tuple[float, float]] = {r[0]: (r[1], r[2]) for r in rows}

    result: dict[str, float] = {}
    for action in ACTION_TYPES:
        a, b = bandit.get(action, (1.0, 1.0))
        sample = float(rng.beta(a, b))
        result[action] = round(sample, 4)

    return result
