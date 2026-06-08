"""Action-type Beta-Bernoulli bandit — persist step에서 결정적 갱신, LLM 없음 (BON-109).

scope='local'(competition_id별) Local 티어만 구현.
Global/Cluster 티어는 대회 누적 후 승격 예정.

업데이트 규칙:
  jump 또는 gain_vs_best > 0  → α += 1.0
  regression 또는 error_trace → β += 1.0
  neutral                     → α, β 각 0.1 (약한 신호 유지)
"""
from __future__ import annotations

import numpy as np

from config.settings import ACTION_TYPES
from store.db import PgConn

_NEUTRAL_INCREMENT = 0.1
_SCOPE_LOCAL = "local"


def update_bandit(
    conn: PgConn,
    competition_id: str,
    action_type: str,
    label: str,
    gain_vs_best: float | None,
    error_trace: str | None,
) -> None:
    if action_type not in ACTION_TYPES:
        return

    if error_trace is not None or label == "regression":
        da, db = 0.0, 1.0
    elif label == "jump" or (gain_vs_best is not None and gain_vs_best > 0):
        da, db = 1.0, 0.0
    else:
        da, db = _NEUTRAL_INCREMENT, _NEUTRAL_INCREMENT

    conn.execute(
        """
        INSERT INTO raw.action_bandit (scope, scope_key, action_type, alpha, beta, updated_at)
        VALUES (%s, %s, %s, 1.0 + %s, 1.0 + %s, now())
        ON CONFLICT (scope, scope_key, action_type)
        DO UPDATE SET
            alpha      = raw.action_bandit.alpha + %s,
            beta       = raw.action_bandit.beta  + %s,
            updated_at = now()
        """,
        [_SCOPE_LOCAL, competition_id, action_type, da, db, da, db],
    )


def assign_super_cycle_actions(
    conn: PgConn,
    competition_id: str,
    n_attempts: int = 3,
) -> list[str]:
    """bandit Thompson sample 1회로 전체 action을 순위 매기고 top-n을 배정한다.

    같은 사이클 안에서 일관된 선호 순서를 유지하면서 다양성을 보장한다.
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
    rng = np.random.default_rng(0)
    scores = {
        action: float(rng.beta(*bandit.get(action, (1.0, 1.0))))
        for action in ACTION_TYPES
    }
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return ranked[:n_attempts]


def get_action_prior(
    conn: PgConn,
    competition_id: str,
    n_samples: int = 1000,
    seed: int | None = None,
) -> dict[str, float]:
    """Thompson 샘플 1회씩 → posterior_mean 반환 (advise용, 높을수록 추천).

    DB에 데이터 없으면 균일 Beta(1,1) → 모든 action 동등.
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
