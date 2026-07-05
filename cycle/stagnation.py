"""정체 감지 — 결정적 코드, LLM 없음.

detect_stagnation()은 최근 window 개 super-cycle을 보고 탐색 다양성과 정체 길이를 계산한다.
결과는 Strategist 프롬프트에 주입되어 underused action_type 선택을 유도한다.

action_type 사용 여부(underused_actions)는 승자·패자 구분 없이 전체 attempt에서 집계한다 —
assign_super_cycle_actions가 매 사이클 여러 action_type을 강제 배정하므로 패자로 시도된
action도 이미 최근에 탐색된 것. jump 감지·stagnant_for는 승자 이력만 사용한다(label='jump'는
승자=argmax gain_vs_best에서만 실질적으로 발생).

BON-267: label='jump'는 cycle/run.py에서 promotion과 동일한 is_significant_gain(paired
per-fold t-test) 기준으로 확정된다 — harness의 절대-마진 기준이 아니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import ACTION_TYPES
from store.db import PgConn

_STAGNANT_THRESHOLD = 3  # 이 횟수 이상 jump 없으면 is_stagnant=True


@dataclass(frozen=True, slots=True)
class StagnationSignal:
    is_stagnant: bool
    jumps_in_window: int
    underused_actions: tuple[str, ...]
    stagnant_for: int


def detect_stagnation(
    conn: PgConn,
    competition_id: str,
    window: int = 5,
) -> StagnationSignal:
    # window*3: 슈퍼사이클당 최대 3 attempt(승자+패자2) 감안한 여유 확보.
    rows = conn.execute(
        """
        select was_promoted, label, action_type
        from raw.attempts
        where competition_id = %s
        order by run_ts desc
        limit %s
        """,
        [competition_id, window * 3],
    ).fetchall()

    if not rows:
        return StagnationSignal(
            is_stagnant=False,
            jumps_in_window=0,
            underused_actions=(),
            stagnant_for=0,
        )

    used_actions = {action for _, _, action in rows}
    underused_actions = tuple(a for a in ACTION_TYPES if a not in used_actions)

    winner_labels = [label for was_promoted, label, _ in rows if was_promoted][:window]

    jumps_in_window = sum(1 for label in winner_labels if label == "jump")

    # stagnant_for: 가장 최근 jump 이후 연속 비-jump 승자 횟수
    stagnant_for = 0
    for label in winner_labels:
        if label == "jump":
            break
        stagnant_for += 1

    is_stagnant = stagnant_for >= _STAGNANT_THRESHOLD

    return StagnationSignal(
        is_stagnant=is_stagnant,
        jumps_in_window=jumps_in_window,
        underused_actions=underused_actions,
        stagnant_for=stagnant_for,
    )
