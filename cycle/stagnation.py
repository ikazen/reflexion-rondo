"""정체 감지 — 결정적 코드, LLM 없음.

detect_stagnation()은 최근 window 개 attempt를 보고 탐색 다양성과 정체 길이를 계산한다.
결과는 Strategist 프롬프트에 주입되어 underused action_type 선택을 유도한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from config.settings import ACTION_TYPES

_STAGNANT_THRESHOLD = 3  # 이 횟수 이상 jump 없으면 is_stagnant=True


@dataclass(frozen=True, slots=True)
class StagnationSignal:
    is_stagnant: bool
    jumps_in_window: int
    underused_actions: tuple[str, ...]
    stagnant_for: int


def detect_stagnation(
    conn: duckdb.DuckDBPyConnection,
    competition_id: str,
    window: int = 5,
) -> StagnationSignal:
    rows = conn.execute(
        """
        select label, action_type
        from raw.attempts
        where competition_id = ?
          and cv_score is not null
        order by run_ts desc
        limit ?
        """,
        [competition_id, window],
    ).fetchall()

    if not rows:
        return StagnationSignal(
            is_stagnant=False,
            jumps_in_window=0,
            underused_actions=(),
            stagnant_for=0,
        )

    jumps_in_window = sum(1 for label, _ in rows if label == "jump")
    used_actions = {action for _, action in rows}
    underused_actions = tuple(a for a in ACTION_TYPES if a not in used_actions)

    # stagnant_for: 가장 최근 jump 이후 연속 비-jump 횟수
    stagnant_for = 0
    for label, _ in rows:
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
