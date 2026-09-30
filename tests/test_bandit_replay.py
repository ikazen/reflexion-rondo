"""bin.api._replay_bandit_timeline — update_bandit과 같은 델타/decay로 posterior 시계열을 재현하는지 검증한다."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from bin.api import _replay_bandit_timeline
from cycle.action_optimizer import _BANDIT_DECAY, _NOOP_TIE_PENALTY, update_bandit


def _conn(rows: list[tuple]) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows
    return conn


def _ts(minute: int) -> datetime:
    return datetime(2026, 9, 30, 0, minute)


def test_replay_treats_an_exact_zero_gain_as_a_noop_tie() -> None:
    """#330: tie는 실패(beta+1)보다 약한 벌점이다. 리플레이가 이 분기를 몰라 neutral(0.1/0.1)로 세던 것을 고친다(#420)."""
    timeline = _replay_bandit_timeline(_conn([(_ts(1), "ensemble", "neutral", 0.0, None)]), "c")

    assert timeline[0]["posterior_mean"] == pytest.approx(round(1.0 / (2.0 + _NOOP_TIE_PENALTY), 4))


def test_replay_applies_the_shared_decay_between_steps() -> None:
    rows = [
        (_ts(1), "ensemble", "neutral", 0.0, None),
        (_ts(2), "ensemble", "jump", 0.03, None),
    ]
    timeline = _replay_bandit_timeline(_conn(rows), "c")

    alpha = 1.0 + 1.0
    beta = 1.0 + _NOOP_TIE_PENALTY * _BANDIT_DECAY
    assert timeline[1]["posterior_mean"] == pytest.approx(round(alpha / (alpha + beta), 4))


@pytest.mark.parametrize(
    ("label", "gain", "error"),
    [
        ("jump", 0.03, None),
        ("regression", -0.01, None),
        ("neutral", 0.005, None),
        ("neutral", -0.0004, None),
        ("neutral", 0.0, None),
        ("error", None, "cpu budget exceeded"),
        ("neutral", None, None),
    ],
)
def test_replay_first_step_matches_the_live_update(label, gain, error) -> None:
    is_tie = gain is not None and gain == 0.0
    live = MagicMock()
    update_bandit(live, "c", "ensemble", label, gain, error, is_noop_tie=is_tie)
    da, db = live.execute.call_args[0][1][3:5]

    timeline = _replay_bandit_timeline(_conn([(_ts(1), "ensemble", label, gain, error)]), "c")

    assert timeline[0]["posterior_mean"] == pytest.approx(round((1.0 + da) / (2.0 + da + db), 4))


def test_replay_keeps_action_types_separate_and_skips_unknown_ones() -> None:
    rows = [
        (_ts(1), "ensemble", "jump", 0.03, None),
        (_ts(2), "not_an_action", "jump", 0.03, None),
        (_ts(3), "model_swap", "regression", -0.1, None),
    ]
    timeline = _replay_bandit_timeline(_conn(rows), "c")

    assert [(r["step"], r["action_type"]) for r in timeline] == [(1, "ensemble"), (3, "model_swap")]
