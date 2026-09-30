"""bin.api._replay_bandit_timeline — update_bandit과 같은 델타/decay로 posterior 시계열을 재현하는지 검증한다."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from bin.api import _replay_bandit_timeline
from cycle.action_optimizer import (
    _BANDIT_DECAY,
    _HALF_SUCCESS,
    _NEUTRAL_INCREMENT,
    _NOOP_TIE_PENALTY,
    update_bandit,
)


def _conn(rows: list[tuple]) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows
    return conn


def _ts(minute: int) -> datetime:
    return datetime(2026, 9, 30, 0, minute)


def test_replay_treats_an_exact_zero_gain_as_a_noop_tie() -> None:
    """#330: tie는 실패(beta+1)보다 약한 벌점이다. 리플레이가 이 분기를 몰라 neutral(0.1/0.1)로 세던 것을 고친다(#420)."""
    timeline = _replay_bandit_timeline(_conn([(_ts(1), "ensemble", "neutral", 0.0, None, 12.8)]), "c")

    assert timeline[0]["posterior_mean"] == pytest.approx(round(1.0 / (2.0 + _NOOP_TIE_PENALTY), 4))


@pytest.mark.parametrize(
    ("gain", "cv", "delta"),
    [
        (-1.611e-07, 12.812993, (0.0, _NOOP_TIE_PENALTY)),
        (6.4e-07, 12.812993, (0.0, _NOOP_TIE_PENALTY)),
        (-1.2e-05, 12.812993, (0.0, _NOOP_TIE_PENALTY)),
        (-1.4e-05, 12.812993, (_NEUTRAL_INCREMENT, _NEUTRAL_INCREMENT)),
        (9e-07, 0.95, (0.0, _NOOP_TIE_PENALTY)),
        (2e-06, 0.95, (_HALF_SUCCESS, _NEUTRAL_INCREMENT)),
    ],
)
def test_replay_treats_a_gain_within_float_noise_as_a_noop_tie(gain, cv, delta) -> None:
    """#449: 같은 계산이 프로세스마다 1e-7~2e-6 어긋나 비트 일치로는 tie를 못 잡는다. 허용폭은 metric 스케일에 비례한다(ADR-062)."""
    timeline = _replay_bandit_timeline(_conn([(_ts(1), "hyperparam_search", "neutral", gain, None, cv)]), "c")

    da, db = delta
    assert timeline[0]["posterior_mean"] == pytest.approx(round((1.0 + da) / (2.0 + da + db), 4))


def test_replay_applies_the_shared_decay_between_steps() -> None:
    rows = [
        (_ts(1), "ensemble", "neutral", 0.0, None, 0.9),
        (_ts(2), "ensemble", "jump", 0.03, None, 0.93),
    ]
    timeline = _replay_bandit_timeline(_conn(rows), "c")

    alpha = 1.0 + 1.0
    beta = 1.0 + _NOOP_TIE_PENALTY * _BANDIT_DECAY
    assert timeline[1]["posterior_mean"] == pytest.approx(round(alpha / (alpha + beta), 4))


@pytest.mark.parametrize(
    ("label", "gain", "error", "cv", "is_tie"),
    [
        ("jump", 0.03, None, 0.93, False),
        ("regression", -0.01, None, 0.89, False),
        ("neutral", 0.005, None, 0.905, False),
        ("neutral", -0.0004, None, 0.8996, False),
        ("neutral", 0.0, None, 0.9, True),
        ("neutral", -1.611e-07, None, 12.812993, True),
        ("neutral", 6.4e-07, None, 12.812993, True),
        ("error", None, "cpu budget exceeded", None, False),
        ("neutral", None, None, None, False),
    ],
)
def test_replay_first_step_matches_the_live_update(label, gain, error, cv, is_tie) -> None:
    live = MagicMock()
    update_bandit(live, "c", "ensemble", label, gain, error, is_noop_tie=is_tie)
    da, db = live.execute.call_args[0][1][3:5]

    timeline = _replay_bandit_timeline(_conn([(_ts(1), "ensemble", label, gain, error, cv)]), "c")

    assert timeline[0]["posterior_mean"] == pytest.approx(round((1.0 + da) / (2.0 + da + db), 4))


def test_replay_keeps_action_types_separate_and_skips_unknown_ones() -> None:
    rows = [
        (_ts(1), "ensemble", "jump", 0.03, None, 0.93),
        (_ts(2), "not_an_action", "jump", 0.03, None, 0.93),
        (_ts(3), "model_swap", "regression", -0.1, None, 0.8),
    ]
    timeline = _replay_bandit_timeline(_conn(rows), "c")

    assert [(r["step"], r["action_type"]) for r in timeline] == [(1, "ensemble"), (3, "model_swap")]
