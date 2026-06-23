"""cycle/promotion.py — confirm_and_measure 단위 테스트.

eval_isolated를 monkeypatch해 cross-seed 확인 로직과 holdout 경로를 검증한다.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from cycle.promotion import ConfirmResult, confirm_and_measure
from runtime.isolate import IsolatedResult


def _ok(gain: float = 0.01) -> IsolatedResult:
    return IsolatedResult(
        cv_score=0.85, cv_fold_var=0.0, fold_scores=[0.85],
        label="neutral", gain_vs_best=gain, error_trace=None,
    )


def _fail(gain: float = -0.001) -> IsolatedResult:
    return IsolatedResult(
        cv_score=0.84, cv_fold_var=0.0, fold_scores=[0.84],
        label="neutral", gain_vs_best=gain, error_trace=None,
    )


def _err() -> IsolatedResult:
    return IsolatedResult(
        cv_score=None, cv_fold_var=None, fold_scores=None,
        label=None, gain_vs_best=None, error_trace="RuntimeError",
    )


def _ok_with_holdout(gain: float = 0.01, holdout_score: float = 0.83) -> IsolatedResult:
    return IsolatedResult(
        cv_score=0.85, cv_fold_var=0.0, fold_scores=[0.85],
        label="neutral", gain_vs_best=gain, error_trace=None,
        holdout_score=holdout_score,
    )


def _df(n: int = 50) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame({"x": rng.standard_normal(n), "y": rng.integers(0, 2, n).astype(float)})


_COMMON = dict(
    source="class Patch: pass",
    best_source=None,
    train90=_df(),
    target_col="y",
    metric="auc",
    n_splits=3,
    seed=42,
    is_classification=True,
    prev_best=0.84,
    action_type="feature_engineering",
)


def test_all_seeds_confirmed():
    """모든 confirm_seeds에서 gain > 0 → confirmed=True."""
    with patch("cycle.promotion.eval_isolated", return_value=_ok()):
        result = confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[7, 101],
        )
    assert result.confirmed is True
    assert result.holdout_score is None


def test_first_seed_fails_not_confirmed():
    """첫 seed에서 gain <= 0 → confirmed=False."""
    with patch("cycle.promotion.eval_isolated", return_value=_fail()):
        result = confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[7, 101],
        )
    assert result.confirmed is False


def test_error_trace_not_confirmed():
    """error_trace 있으면 confirmed=False."""
    with patch("cycle.promotion.eval_isolated", return_value=_err()):
        result = confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[7],
        )
    assert result.confirmed is False


def test_second_seed_fails_not_confirmed():
    """첫 seed는 통과, 두 번째 seed 실패 → confirmed=False."""
    call_count = 0

    def _side_effect(*a, **kw):
        nonlocal call_count
        call_count += 1
        return _ok() if call_count == 1 else _fail()

    with patch("cycle.promotion.eval_isolated", side_effect=_side_effect):
        result = confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[7, 101],
        )
    assert result.confirmed is False
    assert call_count == 2


def test_empty_confirm_seeds_always_confirmed():
    """confirm_seeds=[] → eval_isolated 미호출, confirmed=True."""
    with patch("cycle.promotion.eval_isolated") as mock_eval:
        result = confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[],
        )
    mock_eval.assert_not_called()
    assert result.confirmed is True


def test_holdout10_triggers_holdout_eval():
    """holdout10 전달 시 holdout_score가 반환된다."""
    responses = [_ok(), _ok(), _ok_with_holdout(holdout_score=0.82)]

    def _side_effect(*a, **kw):
        return responses.pop(0)

    with patch("cycle.promotion.eval_isolated", side_effect=_side_effect):
        result = confirm_and_measure(
            **_COMMON, holdout10=_df(), confirm_seeds=[7, 101],
        )
    assert result.holdout_score == pytest.approx(0.82)
    assert result.confirmed is True


def test_holdout_none_no_holdout_score():
    """holdout10=None → holdout_score=None, eval_isolated는 confirm_seeds만큼만 호출."""
    with patch("cycle.promotion.eval_isolated", return_value=_ok()) as mock_eval:
        result = confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[7],
        )
    assert result.holdout_score is None
    assert mock_eval.call_count == 1
