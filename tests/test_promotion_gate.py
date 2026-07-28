"""cycle/promotion.py — confirm_and_measure 단위 테스트.

eval_isolated를 monkeypatch해 paired cross-seed 확인 로직과 holdout 경로를 검증한다.
"""
from __future__ import annotations

from unittest.mock import call, patch

import numpy as np
import polars as pl
import pytest

from cycle.promotion import ConfirmResult, _NOOP_PATCH, confirm_and_measure
from runtime.isolate import IsolatedResult


def _baseline(cv: float = 0.84) -> IsolatedResult:
    return IsolatedResult(
        cv_score=cv, cv_fold_var=0.0, fold_scores=[cv],
        label="neutral", gain_vs_best=None, error_trace=None,
    )


def _ok(cv: float = 0.85, gain: float = 0.01) -> IsolatedResult:
    return IsolatedResult(
        cv_score=cv, cv_fold_var=0.0, fold_scores=[cv],
        label="neutral", gain_vs_best=gain, error_trace=None,
    )


def _fail(cv: float = 0.839, gain: float = -0.001) -> IsolatedResult:
    return IsolatedResult(
        cv_score=cv, cv_fold_var=0.0, fold_scores=[cv],
        label="neutral", gain_vs_best=gain, error_trace=None,
    )


def _err() -> IsolatedResult:
    return IsolatedResult(
        cv_score=None, cv_fold_var=None, fold_scores=None,
        label=None, gain_vs_best=None, error_trace="RuntimeError",
    )


def _ok_with_holdout(cv: float = 0.85, gain: float = 0.01, holdout_score: float = 0.83) -> IsolatedResult:
    return IsolatedResult(
        cv_score=cv, cv_fold_var=0.0, fold_scores=[cv],
        label="neutral", gain_vs_best=gain, error_trace=None,
        holdout_score=holdout_score,
    )


def _df(n: int = 50) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame({"x": rng.standard_normal(n), "y": rng.integers(0, 2, n).astype(float)})


_COMMON = dict(
    source="class Patch: pass",
    best_source="class Patch: pass",
    train90=_df(),
    target_col="y",
    metric="auc",
    n_splits=3,
    seed=42,
    is_classification=True,
    action_type="feature_engineering",
)


def _paired_side_effect(baseline_result=None, candidate_result=None):
    """baseline(best_source=None) / candidate(best_source 있음) 자동 분기."""
    bl = baseline_result or _baseline()
    ca = candidate_result or _ok()

    def _se(*args, **kwargs):
        return bl if kwargs.get("best_source") is None else ca

    return _se


def test_all_seeds_confirmed():
    """모든 confirm_seeds에서 paired gain > 0 → confirmed=True."""
    with patch("cycle.promotion.eval_isolated", side_effect=_paired_side_effect()):
        result = confirm_and_measure(**_COMMON, holdout10=None, confirm_seeds=[7, 101])
    assert result.confirmed is True
    assert result.holdout_score is None


def test_first_seed_fails_not_confirmed():
    """첫 seed에서 candidate gain <= 0 → confirmed=False."""
    with patch("cycle.promotion.eval_isolated", side_effect=_paired_side_effect(candidate_result=_fail())):
        result = confirm_and_measure(**_COMMON, holdout10=None, confirm_seeds=[7, 101])
    assert result.confirmed is False


def test_error_trace_not_confirmed():
    """candidate error_trace 있으면 confirmed=False, seed_gains에 error가 남는다(#73).

    이전엔 seed_gains에 baseline_cv/candidate_cv/gain만 기록해 "gain<=0으로 재현
    안 됨"과 "크래시로 평가 자체가 실패함"을 DB(raw.attempts.confirm_seed_gains)만
    보고는 구분할 수 없었다(s6e6 64475b93 실사고).
    """
    with patch("cycle.promotion.eval_isolated", side_effect=_paired_side_effect(candidate_result=_err())):
        result = confirm_and_measure(**_COMMON, holdout10=None, confirm_seeds=[7])
    assert result.confirmed is False
    assert result.seed_gains["7"]["error"] == "RuntimeError"
    assert result.seed_gains["7"]["candidate_cv"] is None


def test_second_seed_fails_not_confirmed():
    """첫 seed는 통과, 두 번째 seed candidate 실패 → confirmed=False, 총 4회 호출."""
    call_count = 0
    seed_calls: list[int] = []

    def _se(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        seed_calls.append(kwargs.get("seed"))
        if kwargs.get("best_source") is None:
            return _baseline()
        # 첫 번째 seed candidate=ok, 두 번째 seed candidate=fail
        paired_idx = len([s for s in seed_calls if s == kwargs.get("seed")]) - 1
        seed_so_far = [s for s in seed_calls if s != kwargs.get("seed")]
        # 두 번째 시드에서 candidate가 실패하도록: 시드별 첫 candidate(best_source!=None)를 추적
        candidate_count = sum(1 for i, s in enumerate(seed_calls) if
                               i < call_count - 1 and kwargs.get("best_source") is not None)
        return _ok() if call_count <= 3 else _fail()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(**_COMMON, holdout10=None, confirm_seeds=[7, 101])
    assert result.confirmed is False
    assert call_count == 4  # seed7: baseline+candidate, seed101: baseline+candidate(fail)


def test_empty_confirm_seeds_always_confirmed():
    """confirm_seeds=[] → eval_isolated 미호출, confirmed=True."""
    with patch("cycle.promotion.eval_isolated") as mock_eval:
        result = confirm_and_measure(**_COMMON, holdout10=None, confirm_seeds=[])
    mock_eval.assert_not_called()
    assert result.confirmed is True


def test_holdout10_triggers_holdout_eval():
    """holdout10 전달 시 holdout_score가 반환된다."""
    responses_by_role: dict = {"baseline": _baseline(), "candidate": _ok()}
    holdout_called = False

    def _se(*args, **kwargs):
        nonlocal holdout_called
        if kwargs.get("holdout_data") is not None:
            holdout_called = True
            return _ok_with_holdout(holdout_score=0.82)
        if kwargs.get("best_source") is None:
            return _baseline()
        return _ok()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(**_COMMON, holdout10=_df(), confirm_seeds=[7, 101])
    assert result.holdout_score == pytest.approx(0.82)
    assert result.confirmed is True
    assert holdout_called


def test_holdout_none_no_holdout_score():
    """holdout10=None → holdout_score=None. 1 seed → eval_isolated 2회(baseline+candidate)."""
    with patch("cycle.promotion.eval_isolated", side_effect=_paired_side_effect()) as mock_eval:
        result = confirm_and_measure(**_COMMON, holdout10=None, confirm_seeds=[7])
    assert result.holdout_score is None
    assert mock_eval.call_count == 2


def test_seed_gains_recorded():
    """seed_gains에 seed별 baseline_cv/candidate_cv/gain이 담긴다."""
    with patch("cycle.promotion.eval_isolated", side_effect=_paired_side_effect(
        baseline_result=_baseline(cv=0.84),
        candidate_result=_ok(cv=0.851, gain=0.011),
    )):
        result = confirm_and_measure(**_COMMON, holdout10=None, confirm_seeds=[7, 101])
    assert result.confirmed is True
    assert result.seed_gains is not None
    for key in ("7", "101"):
        assert key in result.seed_gains
        entry = result.seed_gains[key]
        assert entry["baseline_cv"] == pytest.approx(0.84)
        assert entry["candidate_cv"] == pytest.approx(0.851)
        assert entry["gain"] == pytest.approx(0.011)
        assert entry["error"] is None


def test_baseline_eval_error_not_confirmed():
    """baseline eval 에러 → confirmed=False, 그 seed도 seed_gains에 기록된다(#73).

    이전엔 baseline 실패 시 seed_gains에 아무것도 안 남기고 바로 return했다 —
    confirm_seeds가 [7]뿐이면 seed_gains 전체가 빈 dict가 됐고, confirm_and_measure가
    빈 dict를 None으로 접어버려(구 코드) "confirm이 아예 안 돎"과 "confirm이 돌았는데
    baseline에서 크래시함"이 DB상 똑같이 보였다.
    """
    def _se(*args, **kwargs):
        if kwargs.get("best_source") is None:
            return _err()
        return _ok()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(**_COMMON, holdout10=None, confirm_seeds=[7])
    assert result.confirmed is False
    assert result.seed_gains is not None
    assert result.seed_gains["7"]["baseline_cv"] is None
    assert "RuntimeError" in result.seed_gains["7"]["error"]


def test_best_source_none_uses_noop_patch():
    """best_source=None일 때 baseline eval은 _NOOP_PATCH source로 호출된다.

    baseline: prev_best=None (구분 기준)
    candidate: prev_best=base_cv (non-None)
    """
    captured: list[dict] = []

    def _se(*args, **kwargs):
        captured.append(dict(kwargs))
        # baseline 호출: prev_best=None
        if kwargs.get("prev_best") is None:
            return _baseline()
        # candidate 호출: prev_best=base_cv
        return _ok()

    common_no_best = {**_COMMON, "best_source": None}
    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(**common_no_best, holdout10=None, confirm_seeds=[7])

    # baseline 호출이 _NOOP_PATCH를 source로 썼는지 확인
    baseline_calls = [c for c in captured if c.get("prev_best") is None]
    assert len(baseline_calls) >= 1
    assert baseline_calls[0]["source"] == _NOOP_PATCH
    assert result.confirmed is True
