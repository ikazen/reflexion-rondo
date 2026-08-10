"""cycle/promotion.py — confirm_and_measure 단위 테스트.

eval_isolated를 monkeypatch해 paired cross-seed 확인 로직과 holdout 경로를 검증한다.
"""
from __future__ import annotations

from unittest.mock import call, patch

import numpy as np
import polars as pl
import pytest

from cycle.promotion import ConfirmResult, _NOOP_PATCH, confirm_and_measure, effective_label
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


# --- holdout_regressed 게이트 (#98) ---
# candidate holdout이 baseline(현재 best 또는 콜드스타트면 BasePipeline) holdout보다
# 나쁘면 cross-seed confirm을 통과해도 confirmed=False로 강제한다 — cross-seed는
# seed만 바꾼 CV라 preprocess의 valid-target 의존 누수(#96/#97)처럼 seed 불변인
# 문제를 못 잡지만, holdout은 dummy target으로 실제 추론 조건을 재현해 잡을 수 있다.

def test_holdout_worse_than_baseline_blocks_confirmation():
    """candidate holdout이 baseline holdout보다 나쁘면 cross-seed 통과해도 confirmed=False."""
    def _se(*args, **kwargs):
        if kwargs.get("holdout_data") is not None:
            # baseline 쪽 eval은 best_source=None, prev_best=None으로 호출됨(_measure_holdout)
            if kwargs.get("best_source") is None:
                return _ok_with_holdout(holdout_score=0.90)  # baseline이 더 좋음
            return _ok_with_holdout(holdout_score=0.80)  # candidate가 더 나쁨
        if kwargs.get("best_source") is None:
            return _baseline()
        return _ok()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(**_COMMON, holdout10=_df(), confirm_seeds=[7])
    assert result.holdout_regressed is True
    assert result.confirmed is False


def test_holdout_better_than_baseline_keeps_confirmation():
    """candidate holdout이 baseline holdout보다 좋으면 cross-seed 결과 그대로 유지."""
    def _se(*args, **kwargs):
        if kwargs.get("holdout_data") is not None:
            if kwargs.get("best_source") is None:
                return _ok_with_holdout(holdout_score=0.80)  # baseline
            return _ok_with_holdout(holdout_score=0.90)  # candidate가 더 좋음
        if kwargs.get("best_source") is None:
            return _baseline()
        return _ok()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(**_COMMON, holdout10=_df(), confirm_seeds=[7])
    assert result.holdout_regressed is False
    assert result.confirmed is True


def test_holdout_regressed_respects_metric_sign():
    """rmse처럼 낮을수록 좋은 metric(metric_sign=-1)에서 candidate가 baseline보다
    수치가 더 크면(즉 더 나쁘면) regressed=True여야 한다."""
    def _se(*args, **kwargs):
        if kwargs.get("holdout_data") is not None:
            if kwargs.get("best_source") is None:
                return _ok_with_holdout(holdout_score=1.0)  # baseline rmse
            return _ok_with_holdout(holdout_score=5.0)  # candidate rmse가 더 나쁨(더 큼)
        if kwargs.get("best_source") is None:
            return _baseline()
        return _ok()

    common_rmse = {**_COMMON, "metric": "rmse"}
    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(**common_rmse, holdout10=_df(), confirm_seeds=[7])
    assert result.holdout_regressed is True
    assert result.confirmed is False


def test_baseline_holdout_eval_error_does_not_block():
    """baseline holdout eval이 실패하면(에러) 비교 근거가 없으므로 막지 않는다 —
    정보 없음과 악화 확인은 다르다."""
    def _se(*args, **kwargs):
        if kwargs.get("holdout_data") is not None:
            if kwargs.get("best_source") is None:
                return _err()  # baseline holdout eval 실패
            return _ok_with_holdout(holdout_score=0.5)
        if kwargs.get("best_source") is None:
            return _baseline()
        return _ok()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(**_COMMON, holdout10=_df(), confirm_seeds=[7])
    assert result.holdout_regressed is False
    assert result.confirmed is True


def test_holdout_baseline_uses_best_source_or_noop_patch():
    """baseline holdout eval의 source는 best_source(콜드스타트면 _NOOP_PATCH)여야 한다."""
    captured: list[dict] = []

    def _se(*args, **kwargs):
        captured.append(dict(kwargs))
        if kwargs.get("holdout_data") is not None:
            return _ok_with_holdout(holdout_score=0.5)
        if kwargs.get("best_source") is None:
            return _baseline()
        return _ok()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        confirm_and_measure(**_COMMON, holdout10=_df(), confirm_seeds=[7])

    holdout_calls = [c for c in captured if c.get("holdout_data") is not None]
    assert len(holdout_calls) == 2  # candidate + baseline
    baseline_holdout_call = next(c for c in holdout_calls if c.get("best_source") is None)
    assert baseline_holdout_call["source"] == _COMMON["best_source"]


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


# --- effective_label: bandit/lesson 보상 신호를 confirm 결과와 연동 (#164) ---
#
# update_bandit/reflect가 confirm 이전의 잠정 label(jump)을 그대로 쓰면, confirm이
# 나중에 cross-seed 미재현이나 holdout 악화로 거부해도 그 보상이 되돌아가지 않아
# 같은 아이디어가 계속 높은 확률로 재선택되는 자기강화 루프가 생긴다(실측: s6e1의
# preprocessing 후보가 cv_score 소수점 10자리까지 동일하게 32회 재생성, 매번 holdout
# 거부). effective_label은 이 다운그레이드를 결정하는 순수 함수다.

def _confirmed() -> ConfirmResult:
    return ConfirmResult(confirmed=True, holdout_score=0.9, seed_gains={"7": {}})


def _rejected_cross_seed() -> ConfirmResult:
    return ConfirmResult(confirmed=False, holdout_score=None, seed_gains={"7": {"error": "..."}})


def _rejected_holdout() -> ConfirmResult:
    return ConfirmResult(
        confirmed=False, holdout_score=0.8, seed_gains={"7": {}}, holdout_regressed=True,
    )


def test_effective_label_jump_confirmed_stays_jump():
    assert effective_label("jump", _confirmed()) == "jump"


def test_effective_label_jump_rejected_cross_seed_downgraded():
    assert effective_label("jump", _rejected_cross_seed()) == "regression"


def test_effective_label_jump_rejected_holdout_downgraded():
    assert effective_label("jump", _rejected_holdout()) == "regression"


def test_effective_label_jump_confirm_none_stays_jump():
    """confirm 자체가 스킵된 경우(예: train 로드 실패) — 판단 근거가 없으니
    원본 그대로 둔다. 보수적으로 다운그레이드하지 않음."""
    assert effective_label("jump", None) == "jump"


@pytest.mark.parametrize("label", ["neutral", "regression", "error"])
def test_effective_label_non_jump_unaffected(label: str):
    """jump가 아닌 label은 애초에 confirm을 안 타므로 confirm 결과와 무관하게
    원본 그대로여야 한다."""
    assert effective_label(label, _rejected_holdout()) == label
    assert effective_label(label, _confirmed()) == label
