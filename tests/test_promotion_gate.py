"""cycle/promotion.py — confirm_and_measure 단위 테스트.

eval_isolated를 monkeypatch해 paired cross-seed 확인 로직과 holdout 경로를 검증한다.
"""
from __future__ import annotations

from unittest.mock import call, patch

import numpy as np
import polars as pl
import pytest

from cycle.promotion import (
    ConfirmResult,
    _NOOP_PATCH,
    _eval_context_key,
    _rounded_signature,
    confirm_and_measure,
    effective_label,
)
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


def test_cpu_budget_sec_propagates_to_all_eval_paths():
    """comp.CPU_BUDGET_SECS 오버라이드가 confirm baseline/candidate/holdout eval
    3경로 전부에 전달돼야 한다 — 안 그러면 메인 eval에서 통과한 대회별 상향
    예산(#176)이 confirm에서 기본값 900s로 되돌아가 영원히 재현 실패한다(#195)."""
    calls: list[float | None] = []

    def _se(*args, **kwargs):
        calls.append(kwargs.get("cpu_budget_sec"))
        if kwargs.get("holdout_data") is not None:
            return _ok_with_holdout(holdout_score=0.82)
        return _baseline() if kwargs.get("best_source") is None else _ok()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        confirm_and_measure(
            **_COMMON, holdout10=_df(), confirm_seeds=[7], cpu_budget_sec=3600.0,
        )
    assert calls, "eval_isolated가 호출되지 않았다"
    assert all(c == 3600.0 for c in calls), calls


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


# holdout_regressed 게이트 (#98)
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


# effective_label: bandit/lesson 보상 신호를 confirm 결과와 연동 (#164)
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


# confirm 게이트 캐시 (#166/#167/#168)
#
# cache 인자를 생략(None)한 위 테스트 전부가 회귀 가드다 — confirm_and_measure의
# 캐시 관련 분기는 cache is not None일 때만 진입하므로, 기존 테스트가 수정 없이
# 그대로 통과하면 cache=None 기본 경로는 변경 전과 동일하다는 뜻이다.

class _FakeCache:
    """PromotionCache와 동일한 4개 메서드만 구현하는 인메모리 테스트 더블.
    실제 SQL(ON CONFLICT 등)은 PromotionCache 자체가 아니라 배포 후 실제
    Postgres에 대해 검증한다 — 여기선 confirm_and_measure가 캐시를 언제/어떻게
    호출하는지(제어 흐름)만 검증한다."""

    def __init__(self) -> None:
        self.memos: dict = {}
        self.baselines: dict = {}
        self.put_memo_calls: list[ConfirmResult] = []
        self.put_baseline_calls: list[tuple] = []

    def _memo_key(self, key, candidate_cv, confirm_seeds):
        return (key, tuple(confirm_seeds), round(candidate_cv, 9))

    def get_memo(self, key, candidate_cv, candidate_fold_scores, confirm_seeds):
        return self.memos.get(self._memo_key(key, candidate_cv, confirm_seeds))

    def put_memo(self, key, candidate_cv, candidate_fold_scores, confirm_seeds, competition_id, result):
        self.put_memo_calls.append(result)
        self.memos[self._memo_key(key, candidate_cv, confirm_seeds)] = result

    def get_baseline(self, key, mode, seed):
        return self.baselines.get((key, mode, seed))

    def put_baseline(self, key, mode, seed, competition_id, score):
        self.put_baseline_calls.append((mode, seed, score))
        self.baselines[(key, mode, seed)] = score


def _ctx_key(**overrides) -> tuple:
    kwargs = dict(
        competition_id="comp1",
        best_source=_COMMON["best_source"],
        train90=_COMMON["train90"],
        target_col=_COMMON["target_col"],
        metric=_COMMON["metric"],
        n_splits=_COMMON["n_splits"],
        is_classification=_COMMON["is_classification"],
    )
    kwargs.update(overrides)
    return _eval_context_key(**kwargs)


# 행동 지문 회귀 가드: 2026-08 실측 s6e1 confirm 39회 리플레이
# 운영 DB(playground-series-s6e1, confirm_seed_gains not null) 실측값 — 소스는
# 39개 전부 distinct(AST 정규화 후에도)인데 이 지문 기준으로는 3그룹으로
# 붕괴해야 negative memo가 실제로 효과가 있다. 붕괴하지 않으면 #166은 무의미
# 하므로 이 테스트가 그 전제를 지킨다.
_S6E1_REPLAY = (
    [(8.754429902767564, (8.776392621802065, 8.737677764250822, 8.734972475765097,
                           8.776041865834122, 8.747064786185712))] * 34
    + [(8.754424521977178, (8.776392621802065, 8.737663089503126, 8.734966771347219,
                             8.776035341047772, 8.747064786185712))] * 3
    + [(8.754424521977178, (8.776392621802065, 8.737663089503126, 8.734966771347219,
                             8.776035341047773, 8.747064786185712))]  # 마지막 fold ULP 차이
    + [(8.755802672173909, (8.778066170260056, 8.74077625508704, 8.737012746244252,
                             8.775355128583746, 8.747803060694443))]
)


def test_rounded_signature_collapses_s6e1_replay():
    assert len(_S6E1_REPLAY) == 39
    signatures = {_rounded_signature(cv, list(folds)) for cv, folds in _S6E1_REPLAY}
    assert len(signatures) == 3


def test_rounded_signature_does_not_over_merge_distinct_candidates():
    """cv_score가 5.4e-6 이상 차이 나는 실제 서로 다른 후보는 9자리 반올림에도
    섞이지 않는다 — round(...,9)가 관측된 최소 후보 간 간극보다 훨씬 촘촘함."""
    sig_a = _rounded_signature(8.754429902767564, [1.0])
    sig_b = _rounded_signature(8.754424521977178, [1.0])
    assert sig_a != sig_b



def test_ctx_key_changes_with_best_source():
    """best_source가 바뀌면(=승격 발생) 캐시 키도 달라져 자동 무효화된다."""
    key_a = _ctx_key(best_source="class Patch: pass")
    key_b = _ctx_key(best_source="class Patch: x = 1")
    assert key_a != key_b


def test_ctx_key_changes_with_train_schema():
    """train90 컬럼 구성이 바뀌면(DROP_COLS/EXTRA_TRAIN_PATHS 등 데이터 설정
    변경의 결과) 캐시 키도 달라진다 — comp 모듈 접근 없이 train90 자체로 판별."""
    wider = _COMMON["train90"].with_columns(pl.lit(1.0).alias("extra"))
    key_a = _ctx_key(train90=_COMMON["train90"])
    key_b = _ctx_key(train90=wider)
    assert key_a != key_b



def test_confirm_memo_hit_skips_all_eval():
    fake = _FakeCache()
    stored = ConfirmResult(confirmed=False, holdout_score=0.75, seed_gains={"7": {}}, holdout_regressed=True)
    key = _ctx_key()
    fake.memos[fake._memo_key(key, 0.85, [7])] = stored

    with patch("cycle.promotion.eval_isolated") as mock_eval:
        result = confirm_and_measure(
            **_COMMON, holdout10=_df(), confirm_seeds=[7],
            cache=fake, competition_id="comp1", candidate_cv=0.85, candidate_fold_scores=[0.85],
        )
    mock_eval.assert_not_called()
    assert result is stored


def test_memo_stored_on_measured_rejection():
    fake = _FakeCache()
    with patch("cycle.promotion.eval_isolated", side_effect=_paired_side_effect(candidate_result=_fail())):
        result = confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[7],
            cache=fake, competition_id="comp1", candidate_cv=0.85, candidate_fold_scores=[0.85],
        )
    assert result.confirmed is False
    assert len(fake.put_memo_calls) == 1


def test_memo_not_stored_on_error_rejection():
    """eval 에러로 인한 거부는 memo에 담지 않는다 — 일시적 OOM/CPU-kill을
    영구 판정으로 굳히지 않기 위해(ADR-027/028)."""
    fake = _FakeCache()
    with patch("cycle.promotion.eval_isolated", side_effect=_paired_side_effect(candidate_result=_err())):
        result = confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[7],
            cache=fake, competition_id="comp1", candidate_cv=0.85, candidate_fold_scores=[0.85],
        )
    assert result.confirmed is False
    assert fake.put_memo_calls == []


def test_memo_not_stored_on_confirmation():
    """확정 승격은 캐시하지 않는다 — 같은 코드가 재현될 일이 없다."""
    fake = _FakeCache()
    with patch("cycle.promotion.eval_isolated", side_effect=_paired_side_effect()):
        result = confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[7],
            cache=fake, competition_id="comp1", candidate_cv=0.85, candidate_fold_scores=[0.85],
        )
    assert result.confirmed is True
    assert fake.put_memo_calls == []


def test_cache_none_default_ignores_candidate_cv():
    """cache=None이면 candidate_cv/candidate_fold_scores를 줘도 memo 조회를
    시도하지 않는다 — 기존 동작 완전 불변이 기본값이어야 한다."""
    with patch("cycle.promotion.eval_isolated", side_effect=_paired_side_effect()) as mock_eval:
        result = confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[7],
            cache=None, candidate_cv=0.85, candidate_fold_scores=[0.85],
        )
    assert mock_eval.call_count == 2  # baseline + candidate, 정상 실행
    assert result.confirmed is True



def test_baseline_cv_cache_hit_skips_baseline_eval():
    fake = _FakeCache()
    key = _ctx_key()
    fake.baselines[(key, "cv", 7)] = 0.84

    baseline_calls = 0

    def _se(*args, **kwargs):
        nonlocal baseline_calls
        if kwargs.get("best_source") is None:
            baseline_calls += 1
            return _baseline()
        return _ok()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[7],
            cache=fake, competition_id="comp1", candidate_cv=0.85, candidate_fold_scores=[0.85],
        )
    assert baseline_calls == 0
    assert result.confirmed is True
    assert result.seed_gains["7"]["baseline_cv"] == pytest.approx(0.84)


def test_baseline_holdout_cached_across_candidates():
    """같은 컨텍스트(best_source/train90 등 불변)의 서로 다른 후보 2개가
    baseline holdout eval을 1번만 공유한다."""
    fake = _FakeCache()
    holdout_baseline_calls = 0

    def _se(*args, **kwargs):
        nonlocal holdout_baseline_calls
        if kwargs.get("holdout_data") is not None:
            if kwargs.get("best_source") is None:
                holdout_baseline_calls += 1
                return _ok_with_holdout(holdout_score=0.80)
            return _ok_with_holdout(holdout_score=0.90)
        if kwargs.get("best_source") is None:
            return _baseline()
        return _ok()

    common_kwargs = dict(
        **_COMMON, holdout10=_df(), confirm_seeds=[7],
        cache=fake, competition_id="comp1",
    )
    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        confirm_and_measure(**common_kwargs, candidate_cv=0.85, candidate_fold_scores=[0.85])
    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        confirm_and_measure(**common_kwargs, candidate_cv=0.86, candidate_fold_scores=[0.86])

    assert holdout_baseline_calls == 1
    holdout_puts = [c for c in fake.put_baseline_calls if c[0] == "holdout"]
    assert len(holdout_puts) == 1


def test_baseline_eval_error_not_cached():
    """baseline eval이 실패하면 캐시에 저장하지 않는다 — 일시적 실패를 굳히지
    않기 위해."""
    fake = _FakeCache()
    with patch("cycle.promotion.eval_isolated", side_effect=_paired_side_effect(baseline_result=_err())):
        confirm_and_measure(
            **_COMMON, holdout10=None, confirm_seeds=[7],
            cache=fake, competition_id="comp1", candidate_cv=0.85, candidate_fold_scores=[0.85],
        )
    assert fake.put_baseline_calls == []



def test_cross_seed_error_rejection_skips_holdout_eval():
    """candidate eval 에러로 cross-seed가 거부하면 holdout eval을 아예 안 돈다
    — 이미 confirmed=False로 확정돼 판정을 못 바꾸고, 실측(s6e8)상 8.3분짜리
    낭비였다."""
    holdout_called = False

    def _se(*args, **kwargs):
        nonlocal holdout_called
        if kwargs.get("holdout_data") is not None:
            holdout_called = True
            return _ok_with_holdout()
        if kwargs.get("best_source") is None:
            return _baseline()
        return _err()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(**_COMMON, holdout10=_df(), confirm_seeds=[7])
    assert result.confirmed is False
    assert result.holdout_score is None
    assert holdout_called is False


def test_baseline_eval_error_also_skips_holdout():
    """baseline eval 크래시로 cross-seed가 거부되는 경로도 에러 기반 거부이므로
    holdout을 스킵한다."""
    holdout_called = False

    def _se(*args, **kwargs):
        nonlocal holdout_called
        if kwargs.get("holdout_data") is not None:
            holdout_called = True
            return _ok_with_holdout()
        if kwargs.get("best_source") is None:
            return _err()
        return _ok()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(**_COMMON, holdout10=_df(), confirm_seeds=[7])
    assert result.confirmed is False
    assert holdout_called is False


def test_cross_seed_measured_rejection_still_measures_holdout():
    """cross-seed가 측정값(gain<=0)으로 거부해도 holdout은 그대로 측정한다 —
    holdout_score는 overfit_gap 드리프트 관측(holdout_cv_gap_trend 뷰 등)에
    쓰이는 telemetry라 값어치가 warm eval 1회보다 크다."""
    holdout_called = False

    def _se(*args, **kwargs):
        nonlocal holdout_called
        if kwargs.get("holdout_data") is not None:
            holdout_called = True
            return _ok_with_holdout()
        if kwargs.get("best_source") is None:
            return _baseline()
        return _fail()

    with patch("cycle.promotion.eval_isolated", side_effect=_se):
        result = confirm_and_measure(**_COMMON, holdout10=_df(), confirm_seeds=[7])
    assert result.confirmed is False
    assert holdout_called is True
    assert result.holdout_score is not None
