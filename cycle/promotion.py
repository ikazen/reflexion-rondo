"""승격 후보 cross-seed 확인 + audit holdout 측정.

1σ 게이트(is_significant_gain) 통과 후 이 모듈을 호출한다.
confirm_and_measure가 confirmed=True를 반환해야 insert_pipeline/materialize 진행.
승격 경로 2곳(cycle/run.py, bin/run_promote_task.py)에서 공유한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

from evaluator.metrics import get as get_metric
from runtime.isolate import eval_isolated

_LOG = logging.getLogger(__name__)

# best_source가 없을 때 베이스라인 평가에 쓰는 기본 패치 (= BasePipeline 그대로)
_NOOP_PATCH = "class Patch:\n    pass\n"


@dataclass(frozen=True, slots=True)
class ConfirmResult:
    confirmed: bool
    holdout_score: float | None
    seed_gains: dict | None = field(default=None)
    # 현재 best(또는 콜드스타트면 BasePipeline) 대비 holdout이 악화됐는지.
    # confirmed는 이미 이 값을 반영해 AND 결합돼 있다 — 별도 필드로 노출하는 건
    # 승격 거부 사유(cross-seed 미재현 vs holdout 악화)를 로그/DB에서 구분하기 위함.
    holdout_regressed: bool = False


def effective_label(original_label: str, confirm: ConfirmResult | None) -> str:
    """confirm이 jump를 거부하면(cross-seed 미재현 또는 holdout 악화) bandit
    보상·reflection lesson에는 regression으로 반영한다 — CV 단계에서만 좋아
    보였을 뿐 실제 검증(cross-seed 재현/holdout)은 통과 못 한 방향이라는 뜻이므로.

    이 구분이 없으면 update_bandit이 attempt 생성 시점의 잠정 label(confirm
    이전)만 보고 α+=1.0을 준다 — confirm이 나중에 거부해도 그 보상은 되돌아가지
    않아, 같은 아이디어가 계속 높은 확률로 재선택되는 자기강화 루프가 생긴다
    (#164 실측: s6e1의 preprocessing 후보가 cv_score 소수점 10자리까지 동일하게
    32회 재생성 — 매번 holdout에서 거부됐지만 bandit은 그때마다 최댓값 보상을 받음).

    confirm=None(스킵됨)이거나 원본이 이미 jump가 아니면 그대로 반환 — jump만
    다운그레이드 대상이다(neutral/regression/error는 confirm을 애초에 안 탄다).
    """
    if original_label == "jump" and confirm is not None and not confirm.confirmed:
        return "regression"
    return original_label


def confirm_and_measure(
    *,
    source: str,
    best_source: str | None,
    train90: pl.DataFrame,
    holdout10: pl.DataFrame | None,
    target_col: str,
    metric: str,
    n_splits: int,
    seed: int,
    is_classification: bool,
    confirm_seeds: list[int],
    action_type: str = "",
) -> ConfirmResult:
    """Cross-seed paired 재현 확인 + audit holdout 1회 측정·게이트.

    cross-seed: 각 seed에서 베이스라인(best pipeline)도 같은 seed로 재평가해
    paired gain(candidate@seed - baseline@seed) > 0이어야 confirmed=True. seed만
    바꾼 CV라 preprocess의 valid-target 의존 누수처럼 seed 불변인 문제는 못 잡는다.
    holdout: holdout10 있으면 train90으로 fit, holdout10 측정 → holdout_score.
    후보뿐 아니라 현재 best(콜드스타트면 BasePipeline)도 같은 holdout으로 측정해
    비교한다 — 후보가 더 나쁘면(holdout_regressed) confirmed를 강제로 False로
    떨어뜨린다. _eval_holdout(runtime/runner.py)이 dummy target으로 실제 추론
    조건을 재현하므로 cross-seed가 못 잡는 누수를 값 자체로 걸러낼 수 있다.
    baseline holdout을 측정 못 하면(에러) 비교 근거가 없으므로 보수적으로
    regressed=False로 두고 막지 않는다 — 정보 없음과 악화 확인은 다르다.
    """
    confirmed, seed_gains = _cross_seed_confirm(
        source=source,
        best_source=best_source,
        train90=train90,
        target_col=target_col,
        metric=metric,
        n_splits=n_splits,
        is_classification=is_classification,
        confirm_seeds=confirm_seeds,
        action_type=action_type,
    )

    holdout_score: float | None = None
    holdout_regressed = False
    if holdout10 is not None:
        holdout_score = _measure_holdout(
            source=source,
            best_source=best_source,
            train90=train90,
            holdout10=holdout10,
            target_col=target_col,
            metric=metric,
            n_splits=n_splits,
            seed=seed,
            is_classification=is_classification,
            action_type=action_type,
        )
        if holdout_score is not None:
            baseline_holdout_score = _measure_holdout(
                source=best_source if best_source else _NOOP_PATCH,
                best_source=None,
                train90=train90,
                holdout10=holdout10,
                target_col=target_col,
                metric=metric,
                n_splits=n_splits,
                seed=seed,
                is_classification=is_classification,
                action_type=action_type,
            )
            if baseline_holdout_score is not None:
                _, metric_sign, _ = get_metric(metric)
                holdout_regressed = (
                    metric_sign * holdout_score < metric_sign * baseline_holdout_score
                )
                if holdout_regressed:
                    _LOG.warning(
                        "holdout 악화로 승격 거부: candidate=%.6f baseline=%.6f",
                        holdout_score, baseline_holdout_score,
                    )

    return ConfirmResult(
        confirmed=confirmed and not holdout_regressed,
        holdout_score=holdout_score,
        seed_gains=seed_gains if seed_gains else None,
        holdout_regressed=holdout_regressed,
    )


_ERROR_TRUNCATE_LEN = 500


def _baseline_cv(
    *,
    best_source: str | None,
    train90: pl.DataFrame,
    target_col: str,
    metric: str,
    n_splits: int,
    is_classification: bool,
    seed: int,
    action_type: str,
) -> tuple[float | None, str | None]:
    """best pipeline(또는 BasePipeline)을 seed 고정으로 단독 평가해 (cv_score, error) 반환.

    에러 시 cv_score=None → 호출부에서 보수적으로 승격 거부. error는 seed_gains에
    남겨 confirm 실패가 "재현 안 됨"인지 "크래시"인지 DB만 봐서도 구분 가능하게 한다.
    """
    src = best_source if best_source else _NOOP_PATCH
    res = eval_isolated(
        source=src,
        train=train90,
        target_col=target_col,
        metric=metric,
        prev_best=None,
        n_splits=n_splits,
        seed=seed,
        is_classification=is_classification,
        action_type=action_type,
        best_source=None,
    )
    if res.error_trace or res.cv_score is None:
        _LOG.warning("baseline eval failed seed=%d err=%s", seed, bool(res.error_trace))
        err = (res.error_trace or "unknown (cv_score is None with no error_trace)")
        return None, err[:_ERROR_TRUNCATE_LEN]
    return res.cv_score, None


def _cross_seed_confirm(
    *,
    source: str,
    best_source: str | None,
    train90: pl.DataFrame,
    target_col: str,
    metric: str,
    n_splits: int,
    is_classification: bool,
    confirm_seeds: list[int],
    action_type: str,
) -> tuple[bool, dict]:
    if not confirm_seeds:
        return True, {}

    seed_gains: dict = {}

    for cseed in confirm_seeds:
        base_cv, base_err = _baseline_cv(
            best_source=best_source,
            train90=train90,
            target_col=target_col,
            metric=metric,
            n_splits=n_splits,
            is_classification=is_classification,
            seed=cseed,
            action_type=action_type,
        )
        if base_cv is None:
            _LOG.warning("cross-seed=%d baseline eval 실패 → 승격 취소: %s", cseed, base_err)
            seed_gains[str(cseed)] = {
                "baseline_cv": None,
                "candidate_cv": None,
                "gain": None,
                "error": f"baseline: {base_err}",
            }
            return False, seed_gains

        cand = eval_isolated(
            source=source,
            train=train90,
            target_col=target_col,
            metric=metric,
            prev_best=base_cv,
            n_splits=n_splits,
            seed=cseed,
            is_classification=is_classification,
            action_type=action_type,
            best_source=best_source,
        )

        seed_gains[str(cseed)] = {
            "baseline_cv": base_cv,
            "candidate_cv": cand.cv_score,
            "gain": cand.gain_vs_best,
            "error": (cand.error_trace or "")[:_ERROR_TRUNCATE_LEN] or None,
        }

        if cand.error_trace or cand.gain_vs_best is None or cand.gain_vs_best <= 0:
            _LOG.warning(
                "cross-seed=%d 미재현 → 승격 취소 (baseline=%.6f candidate=%s gain=%s err=%s)",
                cseed, base_cv, cand.cv_score, cand.gain_vs_best, bool(cand.error_trace),
            )
            return False, seed_gains
        _LOG.info(
            "cross-seed=%d 재현 baseline=%.6f candidate=%.6f gain=%+.6f",
            cseed, base_cv, cand.cv_score, cand.gain_vs_best,
        )

    return True, seed_gains


def _measure_holdout(
    *,
    source: str,
    best_source: str | None,
    train90: pl.DataFrame,
    holdout10: pl.DataFrame,
    target_col: str,
    metric: str,
    n_splits: int,
    seed: int,
    is_classification: bool,
    action_type: str,
) -> float | None:
    result = eval_isolated(
        source=source,
        train=train90,
        target_col=target_col,
        metric=metric,
        prev_best=None,
        n_splits=n_splits,
        seed=seed,
        is_classification=is_classification,
        action_type=action_type,
        best_source=best_source,
        holdout_data=holdout10,
    )
    if result.holdout_score is not None:
        _LOG.info("holdout_score=%.6f", result.holdout_score)
    return result.holdout_score
