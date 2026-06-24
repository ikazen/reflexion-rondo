"""승격 후보 cross-seed 확인 + audit holdout 측정.

1σ 게이트(is_significant_gain) 통과 후 이 모듈을 호출한다.
confirm_and_measure가 confirmed=True를 반환해야 insert_pipeline/materialize 진행.
승격 경로 2곳(cycle/run.py, bin/run_promote_task.py)에서 공유한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

from runtime.isolate import eval_isolated

_LOG = logging.getLogger(__name__)

# best_source가 없을 때 베이스라인 평가에 쓰는 기본 패치 (= BasePipeline 그대로)
_NOOP_PATCH = "class Patch:\n    pass\n"


@dataclass(frozen=True, slots=True)
class ConfirmResult:
    confirmed: bool
    holdout_score: float | None
    seed_gains: dict | None = field(default=None)


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
    """Cross-seed paired 재현 확인 + audit holdout 1회 측정.

    cross-seed: 각 seed에서 베이스라인(best pipeline)도 같은 seed로 재평가해
    paired gain(candidate@seed - baseline@seed) > 0이어야 confirmed=True.
    holdout: holdout10 있으면 train90으로 fit, holdout10 측정 → holdout_score.
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

    return ConfirmResult(
        confirmed=confirmed,
        holdout_score=holdout_score,
        seed_gains=seed_gains if seed_gains else None,
    )


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
) -> float | None:
    """best pipeline(또는 BasePipeline)을 seed 고정으로 단독 평가해 cv_score 반환.

    에러 시 None 반환 → 호출부에서 보수적으로 승격 거부.
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
        return None
    return res.cv_score


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
        base_cv = _baseline_cv(
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
            _LOG.info("cross-seed=%d baseline eval 실패 → 승격 취소", cseed)
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
        }

        if cand.error_trace or cand.gain_vs_best is None or cand.gain_vs_best <= 0:
            _LOG.info(
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
