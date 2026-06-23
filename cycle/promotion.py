"""승격 후보 cross-seed 확인 + audit holdout 측정.

1σ 게이트(is_significant_gain) 통과 후 이 모듈을 호출한다.
confirm_and_measure가 confirmed=True를 반환해야 insert_pipeline/materialize 진행.
승격 경로 2곳(cycle/run.py, bin/run_promote_task.py)에서 공유한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from runtime.isolate import eval_isolated

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConfirmResult:
    confirmed: bool
    holdout_score: float | None


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
    prev_best: float | None,
    confirm_seeds: list[int],
    action_type: str = "",
) -> ConfirmResult:
    """Cross-seed 재현 확인 + audit holdout 1회 측정.

    cross-seed: confirm_seeds 전부에서 gain_vs_best > 0 재현되면 confirmed=True.
    holdout: holdout10 있으면 train90으로 fit, holdout10 측정 → holdout_score.

    비용: 승격 후보에 한해 len(confirm_seeds)회 + holdout 측정 1회 추가.
    """
    confirmed = _cross_seed_confirm(
        source=source,
        best_source=best_source,
        train90=train90,
        target_col=target_col,
        metric=metric,
        n_splits=n_splits,
        is_classification=is_classification,
        prev_best=prev_best,
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
            prev_best=prev_best,
            action_type=action_type,
        )

    return ConfirmResult(confirmed=confirmed, holdout_score=holdout_score)


def _cross_seed_confirm(
    *,
    source: str,
    best_source: str | None,
    train90: pl.DataFrame,
    target_col: str,
    metric: str,
    n_splits: int,
    is_classification: bool,
    prev_best: float | None,
    confirm_seeds: list[int],
    action_type: str,
) -> bool:
    if not confirm_seeds:
        return True

    for cseed in confirm_seeds:
        result = eval_isolated(
            source=source,
            train=train90,
            target_col=target_col,
            metric=metric,
            prev_best=prev_best,
            n_splits=n_splits,
            seed=cseed,
            is_classification=is_classification,
            action_type=action_type,
            best_source=best_source,
        )
        if result.error_trace or result.gain_vs_best is None or result.gain_vs_best <= 0:
            _LOG.info("cross-seed=%d 미재현 → 승격 취소 (gain=%s err=%s)",
                      cseed, result.gain_vs_best, bool(result.error_trace))
            return False
        _LOG.info("cross-seed=%d 재현 gain=%+.6f", cseed, result.gain_vs_best)

    return True


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
    prev_best: float | None,
    action_type: str,
) -> float | None:
    result = eval_isolated(
        source=source,
        train=train90,
        target_col=target_col,
        metric=metric,
        prev_best=prev_best,
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
