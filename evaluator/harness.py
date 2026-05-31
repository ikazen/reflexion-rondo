from dataclasses import dataclass
from typing import Callable
import warnings
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold, KFold

from evaluator.metrics import get as get_metric

LABEL_Z = 1.0  # fold_std 배수. decisions.md TBD — 캘리브레이션 대상


@dataclass
class EvalResult:
    cv_score: float
    cv_fold_var: float
    fold_scores: list[float]
    label: str           # jump / neutral / regression (결정적)
    gain_vs_best: float | None


def run(
    train: pl.DataFrame,
    target_col: str,
    metric: str,
    feature_fn: Callable[[pl.DataFrame, pl.DataFrame, str], tuple[pl.DataFrame, pl.DataFrame]],
    model_fn: Callable[[dict], object],
    params: dict,
    prev_best: float | None = None,
    n_splits: int = 5,
    seed: int = 42,
    is_classification: bool = True,
) -> EvalResult:
    fn, metric_sign, metric_class = get_metric(metric)

    y = train[target_col].to_numpy()
    fold_scores: list[float] = []

    if is_classification:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = kf.split(np.zeros(len(y)), y)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = kf.split(np.zeros(len(y)))

    for tr_idx, va_idx in splits:
        tr = train[list(tr_idx)]
        va = train[list(va_idx)]

        Xtr, Xva = feature_fn(tr, va, target_col)
        ytr = tr[target_col].to_numpy()
        yva = va[target_col].to_numpy()

        Xtr_np = Xtr.to_numpy()
        Xva_np = Xva.to_numpy()
        model = model_fn(params)
        model.fit(Xtr_np, ytr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if metric_class == "binary_proba":
                preds = model.predict_proba(Xva_np)[:, 1]
            else:
                preds = model.predict(Xva_np)

        fold_scores.append(float(fn(yva, preds)))

    cv_score = float(np.mean(fold_scores))
    cv_fold_var = float(np.var(fold_scores))
    fold_std = float(np.std(fold_scores))

    if prev_best is None:
        label = "neutral"
        gain_vs_best = None
    else:
        delta = metric_sign * (cv_score - prev_best)
        gain_vs_best = delta
        if delta > LABEL_Z * fold_std:
            label = "jump"
        elif delta < -LABEL_Z * fold_std:
            label = "regression"
        else:
            label = "neutral"

    return EvalResult(
        cv_score=cv_score,
        cv_fold_var=cv_fold_var,
        fold_scores=fold_scores,
        label=label,
        gain_vs_best=gain_vs_best,
    )
