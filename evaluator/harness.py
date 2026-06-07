from dataclasses import dataclass, field
from typing import Callable
import warnings
import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance as _permutation_importance
from sklearn.model_selection import StratifiedKFold, KFold

from evaluator.metrics import get as get_metric

LABEL_Z = 1.0  # fold_std 배수. decisions.md TBD — 캘리브레이션 대상
_PI_REPEATS = 3
_PI_TOP_N = 20


@dataclass
class EvalResult:
    cv_score: float
    cv_fold_var: float
    fold_scores: list[float]
    label: str           # jump / neutral / regression (결정적)
    gain_vs_best: float | None
    feature_importance: dict | None = None


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
    compute_importance: bool = False,
) -> EvalResult:
    fn, metric_sign, metric_class = get_metric(metric)

    y = train[target_col].to_numpy()
    fold_scores: list[float] = []
    fold_pi_means: list[np.ndarray] = []
    feature_names: list[str] = []

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

        if compute_importance:
            if not feature_names:
                feature_names = list(Xtr.columns)
            # metric_sign normalises direction: drop in (sign*metric) = importance
            _mc = metric_class
            _fn = fn
            _ms = metric_sign
            scorer = lambda est, X, y: _ms * float(  # noqa: E731
                _fn(y, est.predict_proba(X)[:, 1] if _mc == "binary_proba" else est.predict(X))
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pi = _permutation_importance(
                    model, Xva_np, yva,
                    scoring=scorer,
                    n_repeats=_PI_REPEATS,
                    random_state=seed,
                    n_jobs=1,
                )
            fold_pi_means.append(pi.importances_mean)

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

    feature_importance: dict | None = None
    if compute_importance and fold_pi_means and feature_names:
        agg_means = np.array(fold_pi_means).mean(axis=0)
        agg_stds = np.array(fold_pi_means).std(axis=0)
        pairs = sorted(
            zip(feature_names, agg_means.tolist(), agg_stds.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        feature_importance = {
            name: {"mean": round(float(m), 6), "std": round(float(s), 6)}
            for name, m, s in pairs[:_PI_TOP_N]
        }

    return EvalResult(
        cv_score=cv_score,
        cv_fold_var=cv_fold_var,
        fold_scores=fold_scores,
        label=label,
        gain_vs_best=gain_vs_best,
        feature_importance=feature_importance,
    )
