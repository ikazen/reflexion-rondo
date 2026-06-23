from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance as _permutation_importance
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold, ShuffleSplit

from config.settings import LABEL_Z
from evaluator.metrics import get as get_metric

_PI_REPEATS = 3
_PI_TOP_N = 20
_MAX_PARAM_CANDIDATES = 12
_LEAK_PERFECT_HIGH = 0.9999
_LEAK_PERFECT_LOW = 1e-9


def is_significant_gain(gain_vs_best: float | None, cv_fold_var: float) -> bool:
    """gain이 fold noise(LABEL_Z * fold_std)보다 큰 경우만 True — 라벨 jump 기준과 동일."""
    if gain_vs_best is None:
        return False
    return gain_vs_best > LABEL_Z * (cv_fold_var ** 0.5)


def _strip_target(df: pl.DataFrame, target: str) -> pl.DataFrame:
    return df.drop(target) if target in df.columns else df


def _encode_residual_categoricals(
    Xtr: pl.DataFrame, Xva: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    str_cols = [c for c, dt in zip(Xtr.columns, Xtr.dtypes) if dt == pl.String]
    for c in str_cols:
        mapping = {v: i for i, v in enumerate(sorted(Xtr[c].unique().to_list()))}
        Xtr = Xtr.with_columns(pl.col(c).replace_strict(mapping, default=-1).cast(pl.Int32))
        Xva = Xva.with_columns(pl.col(c).replace_strict(mapping, default=-1).cast(pl.Int32))
    return Xtr, Xva

_IMPORTANCE_ACTIONS = frozenset({"feature_engineering", "preprocessing"})


@dataclass
class EvalResult:
    cv_score: float
    cv_fold_var: float
    fold_scores: list[float]
    label: str
    gain_vs_best: float | None
    feature_importance: dict | None = None


@dataclass(frozen=True, slots=True)
class PipelineContext:
    target_col: str
    metric: str
    n_splits: int
    seed: int
    is_classification: bool
    prev_best: float | None = None
    action_type: str = ""


class BasePipeline:
    def preprocess(
        self, train: pl.DataFrame, valid: pl.DataFrame, target: str, ctx: PipelineContext
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        return train, valid

    def feature_transform(
        self, train: pl.DataFrame, valid: pl.DataFrame, target: str, ctx: PipelineContext
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        cols = [c for c in train.columns if c != target]
        return train.select(cols), valid.select(cols)

    def param_candidates(self, ctx: PipelineContext) -> list[dict]:
        return [{}]

    def build_model(self, params: dict, ctx: PipelineContext) -> object:
        if ctx.is_classification:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(random_state=ctx.seed)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=ctx.seed)

    def postprocess_predictions(self, preds: np.ndarray, ctx: PipelineContext) -> np.ndarray:
        return preds


class PatchedPipeline:
    def __init__(self, base: BasePipeline, patch: object) -> None:
        self.base = base
        self.patch = patch

    def preprocess(self, train, valid, target, ctx):
        fn = getattr(self.patch, "preprocess", None)
        return fn(train, valid, target, ctx) if fn else self.base.preprocess(train, valid, target, ctx)

    def feature_transform(self, train, valid, target, ctx):
        fn = getattr(self.patch, "feature_transform", None)
        return fn(train, valid, target, ctx) if fn else self.base.feature_transform(train, valid, target, ctx)

    def param_candidates(self, ctx):
        fn = getattr(self.patch, "param_candidates", None)
        return fn(ctx) if fn else self.base.param_candidates(ctx)

    def build_model(self, params, ctx):
        fn = getattr(self.patch, "build_model", None)
        return fn(params, ctx) if fn else self.base.build_model(params, ctx)

    def postprocess_predictions(self, preds, ctx):
        fn = getattr(self.patch, "postprocess_predictions", None)
        return fn(preds, ctx) if fn else self.base.postprocess_predictions(preds, ctx)


def _make_folds(y: np.ndarray, ctx: PipelineContext) -> list:
    if ctx.is_classification:
        kf = StratifiedKFold(n_splits=ctx.n_splits, shuffle=True, random_state=ctx.seed)
        return list(kf.split(np.zeros(len(y)), y))
    kf = KFold(n_splits=ctx.n_splits, shuffle=True, random_state=ctx.seed)
    return list(kf.split(np.zeros(len(y))))


def preselect_params(
    pipeline: BasePipeline | PatchedPipeline,
    train: pl.DataFrame,
    ctx: PipelineContext,
) -> dict:
    """Select best params via a single 80/20 inner holdout to avoid CV leakage."""
    candidates = pipeline.param_candidates(ctx)[:_MAX_PARAM_CANDIDATES]
    if len(candidates) <= 1:
        return candidates[0] if candidates else {}

    fn, metric_sign, metric_class = get_metric(ctx.metric)
    y = train[ctx.target_col].to_numpy()

    if ctx.is_classification:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=ctx.seed)
        tr_idx, va_idx = next(sss.split(np.zeros(len(y)), y))
    else:
        ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=ctx.seed)
        tr_idx, va_idx = next(ss.split(np.zeros(len(y))))

    tr = train[list(tr_idx)]
    va = train[list(va_idx)]
    tr2, va2 = pipeline.preprocess(tr, va, ctx.target_col, ctx)
    Xtr, Xva = pipeline.feature_transform(tr2, va2, ctx.target_col, ctx)
    Xtr = _strip_target(Xtr, ctx.target_col)
    Xva = _strip_target(Xva, ctx.target_col)
    Xtr, Xva = _encode_residual_categoricals(Xtr, Xva)
    ytr = tr2[ctx.target_col].to_numpy()
    yva = va2[ctx.target_col].to_numpy()
    Xtr_np = Xtr.to_numpy()
    Xva_np = Xva.to_numpy()

    best_score: float | None = None
    best_params: dict = candidates[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for params in candidates:
            model = pipeline.build_model(params, ctx)
            model.fit(Xtr_np, ytr)
            if metric_class == "binary_proba":
                raw_preds = model.predict_proba(Xva_np)[:, 1]
            else:
                raw_preds = model.predict(Xva_np)
            preds = pipeline.postprocess_predictions(raw_preds, ctx)
            score = float(fn(yva, preds))
            if best_score is None or metric_sign * score > metric_sign * best_score:
                best_score = score
                best_params = params

    return best_params


def evaluate_pipeline(
    pipeline: BasePipeline | PatchedPipeline,
    train: pl.DataFrame,
    ctx: PipelineContext,
) -> EvalResult:
    fn, metric_sign, metric_class = get_metric(ctx.metric)
    y = train[ctx.target_col].to_numpy()
    compute_importance = ctx.action_type in _IMPORTANCE_ACTIONS

    selected_params = preselect_params(pipeline, train, ctx)

    fold_scores: list[float] = []
    fold_pi_means: list[np.ndarray] = []
    feature_names: list[str] = []

    for tr_idx, va_idx in _make_folds(y, ctx):
        tr = train[list(tr_idx)]
        va = train[list(va_idx)]

        tr2, va2 = pipeline.preprocess(tr, va, ctx.target_col, ctx)
        Xtr, Xva = pipeline.feature_transform(tr2, va2, ctx.target_col, ctx)
        Xtr = _strip_target(Xtr, ctx.target_col)
        Xva = _strip_target(Xva, ctx.target_col)
        Xtr, Xva = _encode_residual_categoricals(Xtr, Xva)
        ytr = tr2[ctx.target_col].to_numpy()
        yva = va2[ctx.target_col].to_numpy()

        Xtr_np = Xtr.to_numpy()
        Xva_np = Xva.to_numpy()

        model = pipeline.build_model(selected_params, ctx)
        model.fit(Xtr_np, ytr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if metric_class == "binary_proba":
                raw_preds = model.predict_proba(Xva_np)[:, 1]
            else:
                raw_preds = model.predict(Xva_np)
        preds = pipeline.postprocess_predictions(raw_preds, ctx)
        best_model = model
        fold_scores.append(float(fn(yva, preds)))

        if compute_importance and best_model is not None:
            if not feature_names:
                feature_names = list(Xtr.columns)
            _mc = metric_class
            _fn = fn
            _ms = metric_sign
            scorer = lambda est, X, y: _ms * float(  # noqa: E731
                _fn(y, est.predict_proba(X)[:, 1] if _mc == "binary_proba" else est.predict(X))
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pi = _permutation_importance(
                    best_model, Xva_np, yva,
                    scoring=scorer,
                    n_repeats=_PI_REPEATS,
                    random_state=ctx.seed,
                    n_jobs=1,
                )
            fold_pi_means.append(pi.importances_mean)

    cv_score = float(np.mean(fold_scores))
    cv_fold_var = float(np.var(fold_scores))
    fold_std = float(np.std(fold_scores))

    if metric_class in ("binary_proba", "classification") and metric_sign > 0:
        if cv_score >= _LEAK_PERFECT_HIGH:
            raise ValueError(f"suspected target leakage: perfect cv_score={cv_score:.6f} (threshold={_LEAK_PERFECT_HIGH})")
    elif metric_class == "regression_error":
        if cv_score <= _LEAK_PERFECT_LOW:
            raise ValueError(f"suspected target leakage: perfect cv_score={cv_score:.2e} (threshold={_LEAK_PERFECT_LOW})")

    if ctx.prev_best is None:
        label = "neutral"
        gain_vs_best = None
    else:
        delta = metric_sign * (cv_score - ctx.prev_best)
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
