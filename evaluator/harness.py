from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance as _permutation_importance
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold, ShuffleSplit

_AUDIT_SEED = 2025  # 고정 seed — 대회·재시작과 무관하게 항상 동일 holdout 분리

from config.settings import LABEL_Z
from evaluator.metrics import get as get_metric

_PI_REPEATS = 3
_PI_TOP_N = 20
_MAX_PARAM_CANDIDATES = 12
_LEAK_PERFECT_HIGH = 0.9999
_LEAK_PERFECT_LOW = 1e-9
_EARLY_STOPPING_ROUNDS = 50
# issue #4: 회귀 error 메트릭(rmse/mae/rmsle)이 "타깃 평균만 예측하는" trivial baseline보다
# 이 배수 이상 좋으면 스케일/타깃 누수로 간주한다. s5e5 이중 log1p phantom(cv≈0.017)이
# trivial baseline 대비 비정상적으로 큰 개선폭을 보였던 사례의 사후 안전망 — 근본 수정은
# issue #6(raw 타깃 채점 계약). 실제 최상위 모델의 baseline 대비 개선은 대부분 수십 배
# 이내이므로 100배는 정상 개선을 오탐하지 않을 만큼 충분히 관대한 임계값이다.
_REGRESSION_IMPLAUSIBLE_BASELINE_RATIO = 100.0


def is_significant_gain(
    gain_vs_best: float | None,
    cv_fold_var: float,
    candidate_fold_scores: list[float] | None = None,
    baseline_fold_scores: list[float] | None = None,
    metric_sign: int = 1,
) -> bool:
    """gain이 fold noise보다 큰 경우만 True.

    BON-247: candidate/baseline이 같은 seed·같은 fold split에서 나온 fold_scores 쌍을
    받으면 paired per-fold delta의 t-통계로 판정한다(fold 난이도가 상관되므로
    delta 분산이 절대 분산보다 훨씬 작아 더 민감) — LABEL_Z를 그대로 임계값으로 재사용.

    baseline 캐시가 없거나(콜드스타트) fold 수가 안 맞으면 기존 절대-gain 방식
    (gain_vs_best > LABEL_Z * sqrt(cv_fold_var))으로 폴백한다.
    """
    if (
        candidate_fold_scores
        and baseline_fold_scores
        and len(candidate_fold_scores) == len(baseline_fold_scores)
        and len(candidate_fold_scores) >= 2
    ):
        deltas = [
            metric_sign * (c - b)
            for c, b in zip(candidate_fold_scores, baseline_fold_scores)
        ]
        n = len(deltas)
        mean_delta = sum(deltas) / n
        variance = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
        std_delta = variance ** 0.5
        if std_delta == 0:
            return mean_delta > 0
        t_stat = mean_delta / (std_delta / (n ** 0.5))
        return t_stat > LABEL_Z

    if gain_vs_best is None:
        return False
    return gain_vs_best > LABEL_Z * (cv_fold_var ** 0.5)


def _strip_target(df: pl.DataFrame, target: str) -> pl.DataFrame:
    return df.drop(target) if target in df.columns else df


def _mask_target(df: pl.DataFrame, target: str) -> pl.DataFrame:
    """feature_transform 직전 valid fold의 타깃을 null로 교체해 파생 피처 누수를 차단한다.

    yva는 반드시 이 호출 전 va2에서 캡처할 것.
    preprocess 단계는 타깃 변환(log1p 등)이 정당하므로 마스킹 대상 외.
    """
    if target not in df.columns:
        return df
    return df.with_columns(pl.lit(None, dtype=df[target].dtype).alias(target))


def _encode_residual_categoricals(
    Xtr: pl.DataFrame, Xva: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """str 컬럼에 null이 섞이면 sorted()가 None/str 비교로 TypeError(#42) — null 제외 후
    정렬, replace_strict(default=-1)로 unseen 카테고리와 동일 처리."""
    str_cols = [c for c, dt in zip(Xtr.columns, Xtr.dtypes) if dt == pl.String]
    for c in str_cols:
        values = (v for v in Xtr[c].unique().to_list() if v is not None)
        mapping = {v: i for i, v in enumerate(sorted(values))}
        Xtr = Xtr.with_columns(pl.col(c).replace_strict(mapping, default=-1).cast(pl.Int32))
        Xva = Xva.with_columns(pl.col(c).replace_strict(mapping, default=-1).cast(pl.Int32))
    return Xtr, Xva

_IMPORTANCE_ACTIONS = frozenset({"feature_engineering", "preprocessing"})


def _fit_with_early_stopping(model: object, Xtr, ytr, Xva, yva) -> None:
    """BON-246: fold valid를 early stopping에 쓸 수 있는 estimator면 opt-in으로 활용한다.

    LightGBM/XGBoost/CatBoost는 fit()에 eval_set을 받고(라이브러리별로 콜백/kwarg가
    다름), sklearn HistGradientBoosting은 X_val/y_val을 받는다. Patch.build_model이
    반환하는 model이 어떤 타입인지 harness는 알 수 없으므로 fit() 시그니처를 검사해
    감지하고, 무엇이든 실패하면(estimator가 이 인자들을 실제로 지원 안 하거나 조합이
    안 맞는 경우) 조용히 기존 fit(Xtr, ytr)로 폴백한다 — opt-in이라 결과가 나빠지지
    않는다(라이브러리 자체 콜백이 최선의 라운드에서 멈추므로 동일하거나 더 나음).

    XGBoost는 3.x부터 early_stopping_rounds가 생성자 전용이라 harness가 fit()에서
    강제할 수 없다 — eval_set만 전달, 실제 조기 종료 여부는 Patch.build_model이
    생성자에 early_stopping_rounds를 설정했는지에 달려있다.
    """
    try:
        sig_params = inspect.signature(model.fit).parameters
        if "eval_set" in sig_params:
            kwargs: dict = {"eval_set": [(Xva, yva)]}
            if "callbacks" in sig_params:
                import lightgbm as lgb
                kwargs["callbacks"] = [lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False)]
            elif "early_stopping_rounds" in sig_params:
                kwargs["early_stopping_rounds"] = _EARLY_STOPPING_ROUNDS
            model.fit(Xtr, ytr, **kwargs)
            return
        if "X_val" in sig_params and "y_val" in sig_params:
            model.fit(Xtr, ytr, X_val=Xva, y_val=yva)
            return
    except Exception:
        pass  # 미지원 조합/버전 차이 — 조용히 폴백
    model.fit(Xtr, ytr)


@dataclass
class EvalResult:
    cv_score: float
    cv_fold_var: float
    fold_scores: list[float]
    label: str
    gain_vs_best: float | None
    feature_importance: dict | None = None
    is_noop_tie: bool = False
    # BON-247 선행 fix: preselect_params가 고른 params — 이전엔 evaluate_pipeline
    # 내부에서만 쓰이고 반환되지 않아 ctx.best_params(BON-249)의 데이터 소스가
    # 항상 비어 있었다. persist까지 흘려보내 다음 attempt가 참고할 수 있게 한다.
    selected_params: dict = field(default_factory=dict)
    # BON-248: collect_oof=True일 때만 채워지는 fold별 out-of-fold 예측 (원본 행 순서).
    oof_preds: list[float] | None = None


@dataclass(frozen=True, slots=True)
class PipelineContext:
    target_col: str
    metric: str
    n_splits: int
    seed: int
    is_classification: bool
    prev_best: float | None = None
    action_type: str = ""
    # BON-249: 확정 best 파이프라인의 params — hyperparam_search 훅이 로컬 서치에
    # 참고할 수 있는 advisory 필드. 훅이 무시해도 무해(강제 소비 아님).
    best_params: dict | None = None


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


def split_audit_holdout(
    train: pl.DataFrame,
    target: str,
    is_classification: bool,
    frac: float = 0.1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """고정 seed(_AUDIT_SEED)로 대회 단위 1회 결정적 분리.

    반환: (train90, holdout10).
    train90은 모든 CV/preselect에 사용하고, holdout10은 승격 시 1회 측정·기록에만 사용한다.
    내부 k-fold나 파라미터 선택에 절대 사용하지 않는다.
    """
    n = len(train)
    y = train[target].to_numpy()
    if is_classification:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=_AUDIT_SEED)
        tr_idx, ho_idx = next(sss.split(np.zeros(n), y))
    else:
        ss = ShuffleSplit(n_splits=1, test_size=frac, random_state=_AUDIT_SEED)
        tr_idx, ho_idx = next(ss.split(np.zeros(n)))
    return train[list(tr_idx)], train[list(ho_idx)]


def preselect_params(
    pipeline: BasePipeline | PatchedPipeline,
    train: pl.DataFrame,
    ctx: PipelineContext,
) -> dict:
    """Select best params via a single 80/20 inner holdout.

    트레이드오프: 이 80/20 inner split이 이후 k-fold와 동일한 train에서 추출되므로
    낙관 편향(optimistic bias)이 잔존한다. per-fold nested CV(옵션 B)가 정석이나
    계산 비용(k^2 모델 피팅)이 크다. 현재 구현은 단일 inner holdout으로 절충
    (see docs/decisions.md ADR-021).
    """
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
    # issue #6: preprocess가 타깃을 변환(log1p 등)할 수 있으므로, 채점은 변환 이전의
    # raw 타깃(yva_raw)으로 한다. yva(변환된 값)는 early stopping의 eval_set에만 쓴다.
    # 파이프라인이 log 공간에서 학습했다면 postprocess_predictions에서 raw 스케일로
    # inverse-transform 해서 반환해야 점수가 정상적으로 나온다 — submit.py와 동일 계약.
    yva_raw = va[ctx.target_col].to_numpy()
    tr2, va2 = pipeline.preprocess(tr, va, ctx.target_col, ctx)
    ytr = tr2[ctx.target_col].to_numpy()
    yva = va2[ctx.target_col].to_numpy()
    Xtr, Xva = pipeline.feature_transform(tr2, _mask_target(va2, ctx.target_col), ctx.target_col, ctx)
    Xtr = _strip_target(Xtr, ctx.target_col)
    Xva = _strip_target(Xva, ctx.target_col)
    Xtr, Xva = _encode_residual_categoricals(Xtr, Xva)
    Xtr_np = Xtr.to_numpy()
    Xva_np = Xva.to_numpy()

    best_score: float | None = None
    best_params: dict = candidates[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for params in candidates:
            model = pipeline.build_model(params, ctx)
            _fit_with_early_stopping(model, Xtr_np, ytr, Xva_np, yva)
            if metric_class == "binary_proba":
                raw_preds = model.predict_proba(Xva_np)[:, 1]
            else:
                raw_preds = model.predict(Xva_np)
            preds = pipeline.postprocess_predictions(raw_preds, ctx)
            score = float(fn(yva_raw, preds))
            if best_score is None or metric_sign * score > metric_sign * best_score:
                best_score = score
                best_params = params

    return best_params


def evaluate_pipeline(
    pipeline: BasePipeline | PatchedPipeline,
    train: pl.DataFrame,
    ctx: PipelineContext,
    collect_oof: bool = False,
) -> EvalResult:
    """collect_oof=True면 fold별 검증 예측을 원본 행 위치에 모아 EvalResult.oof_preds로
    반환한다(BON-248). K-fold가 전체 행을 정확히 1번씩 커버하므로 NaN 없이 꽉 찬다.
    attempt마다 매번 계산하면 낭비라 기본은 False — 승격 시 merge-verify eval
    (bin/run_promote_task.py)에서만 True로 호출해 추가 평가 비용 없이 확보한다.
    """
    fn, metric_sign, metric_class = get_metric(ctx.metric)
    y = train[ctx.target_col].to_numpy()
    compute_importance = ctx.action_type in _IMPORTANCE_ACTIONS

    selected_params = preselect_params(pipeline, train, ctx)

    fold_scores: list[float] = []
    # issue #4: regression_error 메트릭 전용 trivial baseline(train fold 타깃 평균으로만
    # 예측) 점수 — cv_score가 이 baseline보다 비정상적으로(수백 배) 좋으면 스케일 누수
    # 의심 신호로 쓴다. 다른 metric_class는 채우지 않는다(빈 리스트로 가드 스킵).
    baseline_fold_scores: list[float] = []
    fold_pi_means: list[np.ndarray] = []
    feature_names: list[str] = []
    # metric_class="classification"(accuracy/f1/qwk/balanced_accuracy)는 discrete label
    # 예측이라(멀티클래스는 문자열 라벨) float OOF 배열에 못 담는다(ValueError) — Ridge
    # 블렌딩(bin/blend.py) 대상도 아니므로 애초에 수집하지 않는다. blend.py는 이미
    # oof_preds IS NULL인 pipeline을 자동 제외해 이 경로와 정합적이다.
    oof = (
        np.full(len(train), np.nan)
        if (collect_oof and metric_class != "classification")
        else None
    )

    for tr_idx, va_idx in _make_folds(y, ctx):
        tr = train[list(tr_idx)]
        va = train[list(va_idx)]
        # preprocess가 타깃을 변환(log1p 등)할 수 있으므로 채점은 변환 이전의 raw
        # 타깃(yva_raw)으로 한다 — preselect_params와 동일 계약(위 주석 참고).
        yva_raw = va[ctx.target_col].to_numpy()
        ytr_raw = tr[ctx.target_col].to_numpy()

        tr2, va2 = pipeline.preprocess(tr, va, ctx.target_col, ctx)
        ytr = tr2[ctx.target_col].to_numpy()
        yva = va2[ctx.target_col].to_numpy()
        Xtr, Xva = pipeline.feature_transform(tr2, _mask_target(va2, ctx.target_col), ctx.target_col, ctx)
        Xtr = _strip_target(Xtr, ctx.target_col)
        Xva = _strip_target(Xva, ctx.target_col)
        Xtr, Xva = _encode_residual_categoricals(Xtr, Xva)

        Xtr_np = Xtr.to_numpy()
        Xva_np = Xva.to_numpy()

        model = pipeline.build_model(selected_params, ctx)
        _fit_with_early_stopping(model, Xtr_np, ytr, Xva_np, yva)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if metric_class == "binary_proba":
                raw_preds = model.predict_proba(Xva_np)[:, 1]
            else:
                raw_preds = model.predict(Xva_np)
        preds = pipeline.postprocess_predictions(raw_preds, ctx)
        best_model = model
        fold_scores.append(float(fn(yva_raw, preds)))
        if metric_class == "regression_error":
            baseline_pred = np.full_like(yva_raw, fill_value=float(np.mean(ytr_raw)), dtype=float)
            baseline_fold_scores.append(float(fn(yva_raw, baseline_pred)))
        if oof is not None:
            oof[va_idx] = preds

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

    if metric_sign > 0:
        if cv_score >= _LEAK_PERFECT_HIGH:
            raise ValueError(f"suspected target leakage: perfect cv_score={cv_score:.6f} (threshold={_LEAK_PERFECT_HIGH})")
    else:
        if cv_score <= _LEAK_PERFECT_LOW:
            raise ValueError(f"suspected target leakage: perfect cv_score={cv_score:.2e} (threshold={_LEAK_PERFECT_LOW})")

    # issue #4: "구현 불가 수준"의 회귀 점수 방어 가드 — trivial mean-baseline 대비
    # _REGRESSION_IMPLAUSIBLE_BASELINE_RATIO 배 이상 좋으면 스케일/타깃 누수로 간주.
    # metric_sign<0(rmse/mae/rmsle 전부 해당)인 regression_error 메트릭에만 적용.
    baseline_cv: float | None = None
    if metric_class == "regression_error" and metric_sign < 0 and baseline_fold_scores:
        baseline_cv = float(np.mean(baseline_fold_scores))
        if cv_score > 0 and baseline_cv / cv_score > _REGRESSION_IMPLAUSIBLE_BASELINE_RATIO:
            raise ValueError(
                f"suspected scale leakage: cv_score={cv_score:.6f} is "
                f"{baseline_cv / cv_score:.1f}x better than trivial mean-baseline={baseline_cv:.6f} "
                f"(threshold={_REGRESSION_IMPLAUSIBLE_BASELINE_RATIO}x)"
            )

    is_noop_tie = False
    if ctx.prev_best is None:
        label = "neutral"
        gain_vs_best = None
    else:
        # 정확히 동일한 cv_score(부동소수 16자리까지 일치)는 정상적 확률적 학습으로는
        # 사실상 불가능 — patch hook이 base로 위임/무시되어 유효 계산이 안 바뀐 신호다
        # (BON-239: hyperparam_search의 build_model params 무시, feature_engineering의
        # 기존 base와 동일한 재발명 등 action_type 무관하게 발생).
        is_noop_tie = cv_score == ctx.prev_best
        delta = metric_sign * (cv_score - ctx.prev_best)
        gain_vs_best = delta
        # degenerate 회귀 cv_score의 극단적 gain_vs_best가 reflection_impact 전역
        # z-score를 오염시킨다(#43). label 판정은 클립 전 delta 유지, 저장값만 하한 클립.
        if baseline_cv is not None and baseline_cv > 0:
            worst_plausible_cv = baseline_cv * _REGRESSION_IMPLAUSIBLE_BASELINE_RATIO
            gain_floor = metric_sign * (worst_plausible_cv - ctx.prev_best)
            gain_vs_best = max(delta, gain_floor)
        if delta > LABEL_Z * fold_std:
            label = "jump"
        elif delta < -LABEL_Z * fold_std:
            label = "regression"
        else:
            label = "neutral"
        # BON-267: 이 절대-마진 jump는 cycle/run.py가 promotion과 동일한
        # is_significant_gain(paired per-fold t-test) 기준으로 최종 재판정/강등한다 —
        # 여기 label은 잠정값이다. (수렴한 대회에선 이 절대 마진에 거의 도달 못 함.)

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
        is_noop_tie=is_noop_tie,
        selected_params=selected_params,
        oof_preds=oof.tolist() if oof is not None else None,
    )
