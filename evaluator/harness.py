"""CV 평가 하네스 — BasePipeline/PatchedPipeline 실행, 유의성 검정, 누수 가드, ensemble_spec.

runtime/isolate.py가 이 모듈을 별도 subprocess에서 호출해 격리 실행한다.
"""
from __future__ import annotations

import inspect
import re
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
# 회귀 error 메트릭(rmse/mae/rmsle)이 trivial baseline(train 타깃 평균 예측)보다
# 이 배수 이상 좋으면 스케일/타깃 누수로 간주한다. _check_preprocess_target_leak은
# "훅이 target 컬럼을 읽었는가"만 보는 메커니즘 검사라 다른 경로의 스케일 누수
# (예: postprocess_predictions의 잘못된 역변환)는 못 잡는다 — 이 비율 가드가 그
# 결과 기반 2차 방어선(decisions.md ADR-015/ADR-025).
_REGRESSION_IMPLAUSIBLE_BASELINE_RATIO = 10.0


def is_significant_gain(
    gain_vs_best: float | None,
    cv_fold_var: float,
    candidate_fold_scores: list[float] | None = None,
    baseline_fold_scores: list[float] | None = None,
    metric_sign: int = 1,
) -> bool:
    """gain이 fold noise보다 큰 경우만 True.

    candidate/baseline이 같은 seed·같은 fold split에서 나온 fold_scores 쌍을
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
    preprocess 단계는 타깃 변환(log1p 등)이 정당하므로 마스킹 대상 외 —
    대신 _check_preprocess_target_leak이 별도 경로로 검사한다.
    """
    if target not in df.columns:
        return df
    return df.with_columns(pl.lit(None, dtype=df[target].dtype).alias(target))


def _check_preprocess_target_leak(
    pipeline: "BasePipeline | PatchedPipeline",
    tr: pl.DataFrame,
    va: pl.DataFrame,
    ctx: "PipelineContext",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """preprocess 훅이 valid split의 타깃 값에 의존하는 피처를 만드는지 검사.

    실제 valid와 _mask_target(valid) 버전 각각으로 preprocess를 호출해 결과가
    다르면(또는 마스킹 버전이 크래시하면) valid의 타깃을 읽어 파생시킨 것으로 판정한다.
    동일 입력 재호출이 결정적이라는 전제(fold split·모델 모두 ctx.seed 고정)에 의존한다.

    실제(비마스킹) 호출 결과 (tr_real, va_real)를 반환한다 — 호출부(evaluate_pipeline)가
    fold 0에서 이 결과를 재사용해 preprocess 호출을 3회가 아닌 2회로 유지한다.
    """
    target = ctx.target_col
    tr_real, va_real = pipeline.preprocess(tr, va, target, ctx)

    try:
        _tr_masked, va_masked = pipeline.preprocess(tr, _mask_target(va, target), target, ctx)
    except Exception as exc:
        raise ValueError(
            "suspected target leakage in preprocess: masked-valid re-run crashed "
            f"({exc!r}) — hook likely depends on the validation split's target values"
        ) from exc

    real_cols = [c for c in va_real.columns if c != target]
    masked_cols = [c for c in va_masked.columns if c != target]
    if real_cols != masked_cols or not va_real.select(real_cols).equals(va_masked.select(masked_cols)):
        raise ValueError(
            "suspected target leakage in preprocess: validation split's post-preprocess "
            "features differ depending on whether the target is masked — hook reads the "
            "target column directly on the validation split"
        )
    return tr_real, va_real


def dummy_target_value(train: pl.DataFrame, target_col: str):
    """실제 추론(test) 시점과 동일하게 쓸 더미 타깃 값 — train 실재값이어야 한다.

    synthetic placeholder(0, NaN 등)는 Patch의 exhaustive target 인코딩에서
    매핑 밖 값이라 크래시한다. bin/submit.py와 cycle/promotion.py(audit holdout)가
    이 함수를 공유해 "추론 시점에 타깃이 어떻게 보이는가"를 한 곳에서 정의한다.
    """
    return train[target_col][0]


def replace_with_dummy_target(df: pl.DataFrame, target_col: str, train: pl.DataFrame) -> pl.DataFrame:
    """df(test 또는 holdout)의 타깃 컬럼을 dummy_target_value로 교체한 새 프레임 반환.

    _mask_target(null)과 달리 실제 train 값이라 preprocess 훅의 exhaustive target
    인코딩 등이 크래시하지 않는다 — bin/submit.py의 실제 추론 조건과 동일.
    """
    return df.with_columns(
        pl.lit(dummy_target_value(train, target_col)).cast(train[target_col].dtype).alias(target_col)
    )


def _encode_residual_categoricals(
    Xtr: pl.DataFrame, Xva: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """str 컬럼에 null이 섞이면 sorted()가 None/str 비교로 TypeError — null 제외 후
    정렬, replace_strict(default=-1)로 unseen 카테고리와 동일 처리."""
    str_cols = [c for c, dt in zip(Xtr.columns, Xtr.dtypes) if dt == pl.String]
    for c in str_cols:
        values = (v for v in Xtr[c].unique().to_list() if v is not None)
        mapping = {v: i for i, v in enumerate(sorted(values))}
        Xtr = Xtr.with_columns(pl.col(c).replace_strict(mapping, default=-1).cast(pl.Int32))
        Xva = Xva.with_columns(pl.col(c).replace_strict(mapping, default=-1).cast(pl.Int32))
    return Xtr, Xva

_IMPORTANCE_ACTIONS = frozenset({"feature_engineering", "preprocessing"})

_MAX_BUILD_MODEL_RETRIES = 3
_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument '(\w+)'")
_MULTIPLE_KWARG_RE = re.compile(r"got multiple values for keyword argument '(\w+)'")


def _construct_with_kwarg_retry(build_fn, params: dict):
    """params로 build_fn(params)를 호출하되, 제거된/불명 kwarg로 인한 TypeError면
    그 키 하나만 벗기고 재시도한다.

    _build_model_safe(Patch.build_model 경유)와 _build_ensemble_member(ensemble_spec
    멤버 생성)가 이 재시도 로직을 공유한다 — 둘 다 LLM이 stale
    constructor kwarg(예: LogisticRegression의 구 `multi_class`)를 params dict에
    넣는 동일한 실패 양상을 겪는다.
    """
    current = dict(params)
    last_exc: TypeError | None = None
    for _ in range(_MAX_BUILD_MODEL_RETRIES):
        try:
            return build_fn(current)
        except TypeError as exc:
            msg = str(exc)
            m = _UNEXPECTED_KWARG_RE.search(msg) or _MULTIPLE_KWARG_RE.search(msg)
            if not m or m.group(1) not in current:
                raise
            current = {k: v for k, v in current.items() if k != m.group(1)}
            last_exc = exc
    raise last_exc


def _build_model_safe(pipeline: object, params: dict, ctx: object) -> object:
    """build_model()이 제거된 kwarg를 params로 받아 생성자에서 TypeError로 죽으면
    그 kwarg 하나만 벗기고 재시도한다.

    범위 밖: 생성된 wrapper 클래스 자신의 fit() 메서드 몸체 안에서 내부적으로
    하는 하위 모델 호출은 harness가 볼 수 없는 exec된 코드 내부라 이 함수로
    못 잡는다 — ensemble_spec(선언형)이 이 범위 밖 문제 자체를 우회한다.
    """
    return _construct_with_kwarg_retry(lambda p: pipeline.build_model(p, ctx), params)


def _fit_with_early_stopping(model: object, Xtr, ytr, Xva, yva) -> None:
    """fold valid를 early stopping에 쓸 수 있는 estimator면 opt-in으로 활용한다.

    fit() 시그니처를 검사해 eval_set/X_val 지원 여부를 감지하고, 실패하면 조용히
    기존 fit(Xtr, ytr)로 폴백한다 — opt-in이라 결과가 나빠지지 않는다.

    XGBoost는 3.x부터 early_stopping_rounds가 생성자 전용이라 harness가 fit()에서
    강제할 수 없다 — 실제 조기 종료 여부는 Patch.build_model이 생성자에 설정했는지에 달려있다.
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


# ensemble_spec: 선언형 앙상블 (decisions.md ADR-023)
# Patch는 "무엇을 조합할지"만 선언하고, 모델 생성·적합·결합은 이 모듈이 전담한다.
# 기존 자유형 build_model 기반 ensemble 훅은 병행 허용.

_ENSEMBLE_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "lgbm":          {"module": "lightgbm",           "classifier": "LGBMClassifier",                "regressor": "LGBMRegressor"},
    "xgboost":       {"module": "xgboost",             "classifier": "XGBClassifier",                  "regressor": "XGBRegressor"},
    "catboost":      {"module": "catboost",             "classifier": "CatBoostClassifier",             "regressor": "CatBoostRegressor"},
    "hgb":           {"module": "sklearn.ensemble",     "classifier": "HistGradientBoostingClassifier", "regressor": "HistGradientBoostingRegressor"},
    "random_forest": {"module": "sklearn.ensemble",     "classifier": "RandomForestClassifier",         "regressor": "RandomForestRegressor"},
    "ridge":         {"module": "sklearn.linear_model", "classifier": "RidgeClassifier",                "regressor": "Ridge"},
}


def _resolve_ensemble_model_class(model_name: str, is_classification: bool) -> type:
    entry = _ENSEMBLE_MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise ValueError(
            f"ensemble_spec: unknown member model {model_name!r} — allowed: "
            f"{sorted(_ENSEMBLE_MODEL_REGISTRY)}"
        )
    import importlib
    module = importlib.import_module(entry["module"])
    cls_name = entry["classifier"] if is_classification else entry["regressor"]
    return getattr(module, cls_name)


def _build_ensemble_member(model_name: str, params: dict, ctx: "PipelineContext") -> object:
    """레지스트리 기반으로 멤버 모델을 직접 생성한다 — LLM이 작성한 코드가 아니라
    harness 자신이 생성자를 호출하므로 super() 오용·stale kwarg 문제가 구조적으로
    발생하지 않는다. random_state는 params가 명시하지 않으면 ctx.seed로 채운다."""
    cls = _resolve_ensemble_model_class(model_name, ctx.is_classification)
    base_params = dict(params or {})
    base_params.setdefault("random_state", ctx.seed)
    return _construct_with_kwarg_retry(lambda p: cls(**p), base_params)


def _weighted_majority_vote(member_preds: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """discrete label 예측(classification metric_class)의 가중 다수결."""
    from collections import Counter

    n_rows = len(member_preds[0])
    combined = []
    for i in range(n_rows):
        counter: Counter = Counter()
        for w, preds in zip(weights, member_preds):
            counter[preds[i]] += w
        combined.append(counter.most_common(1)[0][0])
    return np.array(combined, dtype=member_preds[0].dtype)


def _combine_predictions(
    member_preds: list[np.ndarray], method: str, weights: list[float], metric_class: str,
) -> np.ndarray:
    if method == "majority_vote":
        return _weighted_majority_vote(member_preds, weights)
    if method != "weighted_average":
        raise ValueError(
            f"ensemble_spec: unknown method {method!r} — use 'weighted_average' or 'majority_vote'"
        )
    total_w = sum(weights)
    combined = sum(w * np.asarray(p, dtype=float) for w, p in zip(weights, member_preds)) / total_w
    return combined


def _fit_predict_ensemble(
    spec: dict, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray | None,
    ctx: "PipelineContext", metric_class: str,
) -> np.ndarray:
    """Patch.ensemble_spec(ctx)이 선언한 멤버를 harness가 직접 생성·적합·결합한다.

    spec 형태: {"members": [{"model": "lgbm", "params": {...}}, ...],
                "method": "weighted_average" | "majority_vote", "weights": [...]}.
    method 생략 시 metric_class에 따라 기본값을 고른다(discrete label 채점이면
    다수결, 아니면 가중평균). weights 생략 시 균등가중.

    yva가 주어지면(라벨 있는 검증 데이터, CV/holdout)멤버별 early stopping에 쓴다.
    None이면(제출처럼 라벨 없는 예측 대상) 조기종료 없이 전체 학습한다 — 이 분기가
    없어서 이전엔 submit.py/holdout 양쪽이 이 함수를 아예 호출하지 못하고 각자
    build_model 단일 경로로 대체해 ensemble_spec을 조용히 무시했다(#226).
    """
    members = spec.get("members") or []
    if not members:
        raise ValueError("ensemble_spec: 'members' must be a non-empty list")

    weights = spec.get("weights") or [1.0] * len(members)
    if len(weights) != len(members):
        raise ValueError(
            f"ensemble_spec: weights length ({len(weights)}) != members length ({len(members)})"
        )
    method = spec.get("method") or ("majority_vote" if metric_class == "classification" else "weighted_average")

    member_preds: list[np.ndarray] = []
    for member in members:
        model_name = member.get("model")
        params = member.get("params") or {}
        built = _build_ensemble_member(model_name, params, ctx)
        if yva is not None:
            _fit_with_early_stopping(built, Xtr, ytr, Xva, yva)
        else:
            built.fit(Xtr, ytr)
        if metric_class == "binary_proba":
            member_preds.append(built.predict_proba(Xva)[:, 1])
        else:
            # CatBoost의 multiclass predict()는 (n, 1) shape을 반환한다(다른
            # 라이브러리와 다른 관례) — reshape(-1)로 정규화하지 않으면
            # _weighted_majority_vote가 행별 스칼라 대신 1-원소 배열을 카운트해
            # "unhashable type: numpy.ndarray"로 죽는다(실측 확인).
            member_preds.append(np.asarray(built.predict(Xva)).reshape(-1))

    return _combine_predictions(member_preds, method, weights, metric_class)


@dataclass
class EvalResult:
    cv_score: float
    cv_fold_var: float
    fold_scores: list[float]
    label: str
    gain_vs_best: float | None
    feature_importance: dict | None = None
    is_noop_tie: bool = False
    # preselect_params가 고른 params — 이전엔 evaluate_pipeline
    # 내부에서만 쓰이고 반환되지 않아 ctx.best_params의 데이터 소스가
    # 항상 비어 있었다. persist까지 흘려보내 다음 attempt가 참고할 수 있게 한다.
    selected_params: dict = field(default_factory=dict)
    # collect_oof=True일 때만 채워지는 fold별 out-of-fold 예측 (원본 행 순서).
    oof_preds: list[float] | None = None
    # metric마다 gain_vs_best 스케일이 달라 reflection_impact 전역 z-score가
    # 오염된다 — regression_error는 baseline_cv로 나눈 상대값, 나머지는 gain_vs_best 그대로.
    gain_vs_best_relative: float | None = None


@dataclass(frozen=True, slots=True)
class PipelineContext:
    target_col: str
    metric: str
    n_splits: int
    seed: int
    is_classification: bool
    prev_best: float | None = None
    action_type: str = ""
    # 확정 best 파이프라인의 params — hyperparam_search 훅이 로컬 서치에
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

    def ensemble_spec(self, ctx: PipelineContext) -> dict | None:
        """정의돼 있으면 evaluate_pipeline이 build_model 대신 이 스펙으로 멤버들을
        harness가 직접 생성·적합·결합한다(_fit_predict_ensemble)."""
        return None


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

    def ensemble_spec(self, ctx):
        fn = getattr(self.patch, "ensemble_spec", None)
        return fn(ctx) if fn else self.base.ensemble_spec(ctx)


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

    ensemble_spec이 정의된 파이프라인은 하이퍼파라미터가 스펙의 멤버별
    params에 이미 들어있어 이 탐색 대상이 아니다 — 빈 dict를 그대로 반환한다.
    """
    if pipeline.ensemble_spec(ctx) is not None:
        return {}

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
    # preprocess가 타깃을 변환(log1p 등)할 수 있으므로, 채점은 변환 이전의
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
            model = _build_model_safe(pipeline, params, ctx)
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
    반환한다. K-fold가 전체 행을 정확히 1번씩 커버하므로 NaN 없이 꽉 찬다.
    attempt마다 매번 계산하면 낭비라 기본은 False — 승격 시 merge-verify eval
    (bin/run_promote_task.py)에서만 True로 호출해 추가 평가 비용 없이 확보한다.
    """
    fn, metric_sign, metric_class = get_metric(ctx.metric)
    y = train[ctx.target_col].to_numpy()
    compute_importance = ctx.action_type in _IMPORTANCE_ACTIONS

    selected_params = preselect_params(pipeline, train, ctx)
    # fold마다 한 번씩 물으면 매번 같은 dict를 만드는 patch 구현에 낭비 — 1회 조회.
    ensemble_spec_dict = pipeline.ensemble_spec(ctx)

    fold_scores: list[float] = []
    # regression_error 메트릭 전용 trivial baseline(train fold 타깃 평균으로만
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

    for fold_idx, (tr_idx, va_idx) in enumerate(_make_folds(y, ctx)):
        tr = train[list(tr_idx)]
        va = train[list(va_idx)]
        # preprocess가 타깃을 변환(log1p 등)할 수 있으므로 채점은 변환 이전의 raw
        # 타깃(yva_raw)으로 한다 — preselect_params와 동일 계약(위 주석 참고).
        yva_raw = va[ctx.target_col].to_numpy()
        ytr_raw = tr[ctx.target_col].to_numpy()

        if fold_idx == 0:
            # 검사가 어차피 실제 입력으로 preprocess를 1회 호출하므로 그 결과를
            # 재사용 — 따로 또 부르면 fold 0만 preprocess가 3번(검사 2번+본 호출) 돈다.
            tr2, va2 = _check_preprocess_target_leak(pipeline, tr, va, ctx)
        else:
            tr2, va2 = pipeline.preprocess(tr, va, ctx.target_col, ctx)
        ytr = tr2[ctx.target_col].to_numpy()
        yva = va2[ctx.target_col].to_numpy()
        Xtr, Xva = pipeline.feature_transform(tr2, _mask_target(va2, ctx.target_col), ctx.target_col, ctx)
        Xtr = _strip_target(Xtr, ctx.target_col)
        Xva = _strip_target(Xva, ctx.target_col)
        Xtr, Xva = _encode_residual_categoricals(Xtr, Xva)

        Xtr_np = Xtr.to_numpy()
        Xva_np = Xva.to_numpy()

        if ensemble_spec_dict is not None:
            # harness가 멤버를 직접 생성·적합·결합. best_model=None: 앙상블은
            # 단일 estimator 개념이 없고, ensemble action_type은 애초에
            # _IMPORTANCE_ACTIONS 밖이라 permutation importance 대상도 아니다.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                raw_preds = _fit_predict_ensemble(
                    ensemble_spec_dict, Xtr_np, ytr, Xva_np, yva, ctx, metric_class,
                )
            best_model = None
        else:
            model = _build_model_safe(pipeline, selected_params, ctx)
            _fit_with_early_stopping(model, Xtr_np, ytr, Xva_np, yva)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if metric_class == "binary_proba":
                    raw_preds = model.predict_proba(Xva_np)[:, 1]
                else:
                    raw_preds = model.predict(Xva_np)
            best_model = model
        preds = pipeline.postprocess_predictions(raw_preds, ctx)
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

    # "구현 불가 수준"의 회귀 점수 방어 가드 — trivial mean-baseline 대비
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
    gain_vs_best_relative: float | None = None
    if ctx.prev_best is None:
        label = "neutral"
        gain_vs_best = None
    else:
        # 정확히 동일한 cv_score(부동소수 16자리까지 일치)는 정상적 확률적 학습으로는
        # 사실상 불가능 — patch hook이 base로 위임/무시되어 유효 계산이 안 바뀐 신호다
        # (hyperparam_search의 build_model params 무시, feature_engineering의
        # 기존 base와 동일한 재발명 등 action_type 무관하게 발생).
        is_noop_tie = cv_score == ctx.prev_best
        delta = metric_sign * (cv_score - ctx.prev_best)
        gain_vs_best = delta
        # degenerate 회귀 cv_score의 극단적 gain_vs_best가 reflection_impact 전역
        # z-score를 오염시킨다. label 판정은 클립 전 delta 유지, 저장값만 하한 클립.
        if baseline_cv is not None and baseline_cv > 0:
            worst_plausible_cv = baseline_cv * _REGRESSION_IMPLAUSIBLE_BASELINE_RATIO
            gain_floor = metric_sign * (worst_plausible_cv - ctx.prev_best)
            gain_vs_best = max(delta, gain_floor)
        gain_vs_best_relative = gain_vs_best
        if metric_class == "regression_error" and baseline_cv is not None and baseline_cv > 0:
            gain_vs_best_relative = gain_vs_best / baseline_cv
        if delta > LABEL_Z * fold_std:
            label = "jump"
        elif delta < -LABEL_Z * fold_std:
            label = "regression"
        else:
            label = "neutral"
        # 이 절대-마진 jump는 cycle/run.py가 promotion과 동일한
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
        gain_vs_best_relative=gain_vs_best_relative,
    )
