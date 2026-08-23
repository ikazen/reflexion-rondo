"""evaluator.harness의 평가 파이프라인 — 유의성 검정, 조기종료, 누수 가드, ensemble_spec 등 단위 테스트."""
from __future__ import annotations

import pytest
import numpy as np
import polars as pl

from config.settings import LABEL_Z
from evaluator.harness import (
    BasePipeline, EvalResult, PatchedPipeline, PipelineContext,
    evaluate_pipeline, preselect_params, is_significant_gain,
    split_audit_holdout,
    _LEAK_PERFECT_HIGH,
    _EARLY_STOPPING_ROUNDS,
    _fit_with_early_stopping,
    _build_model_safe,
    _MAX_BUILD_MODEL_RETRIES,
    _ENSEMBLE_MODEL_REGISTRY,
    _resolve_ensemble_model_class,
    _build_ensemble_member,
    _combine_predictions,
    _weighted_majority_vote,
    _fit_predict_ensemble,
)
from runtime.runner import _eval_holdout


def _ctx(is_classification: bool = True) -> PipelineContext:
    return PipelineContext(
        target_col="y",
        metric="auc" if is_classification else "mae",
        n_splits=3,
        seed=42,
        is_classification=is_classification,
    )


def _make_df(n: int = 100, is_classification: bool = True) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 3))
    y = (x[:, 0] + x[:, 1] > 0).astype(float) if is_classification else x[:, 0] * 2.0
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y})


class _SingleCandidate(BasePipeline):
    def param_candidates(self, ctx):
        return [{"max_iter": 50}]


class _NoCandidates(BasePipeline):
    def param_candidates(self, ctx):
        return []


class _TwoCandidates(BasePipeline):
    def param_candidates(self, ctx):
        return [{"max_iter": 1}, {"max_iter": 200}]



def test_is_significant_gain_none_is_false():
    assert not is_significant_gain(None, 0.01)


def test_is_significant_gain_above_threshold():
    # gain > LABEL_Z * sqrt(fold_var) => True
    fold_var = 0.04  # fold_std = 0.2
    gain = LABEL_Z * 0.2 + 0.001
    assert is_significant_gain(gain, fold_var)


def test_is_significant_gain_at_threshold_is_false():
    fold_var = 0.04  # fold_std = 0.2
    gain = LABEL_Z * 0.2  # exactly at threshold — not strictly greater
    assert not is_significant_gain(gain, fold_var)


def test_is_significant_gain_below_threshold():
    fold_var = 0.04
    gain = LABEL_Z * 0.2 - 0.001
    assert not is_significant_gain(gain, fold_var)


def test_is_significant_gain_zero_fold_var_positive_gain():
    # fold_var=0 → fold_std=0 → any positive gain is significant
    assert is_significant_gain(0.001, 0.0)


def test_is_significant_gain_zero_fold_var_zero_gain():
    assert not is_significant_gain(0.0, 0.0)



def test_paired_mode_detects_consistent_small_improvement():
    """절대 gain은 작아도 모든 fold에서 일관되게 개선되면(분산 작음) paired 검정은 유의로 본다."""
    candidate = [0.901, 0.902, 0.900, 0.903]
    baseline = [0.899, 0.900, 0.898, 0.901]
    assert is_significant_gain(0.002, 0.0001, candidate, baseline, metric_sign=1)


def test_paired_mode_rejects_noisy_improvement():
    """평균은 양수지만 fold 간 변동이 커서 유의하지 않은 경우 False."""
    candidate = [0.95, 0.80, 0.90, 0.70]
    baseline = [0.90, 0.85, 0.88, 0.75]
    assert not is_significant_gain(0.02, 0.01, candidate, baseline, metric_sign=1)


def test_paired_mode_falls_back_when_baseline_none():
    """baseline fold_scores 캐시가 없으면(콜드스타트) 절대 gain 방식으로 폴백."""
    candidate = [0.9, 0.91, 0.89]
    fold_var = 0.0001
    gain = LABEL_Z * (fold_var ** 0.5) + 0.001
    assert is_significant_gain(gain, fold_var, candidate, None)


def test_paired_mode_falls_back_on_length_mismatch():
    candidate = [0.9, 0.91, 0.89]
    baseline = [0.9, 0.91]
    fold_var = 0.0001
    gain = LABEL_Z * (fold_var ** 0.5) + 0.001
    assert is_significant_gain(gain, fold_var, candidate, baseline)


def test_paired_mode_falls_back_on_single_fold():
    """fold 1개면 분산 정의 불가 — 절대 gain 방식으로 폴백."""
    candidate = [0.9]
    baseline = [0.85]
    fold_var = 0.0001
    gain = LABEL_Z * (fold_var ** 0.5) + 0.001
    assert is_significant_gain(gain, fold_var, candidate, baseline)


def test_paired_mode_zero_variance_positive_mean_is_significant():
    """모든 fold에서 완전히 동일한 양의 delta면 분산 0 — 부호만으로 판정."""
    candidate = [0.91, 0.91, 0.91]
    baseline = [0.90, 0.90, 0.90]
    assert is_significant_gain(0.01, 0.0, candidate, baseline)


def test_paired_mode_zero_variance_nonpositive_mean_is_not_significant():
    candidate = [0.90, 0.90, 0.90]
    baseline = [0.90, 0.90, 0.90]
    assert not is_significant_gain(0.0, 0.0, candidate, baseline)


def test_paired_mode_respects_metric_sign_for_lower_is_better():
    """logloss처럼 낮을수록 좋은 metric(metric_sign=-1)에서 candidate가 baseline보다
    일관되게 낮으면(실제 개선) 유의미로 판정돼야 한다."""
    candidate = [0.10, 0.11, 0.09, 0.105]
    baseline = [0.15, 0.16, 0.14, 0.155]
    assert is_significant_gain(0.05, 0.0001, candidate, baseline, metric_sign=-1)


def test_label_z_imported_from_settings():
    from config.settings import LABEL_Z as settings_LABEL_Z
    from evaluator.harness import LABEL_Z as harness_LABEL_Z
    assert settings_LABEL_Z == harness_LABEL_Z



def test_pipeline_context_best_params_defaults_to_none():
    assert _ctx().best_params is None


def test_pipeline_context_best_params_settable():
    ctx = PipelineContext(
        target_col="y", metric="auc", n_splits=3, seed=42,
        is_classification=True, best_params={"max_depth": 4},
    )
    assert ctx.best_params == {"max_depth": 4}



_XTR, _YTR, _XVA, _YVA = np.zeros((3, 2)), np.zeros(3), np.zeros((2, 2)), np.zeros(2)


class _LGBMLikeModel:
    """lightgbm sklearn API 흉내: fit(..., eval_set=, callbacks=)."""
    def __init__(self):
        self.fit_calls: list[dict] = []

    def fit(self, X, y, sample_weight=None, eval_set=None, callbacks=None):
        self.fit_calls.append({"eval_set": eval_set, "callbacks": callbacks})


class _XGBLikeModel:
    """xgboost 3.x sklearn API 흉내: fit(..., eval_set=) — early_stopping_rounds는 생성자 전용."""
    def __init__(self):
        self.fit_calls: list[dict] = []

    def fit(self, X, y, sample_weight=None, eval_set=None, verbose=None):
        self.fit_calls.append({"eval_set": eval_set})


class _CatBoostLikeModel:
    """catboost sklearn API 흉내: fit(..., eval_set=, early_stopping_rounds=)."""
    def __init__(self):
        self.fit_calls: list[dict] = []

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None):
        self.fit_calls.append({"eval_set": eval_set, "early_stopping_rounds": early_stopping_rounds})


class _HGBLikeModel:
    """sklearn HistGradientBoosting 흉내: fit(..., X_val=, y_val=)."""
    def __init__(self):
        self.fit_calls: list[dict] = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.fit_calls.append({"X_val": X_val, "y_val": y_val})


class _PlainModel:
    """eval_set/X_val 어느 것도 지원하지 않는 estimator."""
    def __init__(self):
        self.fit_calls: list[dict] = []

    def fit(self, X, y):
        self.fit_calls.append({})


class _RaisingModel:
    """eval_set 시그니처는 있지만 실제 호출 시 예외 — 폴백 검증용."""
    def __init__(self):
        self.fit_calls: list[dict] = []

    def fit(self, X, y, eval_set=None):
        if eval_set is not None:
            raise TypeError("simulated version incompatibility")
        self.fit_calls.append({})


def test_fit_early_stopping_lightgbm_style_uses_callbacks():
    model = _LGBMLikeModel()
    _fit_with_early_stopping(model, _XTR, _YTR, _XVA, _YVA)
    assert len(model.fit_calls) == 1
    assert model.fit_calls[0]["eval_set"] == [(_XVA, _YVA)]
    assert model.fit_calls[0]["callbacks"] is not None


def test_fit_early_stopping_catboost_style_uses_rounds():
    model = _CatBoostLikeModel()
    _fit_with_early_stopping(model, _XTR, _YTR, _XVA, _YVA)
    assert model.fit_calls[0]["eval_set"] == [(_XVA, _YVA)]
    assert model.fit_calls[0]["early_stopping_rounds"] == _EARLY_STOPPING_ROUNDS


def test_fit_early_stopping_xgboost_style_eval_set_only():
    model = _XGBLikeModel()
    _fit_with_early_stopping(model, _XTR, _YTR, _XVA, _YVA)
    assert len(model.fit_calls) == 1
    assert model.fit_calls[0]["eval_set"] == [(_XVA, _YVA)]


def test_fit_early_stopping_sklearn_style_uses_x_val_y_val():
    model = _HGBLikeModel()
    _fit_with_early_stopping(model, _XTR, _YTR, _XVA, _YVA)
    assert model.fit_calls[0]["X_val"] is _XVA
    assert model.fit_calls[0]["y_val"] is _YVA


def test_fit_early_stopping_plain_model_uses_plain_fit():
    model = _PlainModel()
    _fit_with_early_stopping(model, _XTR, _YTR, _XVA, _YVA)
    assert model.fit_calls == [{}]


def test_fit_early_stopping_falls_back_on_exception():
    model = _RaisingModel()
    _fit_with_early_stopping(model, _XTR, _YTR, _XVA, _YVA)
    assert model.fit_calls == [{}]  # eval_set 시도 실패 후 plain fit로 폴백, 1회만 기록


# _build_model_safe — #74 후속: stale kwarg 프롬프트 경고가 안 먹혀 런타임 안전망 추가

class _BuildModelRejectsKwarg:
    """LogisticRegression(multi_class=...) 흉내 — 특정 kwarg가 있으면 TypeError."""
    def __init__(self, bad_key: str, exc_message: str):
        self.bad_key = bad_key
        self.exc_message = exc_message
        self.calls: list[dict] = []

    def build_model(self, params, ctx):
        self.calls.append(dict(params))
        if self.bad_key in params:
            raise TypeError(self.exc_message)
        return {"built_with": dict(params)}


def test_build_model_safe_strips_unexpected_kwarg_and_retries():
    pipeline = _BuildModelRejectsKwarg(
        "multi_class", "LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'"
    )
    result = _build_model_safe(pipeline, {"C": 1.0, "multi_class": "multinomial"}, _ctx())
    assert result == {"built_with": {"C": 1.0}}
    assert len(pipeline.calls) == 2
    assert pipeline.calls[0] == {"C": 1.0, "multi_class": "multinomial"}
    assert pipeline.calls[1] == {"C": 1.0}


def test_build_model_safe_strips_multiple_values_kwarg():
    pipeline = _BuildModelRejectsKwarg(
        "verbose", "CatBoostClassifier() got multiple values for keyword argument 'verbose'"
    )
    result = _build_model_safe(pipeline, {"iterations": 100, "verbose": False}, _ctx())
    assert result == {"built_with": {"iterations": 100}}


def test_build_model_safe_no_retry_when_fit_succeeds():
    pipeline = _BuildModelRejectsKwarg("multi_class", "unused")
    _build_model_safe(pipeline, {"C": 1.0}, _ctx())
    assert len(pipeline.calls) == 1


def test_build_model_safe_reraises_unrelated_typeerror():
    """kwarg 이름을 못 뽑아내는(패턴 불일치) TypeError는 조용히 삼키지 않고 그대로 올린다."""
    class _Unrelated:
        def build_model(self, params, ctx):
            raise TypeError("something else entirely")

    with pytest.raises(TypeError, match="something else entirely"):
        _build_model_safe(_Unrelated(), {"C": 1.0}, _ctx())


def test_build_model_safe_reraises_when_retries_exhausted():
    """계속 새로운 bad kwarg가 나오면 _MAX_BUILD_MODEL_RETRIES에서 포기하고 마지막
    예외를 올린다 — 무한 루프 방지."""
    class _AlwaysRejects:
        def __init__(self):
            self.calls = 0

        def build_model(self, params, ctx):
            self.calls += 1
            raise TypeError(f"got an unexpected keyword argument 'k{self.calls}'")

    pipeline = _AlwaysRejects()
    params = {f"k{i}": i for i in range(1, 6)}
    with pytest.raises(TypeError):
        _build_model_safe(pipeline, params, _ctx())
    assert pipeline.calls == _MAX_BUILD_MODEL_RETRIES



def test_evaluate_pipeline_returns_selected_params():
    """preselect_params가 고른 params가 EvalResult까지 흘러나와야 ctx.best_params의
    데이터 소스로 쓸 수 있다 — 이전엔 evaluate_pipeline 내부에서만 쓰이고 버려졌다."""
    result = evaluate_pipeline(_SingleCandidate(), _make_df(), _ctx())
    assert result.selected_params == {"max_iter": 50}


def test_eval_result_selected_params_defaults_to_empty_dict():
    result = EvalResult(cv_score=0.9, cv_fold_var=0.0, fold_scores=[0.9], label="neutral", gain_vs_best=None)
    assert result.selected_params == {}



def test_collect_oof_false_by_default_leaves_oof_preds_none():
    result = evaluate_pipeline(BasePipeline(), _make_df(), _ctx())
    assert result.oof_preds is None


def test_collect_oof_true_fills_every_row_no_nan():
    df = _make_df(n=120)
    result = evaluate_pipeline(BasePipeline(), df, _ctx(), collect_oof=True)
    assert result.oof_preds is not None
    assert len(result.oof_preds) == len(df)
    assert not any(v is None for v in result.oof_preds)
    import math
    assert not any(math.isnan(v) for v in result.oof_preds)



def test_preselect_single_candidate_returned_directly():
    result = preselect_params(_SingleCandidate(), _make_df(), _ctx())
    assert result == {"max_iter": 50}


def test_preselect_no_candidates_returns_empty():
    result = preselect_params(_NoCandidates(), _make_df(), _ctx())
    assert result == {}


def test_preselect_returns_one_of_candidates():
    pipeline = _TwoCandidates()
    result = preselect_params(pipeline, _make_df(), _ctx())
    assert result in pipeline.param_candidates(_ctx())


def test_preselect_is_deterministic():
    pipeline = _TwoCandidates()
    df = _make_df()
    r1 = preselect_params(pipeline, df, _ctx())
    r2 = preselect_params(pipeline, df, _ctx())
    assert r1 == r2


def test_preselect_regression_path():
    pipeline = _TwoCandidates()
    df = _make_df(is_classification=False)
    ctx = _ctx(is_classification=False)
    result = preselect_params(pipeline, df, ctx)
    assert result in pipeline.param_candidates(ctx)


def _ctx_rmse() -> PipelineContext:
    return PipelineContext(
        target_col="y",
        metric="rmse",
        n_splits=3,
        seed=42,
        is_classification=False,
    )


def test_harness_regression_rmse_scores_without_error():
    """rmse 메트릭 경로가 TypeError 없이 CV 스코어를 반환한다."""
    df = _make_df(is_classification=False)
    result = evaluate_pipeline(BasePipeline(), df, _ctx_rmse())
    assert result.cv_score > 0  # rmse는 양수 오류값으로 저장
    assert np.isfinite(result.cv_score)


# rmsle 이중 log 채점 회귀 테스트 (s5e5 phantom cv 원인, 2026-07)
# preprocess가 타깃을 log1p로 변환하면, rmsle 스코어러(evaluator/metrics.py)가 그 위에
# log1p를 한 번 더 적용해 이중 log 압축이 발생한다. harness는 반드시 변환 이전의 raw
# 타깃으로 채점해야 하며, 파이프라인이 postprocess_predictions에서 expm1로 역변환하지
# 않으면 그 사실이 legitimately 나쁜 점수로 드러나야 한다(phantom하게 좋아지면 안 됨).

def _ctx_rmsle() -> PipelineContext:
    return PipelineContext(
        target_col="y",
        metric="rmsle",
        n_splits=3,
        seed=42,
        is_classification=False,
    )


def _make_df_positive_target(n: int = 300) -> pl.DataFrame:
    """rmsle은 y>=0 요구 — Calories류 양수 타깃을 흉내."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((n, 3))
    y = 80.0 + 15.0 * x[:, 0] + rng.standard_normal(n) * 3.0
    y = np.clip(y, 1.0, None)
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y})


class _Log1pNoInversePatch:
    """preprocess에서 타깃을 log1p 변환하지만 postprocess에서 역변환을 빠뜨린 버그 패치.

    s5e5에서 실제로 관측된 패턴 — 이걸 정상 점수로 착각하면 phantom cv가 나온다.
    """
    action_type = "feature_engineering"

    def preprocess(self, train, valid, target, ctx):
        tr = train.with_columns(pl.col(target).log1p())
        va = valid.with_columns(pl.col(target).log1p())
        return tr, va

    def feature_transform(self, train, valid, target, ctx):
        cols = [c for c in train.columns if c != target]
        return train.select(cols), valid.select(cols)

    def build_model(self, params, ctx):
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    def postprocess_predictions(self, preds, ctx):
        return preds  # 버그: log 공간 예측을 그대로 반환 (expm1 누락)


class _Log1pWithInversePatch(_Log1pNoInversePatch):
    """위와 동일하지만 postprocess에서 올바르게 expm1로 역변환한다."""

    def postprocess_predictions(self, preds, ctx):
        return np.expm1(preds)


def test_rmsle_scores_against_raw_target_not_log_transformed():
    """preprocess가 타깃을 log1p 변환해도 harness는 raw 스케일 타깃으로 채점해야 한다.

    역변환을 빠뜨린 패치는 log 공간 예측과 raw 타깃이 스케일 불일치 상태라
    (fold var 대비) 크게 나쁜 rmsle을 내야 한다 — 과거 버그처럼 phantom하게
    좋은(~0.017) 점수가 나오면 안 된다.
    """
    df = _make_df_positive_target()
    result = evaluate_pipeline(PatchedPipeline(BasePipeline(), _Log1pNoInversePatch()), df, _ctx_rmsle())
    # 역변환 누락 시 raw 타깃(~80) 대비 예측(~4.5, log space)의 rmsle은 매우 크다.
    assert result.cv_score > 1.0


def test_rmsle_correct_inverse_transform_recovers_honest_score():
    """postprocess에서 expm1로 올바르게 역변환하면 정상 범위의 낮은 rmsle이 나와야 한다."""
    df = _make_df_positive_target()
    result = evaluate_pipeline(PatchedPipeline(BasePipeline(), _Log1pWithInversePatch()), df, _ctx_rmsle())
    assert 0.0 < result.cv_score < 0.5


def test_preselect_params_scores_against_raw_target_not_log_transformed():
    """preselect_params도 evaluate_pipeline과 동일한 raw-타깃 채점 계약을 따라야 한다."""

    class _TwoCandidatesLog1pNoInverse(_Log1pNoInversePatch):
        def param_candidates(self, ctx):
            return [{"tag": "a"}, {"tag": "b"}]

    df = _make_df_positive_target()
    pipeline = PatchedPipeline(BasePipeline(), _TwoCandidatesLog1pNoInverse())
    # preselect_params 호출 경로가 raw 타깃과 스케일이 안 맞는 예측을 정상 낮은 점수로
    # 착각하지 않고 끝까지 실행되는지만 확인 (에러 없이 후보 중 하나를 선택).
    result = preselect_params(pipeline, df, _ctx_rmsle())
    assert result in pipeline.param_candidates(_ctx_rmsle())



class _ScaleLeakPatch:
    """preprocess가 (작은 노이즈만 섞은) 타깃 사본을 feature로 흘리는 스케일 누수 패치.

    _make_df_positive_target은 irreducible noise(std=3)가 있어 leak 없이는 완전한
    0-residual이 나올 수 없다 — 그래서 leak feature의 잔차(noise std=0.05)가 기존
    perfect-score tripwire(_LEAK_PERFECT_LOW=1e-9)는 피해간다.

    _REGRESSION_IMPLAUSIBLE_BASELINE_RATIO 가드를 노리고 만든 픽스처였으나, valid의
    "leaked" 컬럼이 valid[target] 의존이라 _check_preprocess_target_leak(#97, GH #96
    s5e10 사후 도입)이 그보다 먼저, 더 구체적으로 잡는다 — fold_scores 계산 전 단계라
    ratio 가드는 아예 도달하지 못한다. 두 가드 다 유효하나 이 픽스처로는 이제
    target-leak 가드만 검증 가능(아래 테스트 docstring 참고).
    """
    action_type = "feature_engineering"

    def preprocess(self, train, valid, target, ctx):
        rng = np.random.default_rng(123)
        noise_tr = rng.normal(0, 0.05, size=train.height)
        noise_va = rng.normal(0, 0.05, size=valid.height)
        tr = train.with_columns((pl.col(target) + pl.Series(noise_tr)).alias("leaked"))
        va = valid.with_columns((pl.col(target) + pl.Series(noise_va)).alias("leaked"))
        return tr, va

    def feature_transform(self, train, valid, target, ctx):
        # 오직 leaked 컬럼만 남겨 leak 신호를 명확히 분리한다.
        return train.select("leaked"), valid.select("leaked")

    def build_model(self, params, ctx):
        from sklearn.linear_model import LinearRegression
        return LinearRegression()


def test_regression_phantom_guard_trips_on_scale_leak():
    """preprocess에서 valid[target] 의존 피처를 흘리면 거부돼야 한다.

    #97 이전엔 이 케이스가 _REGRESSION_IMPLAUSIBLE_BASELINE_RATIO(결과 기반 사후 감지)로
    잡혔지만, 이제 _check_preprocess_target_leak(메커니즘 자체를 직접 감지)가 fold_scores
    계산도 하기 전에 먼저 잡는다 — 더 이른 시점에 더 구체적인 원인으로 거부.
    """
    df = _make_df_positive_target()
    pipeline = PatchedPipeline(BasePipeline(), _ScaleLeakPatch())
    with pytest.raises(ValueError, match="suspected target leakage in preprocess"):
        evaluate_pipeline(pipeline, df, _ctx_rmse())


def test_regression_phantom_guard_does_not_trip_on_honest_score():
    """평범한(누수 없는) 회귀 파이프라인은 이 가드에 걸리지 않는다."""
    df = _make_df(is_classification=False)
    result = evaluate_pipeline(BasePipeline(), df, _ctx_rmse())
    assert result.cv_score > 0
    assert np.isfinite(result.cv_score)


# rmse degenerate 예측(모델이 완전히 빗나간 상수를 반환하는 등)은 raise 가드(위)에
# 걸리지 않으면서도 gain_vs_best를 비정상적으로 큰 음수로 만든다. 이 값이 그대로
# reflection_impact에 흘러가면 전역 z-score(memory/retriever.py._global_gain_stats)를
# 오염시킨다 — label 판정은 그대로 두고 저장되는 gain_vs_best만 baseline 100배
# 나쁜 지점으로 하한을 둬야 한다.

class _DegenerateRegressionPatch:
    """postprocess에서 예측에 거대한 offset을 더해 rmse를 극단적으로 나쁘게 만든다."""

    action_type = "feature_engineering"

    def postprocess_predictions(self, preds, ctx):
        return preds + 1e5


def test_degenerate_regression_gain_is_clipped_not_raw():
    """degenerate 회귀 점수의 gain_vs_best는 raw delta가 아니라 baseline 100배 하한으로 클립된다."""
    df = _make_df_positive_target()
    pipeline = PatchedPipeline(BasePipeline(), _DegenerateRegressionPatch())
    ctx = PipelineContext(
        target_col="y", metric="rmse", n_splits=3, seed=42,
        is_classification=False, prev_best=3.0,
    )
    result = evaluate_pipeline(pipeline, df, ctx)

    assert result.cv_score > 1000  # degenerate offset이 rmse를 압도적으로 키운다
    raw_delta = -1 * (result.cv_score - ctx.prev_best)
    # 클립 없이 그대로 흘렀다면 gain_vs_best가 raw_delta(수만 단위 음수)와 같아야 한다 —
    # 클립이 동작하면 훨씬 작은(덜 극단적인) 음수로 잡힌다.
    assert result.gain_vs_best > raw_delta
    assert result.gain_vs_best < 0  # 여전히 명백히 나쁜 점수라는 신호는 유지
    assert result.label == "regression"


def test_honest_regression_gain_is_not_clipped():
    """baseline과 비슷한 스케일의 정상 회귀 점수는 클립 가드의 영향을 받지 않는다."""
    df = _make_df_positive_target()
    ctx = PipelineContext(
        target_col="y", metric="rmse", n_splits=3, seed=42,
        is_classification=False, prev_best=100.0,  # 압도적으로 나쁜 prev_best로 확실한 jump 유도
    )
    result = evaluate_pipeline(BasePipeline(), df, ctx)
    raw_delta = -1 * (result.cv_score - ctx.prev_best)
    assert result.gain_vs_best == pytest.approx(raw_delta)



def test_regression_error_gain_relative_is_scaled_by_baseline():
    """regression_error는 gain_vs_best_relative = gain_vs_best / baseline_cv 여야 한다."""
    df = _make_df_positive_target()
    ctx = PipelineContext(
        target_col="y", metric="rmse", n_splits=3, seed=42,
        is_classification=False, prev_best=100.0,
    )
    result = evaluate_pipeline(BasePipeline(), df, ctx)
    assert result.gain_vs_best_relative is not None
    assert result.gain_vs_best_relative != result.gain_vs_best  # 원시값과 달라야 함(정규화됨)


def test_classification_gain_relative_equals_raw_gain():
    """AUC 등 이미 0~1 스케일인 metric은 gain_vs_best_relative가 gain_vs_best와 동일."""
    df = _make_df(is_classification=True)
    ctx = PipelineContext(
        target_col="y", metric="auc", n_splits=3, seed=42,
        is_classification=True, prev_best=0.5,
    )
    result = evaluate_pipeline(BasePipeline(), df, ctx)
    assert result.gain_vs_best_relative == result.gain_vs_best


def test_gain_relative_is_none_when_no_prev_best():
    """첫 attempt(prev_best 없음)는 gain_vs_best와 마찬가지로 relative도 None."""
    df = _make_df(is_classification=True)
    ctx = _ctx()  # prev_best 기본값 None
    result = evaluate_pipeline(BasePipeline(), df, ctx)
    assert result.gain_vs_best is None
    assert result.gain_vs_best_relative is None


def test_preselect_evaluates_all_candidates():
    evaluated: list[dict] = []

    class _TrackingPipeline(BasePipeline):
        def param_candidates(self, ctx):
            return [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}]

        def build_model(self, params, ctx):
            evaluated.append(params)
            return super().build_model(params, ctx)

    preselect_params(_TrackingPipeline(), _make_df(), _ctx())
    assert len(evaluated) == 3
    assert {p["tag"] for p in evaluated} == {"a", "b", "c"}



class _LeakyPatch:
    """feature_transform이 target 컬럼을 drop하지 않는 패치."""
    action_type = "feature_engineering"

    def feature_transform(self, train, valid, target, ctx):
        # target을 포함한 채 반환 — 누수 시뮬레이션
        return train, valid


def test_harness_strips_target_from_leaky_patch():
    """harness가 leaky patch의 target을 강제 drop해 정상 점수를 낸다."""
    df = _make_df(n=200)
    ctx = _ctx()
    base = BasePipeline()
    patch = _LeakyPatch()
    pipeline = PatchedPipeline(base, patch)
    result = evaluate_pipeline(pipeline, df, ctx)
    # target leakage 없이 정상 범위의 AUC 반환돼야 함
    assert result.cv_score < _LEAK_PERFECT_HIGH


class _StringColumnPatch:
    """feature_transform이 pl.String 컬럼을 인코딩하지 않고 남긴다."""
    action_type = "feature_engineering"

    def feature_transform(self, train, valid, target, ctx):
        cols = [c for c in train.columns if c != target]
        return train.select(cols), valid.select(cols)


def _make_df_with_strings(n: int = 120) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(1)
    x = rng.standard_normal((n, 2))
    cats = ["France", "Spain", "Germany"]
    geo = [cats[i % 3] for i in range(n)]
    # 노이즈를 충분히 섞어 AUC가 leakage tripwire(0.9999) 아래에 머물도록 함
    y = ((x[:, 0] + rng.standard_normal(n) * 3.0) > 0).astype(float)
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "geo": geo, "y": y})


def test_harness_encodes_residual_string_columns():
    """harness가 pl.String 컬럼을 자동 ordinal 인코딩해 ValueError 없이 cv를 산출한다."""
    df = _make_df_with_strings()
    ctx = _ctx()
    base = BasePipeline()
    patch = _StringColumnPatch()
    pipeline = PatchedPipeline(base, patch)
    result = evaluate_pipeline(pipeline, df, ctx)
    assert 0.0 <= result.cv_score <= 1.0


def test_harness_encodes_residual_string_column_with_nulls():
    """str 컬럼에 null이 섞여도 크래시하면 안 된다.

    _encode_residual_categoricals가 sorted()로 카테고리를 정렬하는데 None이 섞이면
    'TypeError: <' not supported between instances of NoneType and str'로 죽었다
    (Coder 패치가 아니라 harness 자체 버그). null은 미확인 카테고리와 동일하게
    -1로 인코딩돼야 한다.
    """
    import numpy as np
    rng = np.random.default_rng(3)
    n = 120
    x = rng.standard_normal((n, 2))
    geo = ["France" if i % 2 == 0 else None for i in range(n)]
    geo[1] = "Spain"  # valid fold에 non-null 카테고리도 반드시 포함
    y = ((x[:, 0] + rng.standard_normal(n) * 3.0) > 0).astype(float)
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "geo": geo, "y": y})

    base = BasePipeline()
    patch = _StringColumnPatch()
    pipeline = PatchedPipeline(base, patch)
    result = evaluate_pipeline(pipeline, df, _ctx())  # TypeError 없이 완료돼야 함
    assert 0.0 <= result.cv_score <= 1.0


def test_harness_encodes_unknown_category_as_minus_one():
    """valid에만 있는 미지 카테고리가 -1로 인코딩돼 에러 없이 처리된다."""
    import numpy as np
    rng = np.random.default_rng(2)
    n = 120
    x = rng.standard_normal((n, 2))
    geo = ["France" if i % 2 == 0 else "Spain" for i in range(n)]
    geo[1] = "Germany"  # valid fold에 반드시 포함될 위치
    y = ((x[:, 0] + rng.standard_normal(n) * 3.0) > 0).astype(float)
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "geo": geo, "y": y})

    base = BasePipeline()
    patch = _StringColumnPatch()
    pipeline = PatchedPipeline(base, patch)
    result = evaluate_pipeline(pipeline, df, _ctx())
    assert result.cv_score is not None


class _TargetEncodingLeakyPatch:
    """feature_transform이 valid[target]으로 타깃 인코딩 파생 피처를 생성한다.

    _strip_target이 새 컬럼명(te_geo)을 drop하지 않으므로, 마스킹이 없으면 누수가 생존한다.
    """
    action_type = "feature_engineering"

    def feature_transform(self, train, valid, target, ctx):
        import polars as pl
        mean_map = train.group_by("geo").agg(pl.col(target).mean().alias("te_geo"))
        tr_out = train.join(mean_map, on="geo", how="left").drop("geo")
        va_out = valid.join(mean_map, on="geo", how="left").drop("geo")
        return tr_out, va_out


def test_target_masking_blocks_derived_feature_leakage():
    """valid 타깃 마스킹 후 타깃 인코딩 파생 피처가 null 처리돼 누수 억제됨을 검증.

    마스킹 전이라면 te_geo(valid target 평균)가 label과 완전히 상관돼 AUC가 near-perfect에 도달한다.
    마스킹 후에는 te_geo가 null(→ 0으로 채워짐)이 되어 AUC가 완전누수 임계값 아래에 머문다.
    """
    df = _make_df_with_strings(n=300)
    ctx = _ctx()
    pipeline = PatchedPipeline(BasePipeline(), _TargetEncodingLeakyPatch())
    result = evaluate_pipeline(pipeline, df, ctx)
    assert result.cv_score < _LEAK_PERFECT_HIGH


class _HoldoutSelfTargetLeakPatch:
    """feature_transform이 각 split의 자기 타깃을 피처(leak)로 복사한다.

    holdout의 타깃이 실제 값 그대로 남아있으면 holdout에서 leak==label로 생존해
    near-perfect가 된다.
    """
    action_type = "feature_engineering"

    def feature_transform(self, train, valid, target, ctx):
        tr_out = train.with_columns(pl.col(target).alias("leak"))
        va_out = valid.with_columns(pl.col(target).alias("leak"))
        return tr_out, va_out


def test_eval_holdout_masks_target_derived_leakage():
    """_eval_holdout이 holdout 타깃을 dummy 상수로 치환해 파생 피처 누수를 차단함을 검증.

    #98 이전엔 별도 _mask_target(null)로 막았다 — 지금은 holdout10을 preprocess에
    넘기기 전 replace_with_dummy_target으로 치환(bin/submit.py와 동일 조건)해서
    같은 효과를 낸다. 수정 전(치환 없이 real target 그대로)이면 holdout_score가
    near-perfect(>0.9999)라 이 테스트가 실패한다.
    """
    df = _make_df_with_strings(n=300)
    ctx = _ctx()
    train90, holdout10 = split_audit_holdout(df, "y", is_classification=True)
    pipeline = PatchedPipeline(BasePipeline(), _HoldoutSelfTargetLeakPatch())
    score = _eval_holdout(pipeline, train90, holdout10, ctx)
    assert score is not None
    assert score < _LEAK_PERFECT_HIGH


class _HoldoutPreprocessSelfTargetLeakPatch:
    """preprocess가 각 split의 자기 타깃을 피처(leak)로 복사한다 — s5e10(GH #96)과
    같은 경로(feature_transform이 아니라 preprocess에서 valid 타깃을 직접 참조).

    #98 이전엔 _eval_holdout이 real target을 그대로 preprocess에 넘겨 이 경로를
    전혀 못 막았다(s5e10 실측: holdout_score 0.02135 ≈ cv_score 0.02151). 지금은
    holdout10이 dummy 상수로 치환된 뒤 preprocess를 타므로 near-perfect가 나오면 안 된다.
    """
    action_type = "preprocessing"

    def preprocess(self, train, valid, target, ctx):
        tr_out = train.with_columns(pl.col(target).alias("leak"))
        va_out = valid.with_columns(pl.col(target).alias("leak"))
        return tr_out, va_out


def test_eval_holdout_dummy_target_blocks_preprocess_leak():
    """#98: holdout의 타깃을 dummy 상수로 치환해 preprocess 경로의 valid-target
    직접 참조 leak도 near-perfect를 만들지 못하게 한다."""
    df = _make_df_with_strings(n=300)
    ctx = _ctx()
    train90, holdout10 = split_audit_holdout(df, "y", is_classification=True)
    pipeline = PatchedPipeline(BasePipeline(), _HoldoutPreprocessSelfTargetLeakPatch())
    score = _eval_holdout(pipeline, train90, holdout10, ctx)
    assert score is not None
    assert score < _LEAK_PERFECT_HIGH


class _EnsembleForHoldoutPatch:
    action_type = "ensemble"

    def ensemble_spec(self, ctx):
        return {"members": [{"model": "ridge"}, {"model": "random_forest", "params": {"n_estimators": 10}}]}


def test_eval_holdout_routes_ensemble_spec_to_declarative_path():
    """#226: _eval_holdout이 ensemble_spec을 무시하고 build_model만 호출하면
    confirm 게이트의 holdout 점수가 실제 제출될 ensemble이 아니라 엉뚱한 단일
    모델을 측정한 값이 된다 — CV 경로(evaluate_pipeline)와 다른 모델을 감사하는
    셈이라 게이트가 무의미해진다. ensemble 경로를 타면 에러 없이 finite score가
    나와야 한다(구체적 수치보다 '경로를 안 놓친다'가 이 테스트의 목적)."""
    df = _make_df(n=200, is_classification=False)
    ctx = _ctx(is_classification=False)
    train90, holdout10 = split_audit_holdout(df, "y", is_classification=False)
    pipeline = PatchedPipeline(BasePipeline(), _EnsembleForHoldoutPatch())
    score = _eval_holdout(pipeline, train90, holdout10, ctx)
    assert score is not None
    assert np.isfinite(score)


class _PerfectLoglossLeakyViaPreprocessPatch:
    """preprocess가 valid[target]을 새 컬럼명(leaked)으로 복사하고,
    postprocess_predictions가 모델 출력을 극값(1e-10 / 1-1e-10)으로 클리핑한다.

    preprocess는 타깃 변환이 정당하므로 마스킹 대상 외다 — 이 경로의 완전 누수는
    트립와이어(else 절: cv_score <= _LEAK_PERFECT_LOW)가 백스톱으로 잡아야 한다.

    logloss ≈ -log(1-1e-10) ≈ 1e-10 < _LEAK_PERFECT_LOW(1e-9) 이므로 트립와이어 발동.
    """
    action_type = "feature_engineering"

    def preprocess(self, train, valid, target, ctx):
        tr_out = train.with_columns(pl.col(target).cast(pl.Float64).alias("leaked"))
        va_out = valid.with_columns(pl.col(target).cast(pl.Float64).alias("leaked"))
        return tr_out, va_out

    def feature_transform(self, train, valid, target, ctx):
        cols = [c for c in train.columns if c != target]
        return train.select(cols), valid.select(cols)

    def postprocess_predictions(self, preds, ctx):
        # 모델이 leaked 피처로 near-perfect로 학습했으므로 preds > 0.5 == y=1
        # 극값으로 클리핑 → logloss ≈ 1e-10 < 1e-9 → tripwire 발동
        return np.where(preds > 0.5, 1 - 1e-10, 1e-10)


def test_logloss_tripwire_raises_on_perfect_leak():
    """logloss(metric_sign=-1)에서 완전 누수 시 트립와이어가 ValueError를 발생시킨다.

    preprocess를 통한 누수는 feature_transform 마스킹이 차단하지 않으므로, cv_score ≈ 0이
    되어 metric_sign < 0 분기(else 절)의 _LEAK_PERFECT_LOW 트립와이어가 발동해야 한다.
    """
    df = _make_df(n=300)
    ctx = PipelineContext(
        target_col="y",
        metric="logloss",
        n_splits=3,
        seed=42,
        is_classification=True,
    )
    pipeline = PatchedPipeline(BasePipeline(), _PerfectLoglossLeakyViaPreprocessPatch())
    with pytest.raises(ValueError, match="suspected target leakage"):
        evaluate_pipeline(pipeline, df, ctx)


# _check_preprocess_target_leak (#97, GH #96)
# s5e10 확정 승격 파이프라인이 preprocess에서 valid[target]으로 quantile bin을 만들어
# CV가 2.6배 "개선"됐지만 LB는 5배 악화됐다. feature_transform과 달리 preprocess는
# _mask_target을 못 받으므로(타깃 변환이 정당한 용도라) 별도 동등성 검사로 잡는다.

class _QuantileBinTargetLeakPatch:
    """s5e10 실제 승격 파이프라인과 동일한 패턴: valid[target]으로 10-quantile bin을
    만들어 feature로 흘린다."""
    action_type = "preprocessing"

    def preprocess(self, train, valid, target, ctx):
        import numpy as np
        y_train = train[target].to_numpy()
        edges = np.quantile(y_train, np.linspace(0, 1, 11))
        edges = np.unique(edges)

        def to_bins(arr):
            return pl.Series(np.digitize(arr, edges, right=False) - 1).cast(pl.Int32)

        train = train.with_columns(to_bins(y_train).alias("target_bin"))
        valid = valid.with_columns(to_bins(valid[target].to_numpy()).alias("target_bin"))
        return train, valid


def test_preprocess_quantile_bin_target_leak_raises():
    df = _make_df_positive_target()
    pipeline = PatchedPipeline(BasePipeline(), _QuantileBinTargetLeakPatch())
    with pytest.raises(ValueError, match="suspected target leakage in preprocess"):
        evaluate_pipeline(pipeline, df, _ctx_rmse())


def test_preprocess_log1p_target_transform_is_not_flagged_as_leak():
    """preprocess가 타깃 자체를 log1p 변환하는 정당한 용도는 leak 판정에서 제외돼야 한다.

    타깃 컬럼 자체의 변환은 real/masked 비교에서 타깃 컬럼을 제외하므로 걸리지 않는다 —
    이 테스트가 실패하면 정당한 preprocess 용도까지 오탐으로 막게 된 것.
    """
    df = _make_df_positive_target()
    result = evaluate_pipeline(
        PatchedPipeline(BasePipeline(), _Log1pWithInversePatch()), df, _ctx_rmsle()
    )
    assert 0.0 < result.cv_score < 0.5


class _PreprocessCrashesOnMaskedTargetPatch:
    """preprocess가 valid[target] 첫 값을 int()로 변환해 새 컬럼을 만드는 패치.

    마스킹(null) 버전에서는 int(None)이 TypeError를 던진다 — 산출물 비교 이전에
    마스킹 재실행 자체가 크래시하는 경로 검증.
    """
    action_type = "preprocessing"

    def preprocess(self, train, valid, target, ctx):
        first_val = int(valid[target][0])
        valid = valid.with_columns(pl.lit(first_val).alias("first_val"))
        return train, valid


def test_preprocess_crash_on_masked_target_is_treated_as_leak():
    df = _make_df_positive_target()
    pipeline = PatchedPipeline(BasePipeline(), _PreprocessCrashesOnMaskedTargetPatch())
    with pytest.raises(ValueError, match="suspected target leakage in preprocess"):
        evaluate_pipeline(pipeline, df, _ctx_rmse())



def test_no_prev_best_is_not_noop_tie():
    """prev_best 없음(첫 attempt)은 no-op tie 판정 대상이 아니다."""
    df = _make_df()
    result = evaluate_pipeline(BasePipeline(), df, _ctx())
    assert result.is_noop_tie is False


def test_exact_tie_with_prev_best_is_noop_tie():
    """cv_score가 prev_best와 정확히 같으면 (예: build_model이 params를 무시해
    patch가 유효 계산을 못 바꾼 경우) is_noop_tie=True."""
    df = _make_df()
    baseline = evaluate_pipeline(BasePipeline(), df, _ctx())
    ctx_with_prev = PipelineContext(
        target_col="y", metric="auc", n_splits=3, seed=42,
        is_classification=True, prev_best=baseline.cv_score,
    )
    # 동일 pipeline·동일 데이터·동일 seed → 결정적으로 같은 cv_score 재현
    result = evaluate_pipeline(BasePipeline(), df, ctx_with_prev)
    assert result.cv_score == baseline.cv_score
    assert result.is_noop_tie is True


def test_different_prev_best_is_not_noop_tie():
    """cv_score가 prev_best와 다르면 is_noop_tie=False."""
    df = _make_df()
    ctx_with_prev = PipelineContext(
        target_col="y", metric="auc", n_splits=3, seed=42,
        is_classification=True, prev_best=0.01,
    )
    result = evaluate_pipeline(BasePipeline(), df, ctx_with_prev)
    assert result.is_noop_tie is False



def test_split_audit_holdout_deterministic():
    """같은 입력 → 항상 같은 분리 (고정 seed)."""
    df = _make_df(n=200)
    train1, ho1 = split_audit_holdout(df, "y", is_classification=True)
    train2, ho2 = split_audit_holdout(df, "y", is_classification=True)
    assert train1.equals(train2)
    assert ho1.equals(ho2)


def test_split_audit_holdout_fraction():
    """holdout 비율 기본값 0.1 검증 (±1 row 허용)."""
    df = _make_df(n=200)
    train, holdout = split_audit_holdout(df, "y", is_classification=True)
    assert len(train) + len(holdout) == 200
    assert abs(len(holdout) - 20) <= 1


def test_split_audit_holdout_no_overlap():
    """train/holdout 행이 겹치지 않는다."""
    df = _make_df(n=200).with_row_index("_row_idx")
    train, holdout = split_audit_holdout(df, "y", is_classification=True)
    train_rows = set(train["_row_idx"].to_list())
    holdout_rows = set(holdout["_row_idx"].to_list())
    assert train_rows.isdisjoint(holdout_rows)
    assert len(train_rows) + len(holdout_rows) == 200


def test_split_audit_holdout_regression_path():
    """회귀(is_classification=False)도 결정적 분리."""
    df = _make_df(n=200, is_classification=False)
    train1, ho1 = split_audit_holdout(df, "y", is_classification=False)
    train2, ho2 = split_audit_holdout(df, "y", is_classification=False)
    assert train1.equals(train2)
    assert ho1.equals(ho2)


def test_split_audit_holdout_class_balance_maintained():
    """분류 시 stratify — holdout의 클래스 비율이 원본과 유사."""
    rng = np.random.default_rng(99)
    n = 300
    x = rng.standard_normal(n)
    y = (x > 0).astype(float)
    df = pl.DataFrame({"x": x, "y": y})
    train, holdout = split_audit_holdout(df, "y", is_classification=True)
    orig_ratio = float(y.mean())
    ho_ratio = holdout["y"].mean()
    assert abs(ho_ratio - orig_ratio) < 0.05


def test_tripwire_rejects_perfect_classification_score():
    """target을 직접 feature로 넘기는 극단 케이스에서 tripwire가 ValueError를 raise한다."""

    class _DirectLeakPatch:
        """preprocess가 valid의 target을 feature 컬럼으로 복제해 모델에 주입."""
        action_type = "preprocessing"

        def preprocess(self, train, valid, target, ctx):
            # target을 'leaked' 라는 별도 컬럼으로 복제해 leakage 우회 시뮬레이션
            train2 = train.with_columns(pl.col(target).alias("leaked"))
            valid2 = valid.with_columns(pl.col(target).alias("leaked"))
            return train2, valid2

    df = _make_df(n=200)
    # y와 완전히 동일한 컬럼을 추가하면 AUC=1.0 → tripwire
    ctx = PipelineContext(
        target_col="y",
        metric="auc",
        n_splits=3,
        seed=42,
        is_classification=True,
    )
    base = BasePipeline()
    pipeline = PatchedPipeline(base, _DirectLeakPatch())
    with pytest.raises(ValueError, match="suspected target leakage"):
        evaluate_pipeline(pipeline, df, ctx)


# ensemble_spec: 선언형 앙상블 (#74)
# #42 fix 이후에도 자유형 ensemble wrapper 클래스의 크래시율이 70%→55%에서
# 멈춘 근본 원인(harness가 볼 수 없는 exec된 클래스 몸체 내부의 super() 오용·
# stale kwarg·하위 모델 재구성 실패)에 대한 구조적 대안 — LLM은 "무엇을 조합할지"만
# 선언하고 harness가 생성·적합·결합을 전담한다.

def _reg_ctx() -> PipelineContext:
    return PipelineContext(target_col="y", metric="rmse", n_splits=3, seed=42, is_classification=False)


def test_resolve_ensemble_model_class_known_names():
    from sklearn.linear_model import Ridge, RidgeClassifier
    assert _resolve_ensemble_model_class("ridge", is_classification=False) is Ridge
    assert _resolve_ensemble_model_class("ridge", is_classification=True) is RidgeClassifier


def test_resolve_ensemble_model_class_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown member model"):
        _resolve_ensemble_model_class("not_a_real_model", is_classification=False)


def test_ensemble_model_registry_covers_common_libraries():
    """실제로 자유형 ensemble 사고에 등장했던 라이브러리(lgbm/xgboost/catboost)가
    레지스트리에 있어야 declarative 대안이 실질적 대체가 된다."""
    assert {"lgbm", "xgboost", "catboost", "hgb"}.issubset(_ENSEMBLE_MODEL_REGISTRY)


def test_build_ensemble_member_defaults_random_state_to_ctx_seed():
    model = _build_ensemble_member("ridge", {}, _reg_ctx())
    assert model.random_state == 42


def test_build_ensemble_member_respects_explicit_random_state():
    model = _build_ensemble_member("ridge", {"random_state": 7}, _reg_ctx())
    assert model.random_state == 7


def test_build_ensemble_member_strips_stale_kwarg_and_retries():
    """생성자에 없는 kwarg(LLM의 stale API 지식)가 params에 섞여도 _construct_with_
    kwarg_retry가 벗기고 재시도한다 — _build_model_safe와 동일 안전망 공유."""
    model = _build_ensemble_member("ridge", {"multi_class": "ovr"}, _reg_ctx())
    assert not hasattr(model, "multi_class")


def test_combine_predictions_weighted_average():
    preds = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0])]
    combined = _combine_predictions(preds, "weighted_average", [1.0, 1.0], "regression_error")
    assert combined.tolist() == pytest.approx([2.0, 3.0, 4.0])


def test_combine_predictions_weighted_average_respects_weights():
    preds = [np.array([0.0]), np.array([10.0])]
    combined = _combine_predictions(preds, "weighted_average", [3.0, 1.0], "regression_error")
    assert combined[0] == pytest.approx(2.5)  # (3*0 + 1*10) / 4


def test_combine_predictions_unknown_method_raises():
    preds = [np.array([1.0]), np.array([2.0])]
    with pytest.raises(ValueError, match="unknown method"):
        _combine_predictions(preds, "bogus_method", [1.0, 1.0], "regression_error")


def test_weighted_majority_vote_picks_higher_weighted_label():
    preds = [np.array(["a", "a", "b"]), np.array(["b", "b", "b"])]
    combined = _weighted_majority_vote(preds, weights=[1.0, 3.0])
    # 각 행에서 member2(가중치 3)가 전부 'b'를 찍으므로 'b'가 항상 이긴다.
    assert combined.tolist() == ["b", "b", "b"]


def test_weighted_majority_vote_equal_weights_majority_wins():
    preds = [np.array(["a"]), np.array(["a"]), np.array(["b"])]
    combined = _weighted_majority_vote(preds, weights=[1.0, 1.0, 1.0])
    assert combined.tolist() == ["a"]


def test_fit_predict_ensemble_empty_members_raises():
    with pytest.raises(ValueError, match="non-empty"):
        _fit_predict_ensemble(
            {"members": []}, np.zeros((4, 2)), np.zeros(4), np.zeros((2, 2)), np.zeros(2),
            _reg_ctx(), "regression_error",
        )


def test_fit_predict_ensemble_weights_length_mismatch_raises():
    spec = {"members": [{"model": "ridge"}, {"model": "ridge"}], "weights": [1.0]}
    with pytest.raises(ValueError, match="weights length"):
        _fit_predict_ensemble(
            spec, np.zeros((4, 2)), np.zeros(4), np.zeros((2, 2)), np.zeros(2),
            _reg_ctx(), "regression_error",
        )


def test_fit_predict_ensemble_regression_end_to_end():
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((60, 3))
    ytr = Xtr[:, 0] * 2 + rng.standard_normal(60) * 0.01
    Xva = rng.standard_normal((20, 3))
    yva = Xva[:, 0] * 2

    spec = {
        "members": [{"model": "ridge"}, {"model": "ridge", "params": {"alpha": 5.0}}],
        "method": "weighted_average",
        "weights": [0.7, 0.3],
    }
    preds = _fit_predict_ensemble(spec, Xtr, ytr, Xva, yva, _reg_ctx(), "regression_error")
    assert preds.shape == (20,)
    # 신호가 강한 데이터라 예측이 실제 타깃과 대체로 같은 부호/스케일이어야 함
    assert np.corrcoef(preds, yva)[0, 1] > 0.8


def test_fit_predict_ensemble_no_early_stopping_when_yva_none():
    """#226: yva=None(라벨 없는 예측 대상, 제출/전체학습 상황)이면 조기종료 없이
    전체 학습해야 한다 — 이 분기가 없어서 이전엔 bin/submit.py와 runtime/runner.py의
    holdout이 ensemble_spec을 아예 호출하지 못하고 각자 build_model 단일 경로로
    조용히 대체했다."""
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((60, 3))
    ytr = Xtr[:, 0] * 2 + rng.standard_normal(60) * 0.01
    Xtest = rng.standard_normal((20, 3))

    spec = {
        "members": [{"model": "ridge"}, {"model": "ridge", "params": {"alpha": 5.0}}],
        "method": "weighted_average",
        "weights": [0.7, 0.3],
    }
    preds = _fit_predict_ensemble(spec, Xtr, ytr, Xtest, None, _reg_ctx(), "regression_error")
    assert preds.shape == (20,)
    assert np.all(np.isfinite(preds))



class _EnsembleSpecPatch:
    action_type = "ensemble"

    def ensemble_spec(self, ctx):
        return {"members": [{"model": "ridge"}, {"model": "random_forest", "params": {"n_estimators": 10}}]}


def test_base_pipeline_ensemble_spec_defaults_to_none():
    assert BasePipeline().ensemble_spec(_reg_ctx()) is None


def test_patched_pipeline_ensemble_spec_delegates_to_patch():
    pipeline = PatchedPipeline(BasePipeline(), _EnsembleSpecPatch())
    spec = pipeline.ensemble_spec(_reg_ctx())
    assert spec is not None
    assert len(spec["members"]) == 2


def test_patched_pipeline_ensemble_spec_falls_back_to_base_when_patch_lacks_it():
    class _NoEnsemblePatch:
        action_type = "feature_engineering"

    base_with_spec = PatchedPipeline(BasePipeline(), _EnsembleSpecPatch())
    pipeline = PatchedPipeline(base_with_spec, _NoEnsemblePatch())
    assert pipeline.ensemble_spec(_reg_ctx()) is not None


def test_preselect_params_bypasses_search_when_ensemble_spec_present():
    """ensemble_spec이 있으면 param_candidates가 여러 개라도 탐색을 건너뛰고 빈 dict."""

    class _EnsembleWithManyCandidates(_EnsembleSpecPatch):
        def param_candidates(self, ctx):
            raise AssertionError("ensemble_spec이 있으면 param_candidates를 호출하면 안 됨")

    pipeline = PatchedPipeline(BasePipeline(), _EnsembleWithManyCandidates())
    result = preselect_params(pipeline, _make_df(is_classification=False), _reg_ctx())
    assert result == {}


def test_evaluate_pipeline_routes_ensemble_spec_to_declarative_path():
    # _make_df의 노이즈 없는 회귀 타깃(x0*2.0)은 앙상블이 trivial baseline을 너무
    # 압도적으로 이겨(신호가 실제로 그만큼 깨끗해서) 스케일 누수 tripwire를 오탐
    # 시킨다 — irreducible noise를 섞어 정상적인 회귀 신호로 만든다.
    rng = np.random.default_rng(3)
    n = 150
    x = rng.standard_normal((n, 3))
    y = x[:, 0] * 2.0 + rng.standard_normal(n) * 3.0
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y})
    ctx = _reg_ctx()
    pipeline = PatchedPipeline(BasePipeline(), _EnsembleSpecPatch())
    result = evaluate_pipeline(pipeline, df, ctx)
    assert np.isfinite(result.cv_score)


def test_evaluate_pipeline_ensemble_binary_classification_weighted_average():
    class _BinaryEnsemblePatch:
        action_type = "ensemble"

        def ensemble_spec(self, ctx):
            return {
                "members": [{"model": "hgb"}, {"model": "random_forest", "params": {"n_estimators": 10}}],
                "method": "weighted_average",
            }

    df = _make_df(is_classification=True, n=150)
    ctx = _ctx(is_classification=True)
    pipeline = PatchedPipeline(BasePipeline(), _BinaryEnsemblePatch())
    result = evaluate_pipeline(pipeline, df, ctx)
    assert 0.0 <= result.cv_score <= 1.0
