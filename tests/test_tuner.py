"""evaluator.tuner — Optuna 튜닝 레인(#230) 단위 테스트. n_trials를 작게 줘 빠르게 돈다."""
from __future__ import annotations

import numpy as np
import optuna
import polars as pl
import pytest

from evaluator.harness import BasePipeline, PatchedPipeline, PipelineContext
from evaluator.tuner import (
    TunerResult,
    _to_result,
    infer_registry_model,
    tune_confirmed_pipeline,
    tune_ensemble_member,
    tune_single_model,
)


def _ctx(is_classification: bool = True) -> PipelineContext:
    # accuracy(classification metric_class)를 쓴다 — ridge를 멤버로 쓰는 테스트가 있는데
    # RidgeClassifier는 predict_proba가 없어 auc(binary_proba)와는 호환 안 됨.
    return PipelineContext(
        target_col="y", metric="accuracy" if is_classification else "mae",
        n_splits=3, seed=42, is_classification=is_classification,
    )


def _make_df(n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 3))
    y = (x[:, 0] + x[:, 1] > 0).astype(float)
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y})


class _ModelSpecPatch:
    action_type = "model_swap"

    def model_spec(self, ctx):
        return {"model": "ridge", "params": {"alpha": 1.0}}


class _EnsembleSpecPatch:
    action_type = "ensemble"

    def ensemble_spec(self, ctx):
        return {"members": [{"model": "ridge", "params": {"alpha": 1.0}}, {"model": "random_forest", "params": {"n_estimators": 10}}]}


class _FreeformPatch:
    action_type = "model_swap"

    def build_model(self, params, ctx):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression()


def test_tune_single_model_returns_result_with_correct_shape():
    pipeline = PatchedPipeline(BasePipeline(), _ModelSpecPatch())
    ctx = _ctx()
    df = _make_df()
    result = tune_single_model(pipeline, df, ctx, "ridge", n_trials=3)
    assert isinstance(result, TunerResult)
    assert result.model_name == "ridge"
    assert result.member_index is None
    assert result.n_trials == 3
    assert np.isfinite(result.best_cv_score)
    assert np.isfinite(result.baseline_cv_score)
    assert "alpha" in result.best_params


def test_tune_single_model_baseline_matches_direct_eval():
    from evaluator.harness import evaluate_pipeline
    pipeline = PatchedPipeline(BasePipeline(), _ModelSpecPatch())
    ctx = _ctx()
    df = _make_df()
    direct = evaluate_pipeline(pipeline, df, ctx).cv_score
    result = tune_single_model(pipeline, df, ctx, "ridge", n_trials=2)
    assert result.baseline_cv_score == direct


def test_tune_ensemble_member_tunes_only_target_member():
    pipeline = PatchedPipeline(BasePipeline(), _EnsembleSpecPatch())
    ctx = _ctx()
    df = _make_df()
    result = tune_ensemble_member(pipeline, df, ctx, member_index=0, n_trials=3)
    assert result.model_name == "ridge"
    assert result.member_index == 0
    assert "alpha" in result.best_params


def test_tune_ensemble_member_out_of_range_raises():
    pipeline = PatchedPipeline(BasePipeline(), _EnsembleSpecPatch())
    ctx = _ctx()
    df = _make_df()
    with pytest.raises(ValueError, match="out of range"):
        tune_ensemble_member(pipeline, df, ctx, member_index=5, n_trials=1)


def test_tune_ensemble_member_no_spec_raises():
    pipeline = PatchedPipeline(BasePipeline(), _ModelSpecPatch())
    ctx = _ctx()
    df = _make_df()
    with pytest.raises(ValueError, match="no ensemble_spec"):
        tune_ensemble_member(pipeline, df, ctx, member_index=0, n_trials=1)


def test_tune_confirmed_pipeline_dispatches_to_model_spec():
    pipeline = PatchedPipeline(BasePipeline(), _ModelSpecPatch())
    ctx = _ctx()
    df = _make_df()
    results = tune_confirmed_pipeline(pipeline, df, ctx, n_trials=2)
    assert len(results) == 1
    assert results[0].member_index is None


def test_tune_confirmed_pipeline_dispatches_to_ensemble_members():
    pipeline = PatchedPipeline(BasePipeline(), _EnsembleSpecPatch())
    ctx = _ctx()
    df = _make_df()
    results = tune_confirmed_pipeline(pipeline, df, ctx, n_trials=2)
    assert len(results) == 2
    assert {r.member_index for r in results} == {0, 1}
    assert {r.model_name for r in results} == {"ridge", "random_forest"}


def test_tune_confirmed_pipeline_freeform_build_model_raises():
    pipeline = PatchedPipeline(BasePipeline(), _FreeformPatch())
    ctx = _ctx()
    df = _make_df()
    with pytest.raises(ValueError, match="not tunable"):
        tune_confirmed_pipeline(pipeline, df, ctx, n_trials=1)


# --- infer_registry_model (ADR-035 개정, #252) ---

_S5E4_LIKE = '''
class Patch:
    action_type = "hyperparam_search"
    def build_model(self, params, ctx):
        model = LGBMRegressor(random_state=ctx.seed, **params)
        return model
'''

_S6E8_LIKE = '''
import xgboost as xgb
class Patch:
    action_type = "model_swap"
    def build_model(self, params, ctx):
        base_params = {"objective": "binary:logistic", "n_estimators": 1500}
        base_params.update(params or {})
        model = xgb.XGBClassifier(**base_params)
        return model
'''

_CUSTOM_WRAPPER = '''
class Patch:
    def build_model(self, params, ctx):
        return _EnsembleRegressor(weight_lgb=params.get("weight_lgb", 0.5))
'''

_MULTI_MODEL = '''
class Patch:
    def build_model(self, params, ctx):
        m1 = LGBMClassifier(**params)
        m2 = CatBoostClassifier()
        return StackingClassifier([("a", m1), ("b", m2)])
'''

_PARAMS_IGNORED = '''
class Patch:
    def build_model(self, params, ctx):
        return LGBMRegressor(random_state=ctx.seed)
'''


@pytest.mark.parametrize("src,expected", [
    (_S5E4_LIKE, "lgbm"),
    (_S6E8_LIKE, "xgboost"),
    (_CUSTOM_WRAPPER, None),
    (_MULTI_MODEL, None),
    (_PARAMS_IGNORED, None),
    ("def not_a_patch(): pass", None),
])
def test_infer_registry_model(src, expected):
    assert infer_registry_model(src) == expected


def test_tune_confirmed_pipeline_infers_freeform_when_source_given():
    class _FreeformLgbm:
        action_type = "model_swap"

        def build_model(self, params, ctx):
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params)

    pipeline = PatchedPipeline(BasePipeline(), _FreeformLgbm())
    ctx = _ctx()
    df = _make_df()
    src = 'class Patch:\n    def build_model(self, params, ctx):\n        m = LGBMClassifier(**params)\n        return m\n'
    results = tune_confirmed_pipeline(pipeline, df, ctx, n_trials=2, pipeline_source=src)
    assert len(results) == 1
    assert results[0].model_name == "lgbm"
    assert results[0].member_index is None


def test_tune_single_model_unknown_registry_name_fails_fast():
    """등록 안 된 모델명은 trial 하나하나가 실패하는 게 아니라(catch=(Exception,)로
    조용히 묻히는 설정 오류가 아니라) 최초 호출에서 즉시 ValueError — trial을
    낭비하며 조용히 실패하는 대신 바로 드러나야 한다."""
    pipeline = PatchedPipeline(BasePipeline(), _ModelSpecPatch())
    ctx = _ctx()
    df = _make_df()
    with pytest.raises(ValueError, match="no search space"):
        tune_single_model(pipeline, df, ctx, "not_a_real_model", n_trials=3)


def test_to_result_falls_back_to_baseline_when_no_trials_completed():
    """study.optimize(catch=(Exception,))는 개별 trial 실패를 흡수해 FAIL 상태로
    남긴다 — 전부 실패해도 크래시 대신 baseline으로 안전하게 폴백해야 한다."""
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL, params={}, distributions={}))
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL, params={}, distributions={}))
    result = _to_result(study, "ridge", None, baseline_cv=0.5, ctx=_ctx())
    assert result.n_trials == 0
    assert result.improved is False
    assert result.best_cv_score == 0.5 == result.baseline_cv_score
