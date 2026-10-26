"""evaluator.tuner — Optuna 튜닝 레인(#230) 단위 테스트. n_trials를 작게 줘 빠르게 돈다."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import optuna
import polars as pl
import pytest

import evaluator.tuner as tuner_module
from evaluator.harness import BasePipeline, PatchedPipeline, PipelineContext, evaluate_pipeline
from evaluator.tuner import (
    TunerResult,
    _extract_base_params_literal,
    _optimize,
    _ParamCandidateTrialPipeline,
    _SingleModelTrialPipeline,
    _skip_if_baseline_incomparable,
    _to_result,
    infer_registry_model,
    infer_wrapped_registry_model,
    tune_confirmed_pipeline,
    tune_ensemble_member,
    tune_single_model,
    tune_wrapped_model,
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


def test_seeded_baseline_uses_same_registry_path_as_trials_not_raw_freeform():
    """#341: 자유형 build_model이 레지스트리 생성자와 다르게 동작하면(여기선 alpha를
    5배 증폭하는 커스텀 로직) baseline_cv_score는 trial과 동일한 레지스트리 경로
    (_SingleModelTrialPipeline)로 평가해야 한다 — 원본 freeform 경로 그대로 평가하면
    params가 같아도 다른 모델을 비교하게 돼 improved 판정이 구조적으로 왜곡된다."""
    class _FreeformRidgeAmplifiesAlpha:
        action_type = "model_swap"

        def build_model(self, params, ctx):
            from sklearn.linear_model import Ridge
            alpha = (params or {}).get("alpha", 1.0) * 5.0
            return Ridge(alpha=alpha)

    pipeline = PatchedPipeline(BasePipeline(), _FreeformRidgeAmplifiesAlpha())
    ctx = _ctx(is_classification=False)
    df = _make_df()
    seed_params = {"alpha": 1.0}

    raw_freeform_cv = evaluate_pipeline(pipeline, df, ctx).cv_score
    wrapped_cv = evaluate_pipeline(
        _SingleModelTrialPipeline(pipeline, "ridge", seed_params), df, ctx
    ).cv_score
    assert raw_freeform_cv != wrapped_cv  # 전제: 두 경로가 실제로 다른 모델임을 확인

    result = tune_single_model(pipeline, df, ctx, "ridge", n_trials=1, seed_params=seed_params)
    assert result.baseline_cv_score == wrapped_cv
    assert result.baseline_cv_score != raw_freeform_cv


def test_unseeded_baseline_falls_back_to_raw_pipeline():
    """seed_params가 없으면(알려진 baseline params 자체가 없는 극단적 경우) 동등한
    wrapper를 만들 수 없으니 원본 pipeline 직접 평가로 폴백한다."""
    from evaluator.harness import evaluate_pipeline as _eval
    pipeline = PatchedPipeline(BasePipeline(), _ModelSpecPatch())
    ctx = _ctx()
    df = _make_df()
    direct = _eval(pipeline, df, ctx).cv_score
    result = tune_single_model(pipeline, df, ctx, "ridge", n_trials=1, seed_params=None)
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


def test_tune_confirmed_pipeline_freeform_build_model_has_nothing_to_tune():
    pipeline = PatchedPipeline(BasePipeline(), _FreeformPatch())
    ctx = _ctx()
    df = _make_df()
    assert tune_confirmed_pipeline(pipeline, df, ctx, n_trials=1) == []


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


class _ElasticNetModelSpecPatch:
    action_type = "model_swap"

    def model_spec(self, ctx):
        # baseline eval(트리얼과 무관, tune_single_model이 seed_params 없이 이 patch를 직접
        # 평가)도 정상적으로 elastic-net이 걸리려면 penalty/solver를 여기서도 명시해야
        # 한다 — search_spaces.py의 _elastic_net_space 자체 docstring과 같은 이유.
        return {
            "model": "elastic_net",
            "params": {"C": 1.0, "l1_ratio": 0.5, "penalty": "elasticnet", "solver": "saga", "max_iter": 2000},
        }


def test_tune_single_model_best_params_includes_fixed_search_space_keys():
    """#392: elastic_net search space는 penalty="elasticnet"/solver="saga"/max_iter=2000을
    suggest 없이 고정값으로 넣는다 — study.best_params(optuna 자신의 suggest 기록)만 쓰면 이
    키들이 빠져, 나중에 best_params로 모델을 재구성하면 sklearn 기본값(penalty="l2",
    solver="lbfgs")으로 완전히 다른 모델이 된다."""
    pipeline = PatchedPipeline(BasePipeline(), _ElasticNetModelSpecPatch())
    ctx = _ctx()
    df = _make_df()
    result = tune_single_model(pipeline, df, ctx, "elastic_net", n_trials=3)
    assert result.best_params.get("penalty") == "elasticnet"
    assert result.best_params.get("solver") == "saga"
    assert result.best_params.get("max_iter") == 2000
    assert "C" in result.best_params and "l1_ratio" in result.best_params


def test_tune_single_model_trials_keep_seed_keys_outside_the_search_space():
    """#554: s6e4 base의 class_weight='balanced'처럼 탐색공간에 없는 seed 키가 trial에서 빠지면 baseline과 다른 모델을 비교한다."""
    class _RfClassWeightPatch:
        action_type = "model_swap"

        def model_spec(self, ctx):
            return {"model": "random_forest", "params": {"n_estimators": 10, "class_weight": "balanced"}}

    pipeline = PatchedPipeline(BasePipeline(), _RfClassWeightPatch())
    result = tune_confirmed_pipeline(pipeline, _make_df(), _ctx(), n_trials=3)[0]
    assert result.best_params["class_weight"] == "balanced"
    assert "max_depth" in result.best_params


def test_tune_ensemble_member_trials_keep_seed_keys_outside_the_search_space():
    class _EnsembleClassWeightPatch:
        action_type = "ensemble"

        def ensemble_spec(self, ctx):
            return {"members": [
                {"model": "ridge", "params": {"alpha": 1.0}},
                {"model": "random_forest", "params": {"n_estimators": 10, "class_weight": "balanced"}},
            ]}

    pipeline = PatchedPipeline(BasePipeline(), _EnsembleClassWeightPatch())
    result = tune_ensemble_member(pipeline, _make_df(), _ctx(), member_index=1, n_trials=3)
    assert result.best_params["class_weight"] == "balanced"


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


# #331 — Optuna 튜너가 탐색공간 밖 baseline과 비교되는 문제 (search-space seeding)

_S6E8_LIKE_WITH_NONLITERAL_SEED = '''
import xgboost as xgb
class Patch:
    action_type = "model_swap"
    def build_model(self, params, ctx):
        base_params = {"objective": "binary:logistic", "random_state": ctx.seed, "n_estimators": 1500}
        base_params.update(params or {})
        model = xgb.XGBClassifier(**base_params)
        return model
'''

_FREEFORM_RIDGE_LITERAL = '''
class Patch:
    action_type = "model_swap"
    def build_model(self, params, ctx):
        from sklearn.linear_model import Ridge
        base_params = {"alpha": 5.0}
        base_params.update(params or {})
        model = Ridge(**base_params)
        return model
'''


def test_extract_base_params_literal_skips_nonliteral_values_only():
    """딕셔너리 안 값 하나(random_state: ctx.seed)가 비리터럴이어도 나머지 리터럴
    키는 살아남는다 — 전체를 ast.literal_eval하면 여기서 전부 유실된다(s6e8 실측 형태)."""
    extracted = _extract_base_params_literal(_S6E8_LIKE_WITH_NONLITERAL_SEED)
    assert extracted == {"objective": "binary:logistic", "n_estimators": 1500}
    assert "random_state" not in extracted


def test_extract_base_params_literal_no_dict_literal_returns_empty():
    assert _extract_base_params_literal(_S5E4_LIKE) == {}


def test_extract_base_params_literal_not_a_patch_returns_empty():
    assert _extract_base_params_literal("def not_a_patch(): pass") == {}


def test_extract_base_params_literal_syntax_error_returns_empty():
    assert _extract_base_params_literal("class Patch:\n  def build_model(") == {}


def test_optimize_enqueues_seed_trial_when_given():
    def objective(trial: "optuna.Trial") -> float:
        return trial.suggest_int("x", 1, 10)

    study = _optimize(objective, n_trials=1, timeout_sec=None, direction="minimize", seed_params={"x": 7})
    assert study.trials[0].params["x"] == 7


def test_optimize_no_extra_trial_when_seed_params_none():
    def objective(trial: "optuna.Trial") -> float:
        return trial.suggest_int("x", 1, 10)

    study = _optimize(objective, n_trials=2, timeout_sec=None, direction="minimize")
    assert len(study.trials) == 2


def test_optimize_seed_out_of_range_used_verbatim_not_clipped():
    """탐색범위 밖 seed(예: n_estimators=1500 vs suggest 상한 1000)도 UserWarning만
    내고 그대로 쓴다 — clip이나 예외 없음(2026-09 optuna 4.x 실측, #331)."""
    def objective(trial: "optuna.Trial") -> float:
        return trial.suggest_int("n_estimators", 100, 1000)

    study = _optimize(
        objective, n_trials=1, timeout_sec=None, direction="minimize",
        seed_params={"n_estimators": 1500},
    )
    assert study.trials[0].params["n_estimators"] == 1500


def test_optimize_seed_extra_keys_ignored_silently():
    """objective가 요구 안 하는 키(예: objective/eval_metric)는 조용히 무시된다."""
    def objective(trial: "optuna.Trial") -> float:
        return trial.suggest_int("x", 1, 10)

    study = _optimize(
        objective, n_trials=1, timeout_sec=None, direction="minimize",
        seed_params={"x": 3, "objective": "binary:logistic", "eval_metric": "auc"},
    )
    assert study.trials[0].params == {"x": 3}


def test_tune_confirmed_pipeline_freeform_seeds_first_trial_from_literal():
    """추론된 자유형 경로도 model_spec 경로와 동일하게 baseline params를 시드한다 —
    n_trials=1이면 유일한 trial이 곧 enqueue된 시드 trial이라 best_params로 직접 검증."""
    class _FreeformRidge:
        action_type = "model_swap"

        def build_model(self, params, ctx):
            from sklearn.linear_model import Ridge
            base_params = {"alpha": 5.0}
            base_params.update(params or {})
            return Ridge(**base_params)

    pipeline = PatchedPipeline(BasePipeline(), _FreeformRidge())
    ctx = _ctx(is_classification=False)
    df = _make_df()
    results = tune_confirmed_pipeline(
        pipeline, df, ctx, n_trials=1, pipeline_source=_FREEFORM_RIDGE_LITERAL,
    )
    assert results[0].best_params["alpha"] == 5.0


def test_optimize_seed_aliases_are_mapped_to_the_search_space_keys():
    """확정 pipeline이 LightGBM native 별칭(bagging_fraction 등)으로 쓴 값도 seed trial에 반영된다(#488)."""
    def objective(trial: "optuna.Trial") -> float:
        return trial.suggest_float("subsample", 0.5, 1.0) + trial.suggest_float("colsample_bytree", 0.5, 1.0)

    study = _optimize(
        objective, n_trials=1, timeout_sec=None, direction="minimize",
        seed_params={"bagging_fraction": 0.6, "bagging_freq": 1, "feature_fraction": 0.5, "max_depth": 32},
    )
    assert study.trials[0].params == {"subsample": 0.6, "colsample_bytree": 0.5}


def test_optimize_seed_disables_bagging_when_the_base_never_enabled_it():
    """bagging_freq가 없는 확정 pipeline의 bagging_fraction은 LightGBM이 무시한다 — 탐색공간은 bagging_freq=1 고정이라 subsample=1.0으로 옮겨야 같은 모델이다(#488)."""
    def objective(trial: "optuna.Trial") -> float:
        return trial.suggest_float("subsample", 0.5, 1.0)

    off = _optimize(
        objective, n_trials=1, timeout_sec=None, direction="minimize", seed_params={"bagging_fraction": 0.6},
    )
    on = _optimize(
        objective, n_trials=1, timeout_sec=None, direction="minimize",
        seed_params={"bagging_fraction": 0.6, "bagging_freq": 5},
    )
    assert off.trials[0].params == {"subsample": 1.0}
    assert on.trials[0].params == {"subsample": 0.6}


def test_tune_confirmed_pipeline_freeform_seeds_from_a_single_param_candidate():
    """동결 base는 params를 build_model 리터럴이 아니라 param_candidates 한 개로 갖는다 — 리터럴이 비어도 그 dict가 seed다(#488)."""
    class _FrozenRidge:
        action_type = "model_swap"

        def param_candidates(self, ctx):
            return [{"alpha": 7.0}]

        def build_model(self, params, ctx):
            from sklearn.linear_model import Ridge
            return Ridge(**params)

    source = "class Patch:\n    def build_model(self, params, ctx):\n        from sklearn.linear_model import Ridge\n        return Ridge(**params)\n"
    pipeline = PatchedPipeline(BasePipeline(), _FrozenRidge())
    results = tune_confirmed_pipeline(
        pipeline, _make_df(), _ctx(is_classification=False), n_trials=1, pipeline_source=source,
    )
    assert results[0].best_params["alpha"] == 7.0


def test_tune_single_model_seeds_from_model_spec_params():
    """model_spec 경로는 별도 추출 없이 spec의 params를 그대로 seed로 쓴다."""
    pipeline = PatchedPipeline(BasePipeline(), _ModelSpecPatch())
    ctx = _ctx()
    df = _make_df()
    results = tune_confirmed_pipeline(pipeline, df, ctx, n_trials=1)
    assert results[0].best_params["alpha"] == 1.0


def test_tune_ensemble_member_seeds_from_member_params():
    """ensemble_spec 경로는 base_spec의 해당 멤버 params를 그대로 seed로 쓴다."""
    pipeline = PatchedPipeline(BasePipeline(), _EnsembleSpecPatch())
    ctx = _ctx()
    df = _make_df()
    result = tune_ensemble_member(pipeline, df, ctx, member_index=1, n_trials=1)
    assert result.best_params["n_estimators"] == 10


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _ThreeMemberPipeline:
    def ensemble_spec(self, ctx):
        return {"members": [{}, {}, {}]}


def _fake_members(monkeypatch, clock: _Clock, spend: list[float]) -> list[float | None]:
    monkeypatch.setattr(tuner_module, "time", SimpleNamespace(monotonic=clock))
    seen: list[float | None] = []

    def fake_member(pipeline, train, ctx, i, n_trials, timeout_sec, expected_baseline_cv=None):
        seen.append(None if timeout_sec is None else round(timeout_sec, 1))
        clock.now += spend[i]
        return TunerResult(
            model_name="lgbm", member_index=i, best_params={}, best_cv_score=0.0, baseline_cv_score=0.0,
            n_trials=1, improved=False,
        )

    monkeypatch.setattr(tuner_module, "tune_ensemble_member", fake_member)
    return seen


def test_ensemble_run_budget_is_split_across_remaining_members(monkeypatch):
    """timeout_sec는 멤버별 상한이 아니라 런 전체 예산 — 앞 멤버가 쓴 시간을 뺀 나머지를 남은 멤버 수로 나눈다(#350)."""
    seen = _fake_members(monkeypatch, _Clock(), spend=[100.0, 300.0, 50.0])
    results = tune_confirmed_pipeline(_ThreeMemberPipeline(), None, None, n_trials=5, timeout_sec=900)
    assert seen == [300.0, 400.0, 500.0]
    assert len(results) == 3


def test_ensemble_run_skips_remaining_members_when_budget_is_spent(monkeypatch):
    seen = _fake_members(monkeypatch, _Clock(), spend=[1000.0, 1.0, 1.0])
    results = tune_confirmed_pipeline(_ThreeMemberPipeline(), None, None, n_trials=5, timeout_sec=900)
    assert seen == [300.0]
    assert [r.member_index for r in results] == [0]


def test_ensemble_run_without_timeout_passes_no_budget_to_members(monkeypatch):
    seen = _fake_members(monkeypatch, _Clock(), spend=[100.0, 100.0, 100.0])
    tune_confirmed_pipeline(_ThreeMemberPipeline(), None, None, n_trials=5, timeout_sec=None)
    assert seen == [None, None, None]


def _single_model_study_budget(monkeypatch, baseline_seconds: float, timeout_sec: float | None) -> float | None:
    clock = _Clock()
    monkeypatch.setattr(tuner_module, "time", SimpleNamespace(monotonic=clock))

    def fake_evaluate(pipeline, train, ctx):
        clock.now += baseline_seconds
        return SimpleNamespace(cv_score=0.5)

    captured: dict = {}

    def fake_optimize(objective, n_trials, timeout, direction, seed_params=None):
        captured["timeout"] = timeout
        return SimpleNamespace(trials=[])

    monkeypatch.setattr(tuner_module, "evaluate_pipeline", fake_evaluate)
    monkeypatch.setattr(tuner_module, "_optimize", fake_optimize)
    tune_single_model(object(), None, _ctx(), "lgbm", n_trials=1, timeout_sec=timeout_sec)
    return captured["timeout"]


def test_single_model_study_budget_excludes_baseline_evaluation_time(monkeypatch):
    assert _single_model_study_budget(monkeypatch, baseline_seconds=200.0, timeout_sec=900) == 700.0


def test_single_model_study_budget_keeps_one_second_for_the_seed_trial(monkeypatch):
    assert _single_model_study_budget(monkeypatch, baseline_seconds=5000.0, timeout_sec=900) == 1.0


def test_single_model_study_budget_none_stays_unbounded(monkeypatch):
    assert _single_model_study_budget(monkeypatch, baseline_seconds=200.0, timeout_sec=None) is None


def _fail_if_searched(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("baseline이 비교 불가능한데 탐색이 실행됐다")

    monkeypatch.setattr(tuner_module, "_optimize", boom)


def test_single_model_skips_search_when_baseline_differs_from_confirmed_cv(monkeypatch):
    """#360: tuner baseline이 확정 pipeline cv와 다르면 탐색하지 않고 n_trials=0 결과(행 기록용)를 돌려준다."""
    pipeline = PatchedPipeline(BasePipeline(), _ModelSpecPatch())
    ctx = _ctx()
    df = _make_df()
    direct = evaluate_pipeline(pipeline, df, ctx).cv_score
    _fail_if_searched(monkeypatch)
    result = tune_single_model(pipeline, df, ctx, "ridge", n_trials=3, expected_baseline_cv=direct + 0.01)
    assert result == TunerResult(
        model_name="ridge", member_index=None, best_params={}, best_cv_score=direct, baseline_cv_score=direct,
        n_trials=0, improved=False,
    )


def test_single_model_searches_when_baseline_matches_confirmed_cv():
    pipeline = PatchedPipeline(BasePipeline(), _ModelSpecPatch())
    ctx = _ctx()
    df = _make_df()
    direct = evaluate_pipeline(pipeline, df, ctx).cv_score
    result = tune_single_model(pipeline, df, ctx, "ridge", n_trials=2, expected_baseline_cv=direct)
    assert result.n_trials == 2


def test_baseline_gate_tolerance_boundary(monkeypatch):
    pipeline = PatchedPipeline(BasePipeline(), _ModelSpecPatch())
    ctx = _ctx()
    df = _make_df()
    direct = evaluate_pipeline(pipeline, df, ctx).cv_score
    assert tune_single_model(pipeline, df, ctx, "ridge", n_trials=1, expected_baseline_cv=direct + 9e-7).n_trials == 1
    _fail_if_searched(monkeypatch)
    assert tune_single_model(pipeline, df, ctx, "ridge", n_trials=1, expected_baseline_cv=direct + 2e-6).n_trials == 0


def test_ensemble_member_skips_search_when_baseline_differs_from_confirmed_cv(monkeypatch):
    pipeline = PatchedPipeline(BasePipeline(), _EnsembleSpecPatch())
    ctx = _ctx()
    df = _make_df()
    direct = evaluate_pipeline(pipeline, df, ctx).cv_score
    _fail_if_searched(monkeypatch)
    result = tune_ensemble_member(pipeline, df, ctx, member_index=1, n_trials=3, expected_baseline_cv=direct - 0.02)
    assert result == TunerResult(
        model_name="random_forest", member_index=1, best_params={}, best_cv_score=direct, baseline_cv_score=direct,
        n_trials=0, improved=False,
    )


def test_gate_is_off_without_expected_baseline_cv():
    """expected_baseline_cv를 안 주는 호출(기존 호출부·수동 실행)은 이전과 동일하게 탐색한다."""
    pipeline = PatchedPipeline(BasePipeline(), _ModelSpecPatch())
    assert tune_single_model(pipeline, _make_df(), _ctx(), "ridge", n_trials=2).n_trials == 2


def test_confirmed_pipeline_forwards_expected_baseline_cv_on_every_path(monkeypatch):
    calls: list[tuple[str, float | None]] = []

    def fake_single(pipeline, train, ctx, model_name, **kwargs):
        calls.append(("single:" + model_name, kwargs["expected_baseline_cv"]))
        return TunerResult(model_name, None, {}, 0.0, 0.0, 0, False)

    def fake_member(pipeline, train, ctx, i, **kwargs):
        calls.append((f"member:{i}", kwargs["expected_baseline_cv"]))
        return TunerResult("ridge", i, {}, 0.0, 0.0, 0, False)

    monkeypatch.setattr(tuner_module, "tune_single_model", fake_single)
    monkeypatch.setattr(tuner_module, "tune_ensemble_member", fake_member)
    ctx = _ctx()

    tune_confirmed_pipeline(PatchedPipeline(BasePipeline(), _ModelSpecPatch()), None, ctx, expected_baseline_cv=0.5)
    tune_confirmed_pipeline(PatchedPipeline(BasePipeline(), _EnsembleSpecPatch()), None, ctx, expected_baseline_cv=0.5)
    src = "class Patch:\n    def build_model(self, params, ctx):\n        return LGBMClassifier(**params)\n"
    tune_confirmed_pipeline(
        PatchedPipeline(BasePipeline(), _FreeformPatch()), None, ctx, pipeline_source=src, expected_baseline_cv=0.5,
    )
    assert calls == [
        ("single:ridge", 0.5), ("member:0", 0.5), ("member:1", 0.5), ("single:lgbm", 0.5),
    ]


@pytest.mark.parametrize(("baseline", "expected", "skipped"), [
    (13.046045137, 13.046045327, False),  # s5e4 실측: 같은 pipeline의 baseline 흔들림(상대 1.5e-8)
    (13.046045327 + 5e-6, 13.046045327, False),  # 절대 오차는 1e-6 초과지만 상대 4e-7 — 큰 스케일 지표
    (13.046045327, 13.0373, True),        # 다른 pipeline
    (0.964329682, 0.966210409, True),     # s6e8 실측: 레지스트리 경로 vs 자유형 build_model
    (0.5 + 9e-7, 0.5, False),
    (0.5 + 2e-6, 0.5, True),
])
def test_baseline_tolerance_scales_with_metric_magnitude(baseline, expected, skipped):
    result = _skip_if_baseline_incomparable("xgboost", None, baseline, expected)
    assert (result is not None) is skipped
    if skipped:
        assert result.n_trials == 0 and result.improved is False and result.baseline_cv_score == baseline


class _BaseModelPatch:
    action_type = "model_swap"

    def param_candidates(self, ctx):
        return [{"alpha": 1.0}]

    def build_model(self, params, ctx):
        from sklearn.linear_model import Ridge

        return Ridge(**params)


class _EnsembleWithBasePatch:
    action_type = "ensemble"

    def ensemble_spec(self, ctx):
        return {
            "members": [
                {"model": "base"},
                {"model": "ridge", "params": {"alpha": 1.0}},
                {"model": "random_forest", "params": {"n_estimators": 10}},
            ],
            "method": "weighted_average",
        }


def _ensemble_with_base_pipeline():
    return PatchedPipeline(PatchedPipeline(BasePipeline(), _BaseModelPatch()), _EnsembleWithBasePatch())


def test_tune_confirmed_pipeline_skips_base_members():
    """#362: 예약어 멤버 base는 탐색 공간이 없어 튜닝 대상에서 뺀다 — 나머지 멤버는 그대로 튜닝한다."""
    results = tune_confirmed_pipeline(_ensemble_with_base_pipeline(), _make_df(), _ctx(is_classification=False), n_trials=1)
    assert [r.member_index for r in results] == [1, 2]
    assert [r.model_name for r in results] == ["ridge", "random_forest"]


def test_ensemble_member_trials_resolve_the_base_member_like_the_confirmed_pipeline():
    ctx = _ctx(is_classification=False)
    pipeline = _ensemble_with_base_pipeline()
    df = _make_df()
    baseline = evaluate_pipeline(pipeline, df, ctx).cv_score
    result = tune_ensemble_member(pipeline, df, ctx, member_index=1, n_trials=1)
    assert result.baseline_cv_score == baseline
    assert result.n_trials == 1


def test_tune_ensemble_member_rejects_the_base_member():
    with pytest.raises(ValueError, match="member 0 is 'base'"):
        tune_ensemble_member(_ensemble_with_base_pipeline(), _make_df(), _ctx(is_classification=False), member_index=0)


class _OnlyBaseMembersPipeline:
    def ensemble_spec(self, ctx):
        return {"members": [{"model": "base"}, {"model": "base"}]}


def test_ensemble_of_only_base_members_has_nothing_to_tune():
    assert tune_confirmed_pipeline(_OnlyBaseMembersPipeline(), None, None, n_trials=1) == []


class _BaseAndTwoMembersPipeline:
    def ensemble_spec(self, ctx):
        return {"members": [{"model": "base"}, {}, {}]}


def test_ensemble_run_budget_is_split_across_tunable_members_only(monkeypatch):
    seen = _fake_members(monkeypatch, _Clock(), spend=[0.0, 100.0, 50.0])
    results = tune_confirmed_pipeline(_BaseAndTwoMembersPipeline(), None, None, n_trials=5, timeout_sec=900)
    assert seen == [450.0, 800.0]  # 멤버 1: 900/2, 멤버 2: (900-100)/1 — base 멤버는 몫을 차지하지 않는다
    assert [r.member_index for r in results] == [1, 2]


_XGB_WRAPPER = '''
class _Wrap:
    def __init__(self, params, ctx):
        self._m = XGBClassifier(**params)

class Patch:
    def build_model(self, params, ctx):
        return _Wrap(params, ctx)
'''


@pytest.mark.parametrize("src,expected", [
    (_XGB_WRAPPER, "xgboost"),
    (_S6E8_LIKE, "xgboost"),
    (_CUSTOM_WRAPPER, None),
    (_MULTI_MODEL, None),
    (_PARAMS_IGNORED, None),
    ("def not_a_patch(): pass", None),
    ("class Patch:\n    def build_model(self, params, ctx):\n        return _Wrap(", None),
])
def test_infer_wrapped_registry_model(src, expected):
    assert infer_wrapped_registry_model(src) == expected


class _WrappedLgbm:
    """build_model이 params를 래퍼에 넘기는 형태. param_candidates는 확정 base처럼 한 개다."""
    action_type = "hyperparam_search"

    def param_candidates(self, ctx):
        return [{"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3, "verbosity": -1}]

    def build_model(self, params, ctx):
        from lightgbm import LGBMClassifier

        class _Wrap:
            def __init__(self):
                self._m = LGBMClassifier(**params)

            def fit(self, X, y):
                self._m.fit(X, y)
                return self

            def predict(self, X):
                return self._m.predict(X)

        return _Wrap()


def test_param_candidate_trial_pipeline_overrides_only_param_candidates():
    base = PatchedPipeline(BasePipeline(), _WrappedLgbm())
    trial = _ParamCandidateTrialPipeline(base, {"n_estimators": 5})
    ctx = _ctx()
    assert trial.param_candidates(ctx) == [{"n_estimators": 5}]
    assert trial.model_spec(ctx) is None and trial.ensemble_spec(ctx) is None
    assert trial.build_model({"n_estimators": 5}, ctx)._m.get_params()["n_estimators"] == 5


def test_max_folds_limits_evaluation_to_the_first_folds():
    pipeline = PatchedPipeline(BasePipeline(), _WrappedLgbm())
    ctx = _ctx()
    full = evaluate_pipeline(pipeline, _make_df(), ctx)
    first = evaluate_pipeline(pipeline, _make_df(), replace(ctx, max_folds=1))
    assert len(full.fold_scores) == 3 and len(first.fold_scores) == 1
    assert first.fold_scores[0] == full.fold_scores[0]
    assert first.cv_score == first.fold_scores[0]


def test_tune_wrapped_model_runs_the_pipelines_own_build_model():
    pipeline = PatchedPipeline(BasePipeline(), _WrappedLgbm())
    result = tune_wrapped_model(pipeline, _make_df(), _ctx(), "lgbm", pipeline.param_candidates(_ctx())[0], n_trials=4)
    assert result.model_name == "lgbm" and result.member_index is None
    assert result.n_trials >= 1
    assert result.improved == bool(result.best_params)
    if result.improved:
        assert result.best_cv_score > result.baseline_cv_score
        assert result.best_params["verbosity"] == -1


_WRAPPED_SRC = (
    "class _Wrap:\n    def __init__(self, params, ctx):\n        self._m = LGBMClassifier(**params)\n"
    "class Patch:\n    def build_model(self, params, ctx):\n        return _Wrap(params, ctx)\n"
)


def test_tune_confirmed_pipeline_dispatches_wrapped_build_model():
    pipeline = PatchedPipeline(BasePipeline(), _WrappedLgbm())
    results = tune_confirmed_pipeline(pipeline, _make_df(), _ctx(), n_trials=2, pipeline_source=_WRAPPED_SRC)
    assert [r.model_name for r in results] == ["lgbm"]


def test_tune_confirmed_pipeline_wrapped_without_a_single_candidate_is_untunable():
    class _TwoCandidates(_WrappedLgbm):
        def param_candidates(self, ctx):
            return [{"n_estimators": 5}, {"n_estimators": 9}]

    pipeline = PatchedPipeline(BasePipeline(), _TwoCandidates())
    assert tune_confirmed_pipeline(pipeline, _make_df(), _ctx(), n_trials=1, pipeline_source=_WRAPPED_SRC) == []


def _fake_wrapped_run(monkeypatch, fold1: dict[int, float], full: dict[int, float], baseline: float = 0.5, secs: float = 100.0):
    """trial 번호 -> fold-1 점수/전체 CV. trial 0이 seed다. 반환: (budget 기록, 전체 CV로 재확인한 params 목록)."""
    clock = _Clock()
    monkeypatch.setattr(tuner_module, "time", SimpleNamespace(monotonic=clock))
    seed = {"max_depth": 6, "extra": 1}
    trials = [
        SimpleNamespace(
            state=optuna.trial.TrialState.COMPLETE, value=v, user_attrs={"full_params": seed if i == 0 else {**seed, "max_depth": i}},
        )
        for i, v in sorted(fold1.items())
    ]
    seen: dict = {"confirmed": []}

    def fake_evaluate(pipe, train, ctx):
        clock.now += secs
        if isinstance(pipe, _ParamCandidateTrialPipeline):
            seen["confirmed"].append(pipe.param_candidates(ctx)[0]["max_depth"])
            return SimpleNamespace(cv_score=full[pipe.param_candidates(ctx)[0]["max_depth"]])
        return SimpleNamespace(cv_score=baseline)

    def fake_optimize(objective, n_trials, timeout, direction, seed_params=None):
        seen["budget"] = timeout
        return SimpleNamespace(trials=trials)

    monkeypatch.setattr(tuner_module, "evaluate_pipeline", fake_evaluate)
    monkeypatch.setattr(tuner_module, "_optimize", fake_optimize)
    result = tune_wrapped_model(object(), None, _ctx(), "xgboost", seed, n_trials=5, timeout_sec=1000)
    return result, seen


def test_wrapped_study_budget_reserves_the_confirmation_evaluations(monkeypatch):
    _, seen = _fake_wrapped_run(monkeypatch, {0: 0.5, 1: 0.6}, {1: 0.55})
    assert seen["budget"] == 1000 - 100 - 2 * 100


def test_wrapped_only_trials_above_the_seed_fold1_are_confirmed(monkeypatch):
    result, seen = _fake_wrapped_run(monkeypatch, {0: 0.5, 1: 0.7, 2: 0.6, 3: 0.4, 4: 0.55}, {1: 0.52, 2: 0.51, 4: 0.9})
    assert seen["confirmed"] == [1, 2]
    assert result.best_params["max_depth"] == 1 and result.best_cv_score == 0.52
    assert result.improved and result.n_trials == 5


def test_wrapped_not_improved_when_no_confirmed_candidate_beats_the_baseline(monkeypatch):
    result, _ = _fake_wrapped_run(monkeypatch, {0: 0.5, 1: 0.7}, {1: 0.49})
    assert not result.improved and result.best_params == {} and result.best_cv_score == result.baseline_cv_score


def test_wrapped_skips_confirmation_when_the_seed_has_the_best_fold1(monkeypatch):
    result, seen = _fake_wrapped_run(monkeypatch, {0: 0.9, 1: 0.7, 2: 0.6}, {})
    assert seen["confirmed"] == [] and not result.improved
