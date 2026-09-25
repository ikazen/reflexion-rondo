"""evaluator.tuner — Optuna 튜닝 레인(#230) 단위 테스트. n_trials를 작게 줘 빠르게 돈다."""
from __future__ import annotations

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
    _SingleModelTrialPipeline,
    _skip_if_baseline_incomparable,
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


def test_tune_confirmed_pipeline_freeform_build_model_raises():
    pipeline = PatchedPipeline(BasePipeline(), _FreeformPatch())
    ctx = _ctx()
    df = _make_df()
    with pytest.raises(ValueError, match="not tunable"):
        tune_confirmed_pipeline(pipeline, df, ctx, n_trials=1)


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
    with pytest.raises(ValueError, match="every ensemble member is 'base'"):
        tune_confirmed_pipeline(_OnlyBaseMembersPipeline(), None, None, n_trials=1)


class _BaseAndTwoMembersPipeline:
    def ensemble_spec(self, ctx):
        return {"members": [{"model": "base"}, {}, {}]}


def test_ensemble_run_budget_is_split_across_tunable_members_only(monkeypatch):
    seen = _fake_members(monkeypatch, _Clock(), spend=[0.0, 100.0, 50.0])
    results = tune_confirmed_pipeline(_BaseAndTwoMembersPipeline(), None, None, n_trials=5, timeout_sec=900)
    assert seen == [450.0, 800.0]  # 멤버 1: 900/2, 멤버 2: (900-100)/1 — base 멤버는 몫을 차지하지 않는다
    assert [r.member_index for r in results] == [1, 2]
