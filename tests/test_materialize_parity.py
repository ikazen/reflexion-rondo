"""materialize_best_pipeline과 evaluator.harness.PatchedPipeline의 ensemble_spec/model_spec 상속 억제 규칙이
같은 판정을 내리는지 검증한다(#374, ADR-059). 두 곳이 어긋나면 attempt 시점 평가와 merge-verify가 다른 모델을
채점해 patch의 실제 효과가 병합본에서 조용히 사라진다.
"""
from __future__ import annotations

import textwrap

import numpy as np
import polars as pl
import pytest

from cycle.materialize import materialize_best_pipeline
from evaluator.harness import BasePipeline, PatchedPipeline, PipelineContext, evaluate_pipeline


def _load_patch(source: str):
    ns: dict = {}
    exec(compile(source, "<patch>", "exec"), ns)  # noqa: S102
    return ns["Patch"]()


def _hooks(base_source: str | None, patch_source: str, ctx: PipelineContext) -> tuple[bool, bool]:
    """(ensemble_spec is None, model_spec is None) — attempt 시점(PatchedPipeline 체이닝)."""
    base = PatchedPipeline(BasePipeline(), _load_patch(base_source)) if base_source else BasePipeline()
    pipeline = PatchedPipeline(base, _load_patch(patch_source))
    return pipeline.ensemble_spec(ctx) is None, pipeline.model_spec(ctx) is None


def _merged_hooks(base_source: str | None, patch_source: str, ctx: PipelineContext) -> tuple[bool, bool]:
    """(ensemble_spec is None, model_spec is None) — materialize 병합본을 단일 patch로 평가."""
    merged = materialize_best_pipeline(base_source, patch_source)
    pipeline = PatchedPipeline(BasePipeline(), _load_patch(merged))
    return pipeline.ensemble_spec(ctx) is None, pipeline.model_spec(ctx) is None


def _ctx() -> PipelineContext:
    return PipelineContext(target_col="y", metric="mae", n_splits=3, seed=42, is_classification=False)


_BASE_ENSEMBLE_WITH_STALE_BUILD_MODEL = textwrap.dedent("""
    class Patch:
        action_type = "ensemble"
        changed_stages = ["ensemble_spec"]
        rationale = "stack"

        def param_candidates(self, ctx):
            return [{"alpha": 1.0}]

        # 실제 프로덕션 스냅샷처럼 예전 model_swap 패치의 build_model이 죽은 코드로 남아 있다 —
        # ensemble_spec이 정의돼 있으면 fit_predict가 이걸 부르지 않는다.
        def build_model(self, params, ctx):
            from sklearn.linear_model import Ridge
            return Ridge(alpha=99.0)

        def ensemble_spec(self, ctx):
            return {"members": [{"model": "ridge", "params": {"alpha": 1.0}}, {"model": "ridge", "params": {"alpha": 2.0}}]}
""").strip()

_PATCH_PARAM_CANDIDATES_ONLY = textwrap.dedent("""
    class Patch:
        action_type = "hyperparam_search"
        changed_stages = ["param_candidates"]
        rationale = "widen search"

        def param_candidates(self, ctx):
            return [{"alpha": 3.0}, {"alpha": 4.0}]
""").strip()

_BASE_MODEL_SPEC = textwrap.dedent("""
    class Patch:
        action_type = "model_swap"
        changed_stages = ["model_spec"]
        rationale = "ridge"

        def model_spec(self, ctx):
            return {"model": "ridge", "params": {"alpha": 1.0}}
""").strip()

_PATCH_BUILD_MODEL_ONLY = textwrap.dedent("""
    class Patch:
        action_type = "model_swap"
        changed_stages = ["build_model"]
        rationale = "custom lgbm wrapper"

        def build_model(self, params, ctx):
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=20, random_state=ctx.seed)
""").strip()

_PATCH_FEATURE_TRANSFORM_ONLY = textwrap.dedent("""
    class Patch:
        action_type = "feature_engineering"
        changed_stages = ["feature_transform"]
        rationale = "add a column"

        def feature_transform(self, train, valid, target, ctx):
            cols = [c for c in train.columns if c != target]
            return train.select(cols), valid.select(cols)
""").strip()

_PATCH_NEW_ENSEMBLE = textwrap.dedent("""
    class Patch:
        action_type = "ensemble"
        changed_stages = ["ensemble_spec"]
        rationale = "replace stack"

        def ensemble_spec(self, ctx):
            return {"members": [{"model": "ridge", "params": {"alpha": 5.0}}, {"model": "ridge", "params": {"alpha": 6.0}}]}
""").strip()


@pytest.mark.parametrize(("base_source", "patch_source", "expected"), [
    # (base 정의, patch 정의) -> (attempt-time과 merge 결과가 반드시 같아야 하는 (ensemble_none, model_none))
    # param_candidates만 정의하는 patch는 모델 선택 신호가 아니라 억제 대상에서 빠졌다(#387) —
    # base의 ensemble_spec을 그대로 상속하므로 ensemble_none=False.
    (_BASE_ENSEMBLE_WITH_STALE_BUILD_MODEL, _PATCH_PARAM_CANDIDATES_ONLY, (False, True)),
    (_BASE_MODEL_SPEC, _PATCH_BUILD_MODEL_ONLY, (True, True)),
    (_BASE_ENSEMBLE_WITH_STALE_BUILD_MODEL, _PATCH_FEATURE_TRANSFORM_ONLY, (False, True)),
    (_BASE_ENSEMBLE_WITH_STALE_BUILD_MODEL, _PATCH_NEW_ENSEMBLE, (False, True)),
    (_BASE_MODEL_SPEC, _PATCH_NEW_ENSEMBLE, (False, True)),
])
def test_merge_matches_attempt_time_suppression(base_source, patch_source, expected):
    ctx = _ctx()
    attempt_time = _hooks(base_source, patch_source, ctx)
    merged = _merged_hooks(base_source, patch_source, ctx)
    assert attempt_time == expected
    assert merged == expected


def _make_df(n: int = 80) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 2))
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": x[:, 0] * 2.0 + rng.standard_normal(n) * 1.0})


@pytest.mark.parametrize(("base_source", "patch_source"), [
    (_BASE_ENSEMBLE_WITH_STALE_BUILD_MODEL, _PATCH_PARAM_CANDIDATES_ONLY),
    (_BASE_MODEL_SPEC, _PATCH_BUILD_MODEL_ONLY),
    (_BASE_ENSEMBLE_WITH_STALE_BUILD_MODEL, _PATCH_FEATURE_TRANSFORM_ONLY),
])
def test_merge_cv_is_bit_identical_to_attempt_time(base_source, patch_source):
    """#374 실측 재현의 최소형: 병합본 재평가(merge-verify)가 attempt 시점 평가와 정확히 같은 모델을 채점해야 한다."""
    df = _make_df()
    ctx = _ctx()
    base = PatchedPipeline(BasePipeline(), _load_patch(base_source))
    attempt_pipeline = PatchedPipeline(base, _load_patch(patch_source))
    attempt_cv = evaluate_pipeline(attempt_pipeline, df, ctx).cv_score

    merged = materialize_best_pipeline(base_source, patch_source)
    merged_pipeline = PatchedPipeline(BasePipeline(), _load_patch(merged))
    merged_cv = evaluate_pipeline(merged_pipeline, df, ctx).cv_score

    assert merged_cv == attempt_cv


def test_suppressed_hook_is_absent_from_the_merged_source_text():
    """제거는 조용한 예외 처리가 아니라 실제로 병합 소스에서 훅이 사라져야 한다 — 남아있으면 재로드 시 되살아난다.

    build_model만 정의하는 patch가 진짜 억제 신호다(#387 이후 param_candidates는 아님, 아래 별도 테스트)."""
    merged = materialize_best_pipeline(_BASE_ENSEMBLE_WITH_STALE_BUILD_MODEL, _PATCH_BUILD_MODEL_ONLY)
    assert "def ensemble_spec" not in merged
    assert "def build_model" in merged  # patch의 단일 모델 의도가 쓰는 build_model은 그대로 남아야 한다


def test_param_candidates_only_hook_survives_in_the_merged_source_text():
    """#387: param_candidates만 정의하는 patch는 base의 ensemble_spec을 억제하지 않으므로
    병합 소스에도 그대로 남아야 한다 — 죽은 build_model(예전 model_swap 잔재)도 patch가
    안 건드렸으니 그대로 남는다."""
    merged = materialize_best_pipeline(_BASE_ENSEMBLE_WITH_STALE_BUILD_MODEL, _PATCH_PARAM_CANDIDATES_ONLY)
    assert "def ensemble_spec" in merged
    assert "def build_model" in merged


def test_none_base_has_nothing_to_suppress():
    """base가 없으면(bootstrap) 지울 상속이 없다 — 정상 동작 확인용 경계 케이스."""
    merged = materialize_best_pipeline(None, _PATCH_NEW_ENSEMBLE)
    pipeline = PatchedPipeline(BasePipeline(), _load_patch(merged))
    assert pipeline.ensemble_spec(_ctx()) is not None
