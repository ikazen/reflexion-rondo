from __future__ import annotations

import pytest
import numpy as np
import polars as pl

from evaluator.harness import (
    BasePipeline, PatchedPipeline, PipelineContext,
    evaluate_pipeline, preselect_params,
    _LEAK_PERFECT_HIGH,
)


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


# --- target leakage guard tests ---

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
