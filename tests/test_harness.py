from __future__ import annotations

import pytest
import numpy as np
import polars as pl

from config.settings import LABEL_Z
from evaluator.harness import (
    BasePipeline, PatchedPipeline, PipelineContext,
    evaluate_pipeline, preselect_params, is_significant_gain,
    split_audit_holdout,
    _LEAK_PERFECT_HIGH,
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


# --- is_significant_gain ---

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


def test_label_z_imported_from_settings():
    from config.settings import LABEL_Z as settings_LABEL_Z
    from evaluator.harness import LABEL_Z as harness_LABEL_Z
    assert settings_LABEL_Z == harness_LABEL_Z


# --- preselect ---

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

    holdout 마스킹이 없으면 holdout에서 leak==label로 생존해 near-perfect가 된다.
    """
    action_type = "feature_engineering"

    def feature_transform(self, train, valid, target, ctx):
        tr_out = train.with_columns(pl.col(target).alias("leak"))
        va_out = valid.with_columns(pl.col(target).alias("leak"))
        return tr_out, va_out


def test_eval_holdout_masks_target_derived_leakage():
    """_eval_holdout이 holdout 타깃을 마스킹해 파생 피처 누수를 차단함을 검증.

    수정 전(마스킹 없음)이면 holdout_score가 near-perfect(>0.9999)라 이 테스트가 실패한다.
    """
    df = _make_df_with_strings(n=300)
    ctx = _ctx()
    train90, holdout10 = split_audit_holdout(df, "y", is_classification=True)
    pipeline = PatchedPipeline(BasePipeline(), _HoldoutSelfTargetLeakPatch())
    score = _eval_holdout(pipeline, train90, holdout10, ctx)
    assert score is not None
    assert score < _LEAK_PERFECT_HIGH


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


# --- split_audit_holdout ---

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
