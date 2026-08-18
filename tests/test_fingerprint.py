"""store.fingerprint.compute의 데이터셋 지문(키 완전성, task_type별 통계) 단위 테스트."""
import numpy as np
import polars as pl
import pytest

from store.fingerprint import compute

_EXPECTED_KEYS = {
    "n_rows", "n_cols", "task_type", "metric", "metric_sign",
    "n_numeric", "n_categorical", "n_datetime", "n_text_ish",
    "missing_ratio_overall", "cardinality_max", "cardinality_mean",
    "target_stat", "size_class",
}


def _binary_df(n: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 3))
    y = (x[:, 0] > 0).astype(int)
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y})


def _regression_df(n: int = 200, skew: str = "right") -> pl.DataFrame:
    rng = np.random.default_rng(1 if skew == "right" else 2)
    x = rng.standard_normal((n, 2))
    if skew == "right":
        y = np.exp(rng.standard_normal(n))  # right-skewed
    else:
        y = -np.exp(rng.standard_normal(n))  # left-skewed
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


def test_binary_returns_all_keys():
    fp = compute(_binary_df(), "y", "binary", "auc", +1)
    assert set(fp.keys()) == _EXPECTED_KEYS


def test_regression_returns_all_keys():
    fp = compute(_regression_df(), "y", "regression", "rmse", -1)
    assert set(fp.keys()) == _EXPECTED_KEYS


def test_binary_target_stat_is_minority_ratio():
    df = _binary_df(n=200)
    fp = compute(df, "y", "binary", "auc", +1)
    minority = min(df["y"].value_counts()["count"].to_list())
    assert abs(fp["target_stat"] - minority / 200) < 1e-5


def test_regression_target_stat_nonzero_for_skewed_dist():
    """항등식 버그 회귀 방지: skewed 분포에서 target_stat != 0."""
    fp = compute(_regression_df(skew="right"), "y", "regression", "rmse", -1)
    assert fp["target_stat"] != 0.0


def test_regression_skew_differs_by_distribution():
    """서로 다른 분포 회귀 타깃이 서로 다른 target_stat을 갖는다."""
    fp_right = compute(_regression_df(skew="right"), "y", "regression", "rmse", -1)
    fp_left = compute(_regression_df(skew="left"), "y", "regression", "rmse", -1)
    assert fp_right["target_stat"] != fp_left["target_stat"]


def test_all_null_string_column_no_crash():
    """전체 null 문자열 컬럼 포함 시 n_text_ish 계산이 TypeError 없이 완료."""
    df = pl.DataFrame({
        "x0": [1.0, 2.0, 3.0],
        "notes": pl.Series([None, None, None], dtype=pl.String),
        "y": [0, 1, 0],
    })
    fp = compute(df, "y", "binary", "auc", +1)
    assert fp["n_text_ish"] == 0


def test_size_class_tiny():
    fp = compute(_binary_df(n=100), "y", "binary", "auc", +1)
    assert fp["size_class"] == "tiny"
