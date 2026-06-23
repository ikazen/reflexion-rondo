"""Deterministic meta-feature calculator (spec §1.4).

Target stats are computed from train only — no test leakage.
"""
import polars as pl


def compute(
    train: pl.DataFrame,
    target: str,
    task_type: str,
    metric: str,
    metric_sign: int,
) -> dict:
    feature_cols = [c for c in train.columns if c != target]
    X = train.select(feature_cols)
    y = train[target]

    n_rows, n_cols = X.shape

    n_numeric = sum(
        1 for c in X.columns
        if X[c].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                          pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                          pl.Float32, pl.Float64)
    )
    n_categorical = sum(
        1 for c in X.columns
        if X[c].dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum)
    )
    n_datetime = sum(
        1 for c in X.columns
        if X[c].dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration)
    )
    n_text_ish = sum(
        1 for c in X.columns
        if X[c].dtype in (pl.Utf8, pl.String)
        and (X[c].drop_nulls().cast(pl.String).str.len_chars().mean() or 0) > 20
    )

    total_cells = n_rows * n_cols
    missing = sum(X[c].null_count() for c in X.columns)
    missing_ratio_overall = missing / total_cells if total_cells > 0 else 0.0

    cardinalities = [X[c].n_unique() for c in X.columns if X[c].dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum)]
    cardinality_max  = max(cardinalities) if cardinalities else 0
    cardinality_mean = sum(cardinalities) / len(cardinalities) if cardinalities else 0.0

    if task_type in ("binary", "multiclass"):
        counts = y.value_counts()
        min_count = counts["count"].min()
        target_stat = float(min_count) / n_rows  # minority class ratio
    else:
        arr = y.drop_nulls().to_numpy()
        import numpy as np
        mean = float(arr.mean())
        std  = float(arr.std())
        target_stat = float(((arr - mean) ** 3).mean() / std**3) if std > 0 else 0.0  # skew

    if n_rows < 10_000:
        size_class = "tiny"
    elif n_rows < 100_000:
        size_class = "small"
    elif n_rows < 1_000_000:
        size_class = "mid"
    else:
        size_class = "large"

    return {
        "n_rows":               n_rows,
        "n_cols":               n_cols,
        "task_type":            task_type,
        "metric":               metric,
        "metric_sign":          metric_sign,
        "n_numeric":            n_numeric,
        "n_categorical":        n_categorical,
        "n_datetime":           n_datetime,
        "n_text_ish":           n_text_ish,
        "missing_ratio_overall": round(missing_ratio_overall, 6),
        "cardinality_max":      cardinality_max,
        "cardinality_mean":     round(cardinality_mean, 2),
        "target_stat":          round(float(target_stat), 6),
        "size_class":           size_class,
    }
