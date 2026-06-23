"""Phase 0 PoC — LightGBM baseline, no LLM, no retrieval."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from lightgbm import LGBMClassifier

from evaluator.harness import BasePipeline, PipelineContext, evaluate_pipeline
from store.db import connect, ensure_competition, insert_attempt

COMPETITION_ID = "playground-series-s4e1"
DATA_DIR = Path(__file__).parent.parent / "data" / COMPETITION_ID
TARGET = "Exited"
METRIC = "auc"

PARAMS: dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

CAT_COLS = ["Geography", "Gender"]
DROP_COLS = ["id", "CustomerId", "Surname", TARGET]


def feature_fn(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    target: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # label-encode categoricals using train 맵 (누수 방지)
    cat_maps: dict[str, dict] = {}
    for c in CAT_COLS:
        if c in train.columns:
            vals = train[c].unique().to_list()
            cat_maps[c] = {v: i for i, v in enumerate(sorted(str(v) for v in vals))}

    def prep(df: pl.DataFrame, maps: dict) -> pl.DataFrame:
        drop = [c for c in DROP_COLS if c in df.columns]
        df = df.drop(drop)
        for c, m in maps.items():
            if c in df.columns:
                df = df.with_columns(
                    pl.col(c).cast(pl.String).replace_strict(m, default=-1).cast(pl.Int32)
                )
        return df

    return prep(train, cat_maps), prep(valid, cat_maps)


def model_fn(params: dict) -> LGBMClassifier:
    return LGBMClassifier(**params)


def main() -> None:
    train = pl.read_csv(DATA_DIR / "train.csv")
    print(f"train shape: {train.shape}")

    conn = connect()
    ensure_competition(
        conn,
        competition_id=COMPETITION_ID,
        name="Binary Classification with a Bank Churn Dataset",
        task_type="binary",
        metric=METRIC,
        metric_sign=+1,
    )

    prev_best = conn.execute(
        "select max(metric_sign * cv_score) * max(metric_sign) from raw.attempts "
        "join raw.competitions using (competition_id) "
        "where raw.attempts.competition_id = %s",
        [COMPETITION_ID],
    ).fetchone()[0]

    class _BaselinePatch(BasePipeline):
        def feature_transform(self, train, valid, target, ctx):
            return feature_fn(train, valid, target)

        def build_model(self, params, ctx):
            return model_fn(params or PARAMS)

    ctx = PipelineContext(
        target_col=TARGET, metric=METRIC, n_splits=5, seed=42,
        is_classification=True, prev_best=prev_best,
    )
    result = evaluate_pipeline(_BaselinePatch(), train, ctx)

    print(f"CV AUC:   {result.cv_score:.5f}")
    print(f"fold scores: {[f'{s:.5f}' for s in result.fold_scores]}")
    print(f"fold std: {(result.cv_fold_var ** 0.5):.5f}")
    print(f"label:    {result.label}  gain: {result.gain_vs_best}")

    insert_attempt(conn, {
        "attempt_id":     str(uuid.uuid4()),
        "competition_id": COMPETITION_ID,
        "run_ts":         datetime.now(timezone.utc),
        "stage":          "bootstrap",
        "hypothesis":     "LightGBM baseline — categorical encoding, default params",
        "action_type":    "model_swap",
        "model_type":     "lgbm",
        "params":         json.dumps(PARAMS),
        "features":       json.dumps({"drop": DROP_COLS, "cat": CAT_COLS}),
        "cv_score":       result.cv_score,
        "cv_fold_var":    result.cv_fold_var,
        "lb_score":       None,
        "label":          result.label,
        "gain_vs_best":   result.gain_vs_best,
        "error_trace":    None,
    })
    conn.close()
    print("Postgres 기록 완료.")


if __name__ == "__main__":
    main()
