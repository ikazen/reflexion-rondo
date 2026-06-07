"""Super-cycle attempt step — Airflow task 2-4 of 4 (3 parallel).

Reads shared context from raw.super_cycle_context, runs one attempt.

Usage (container):
    uv run python -m bin.run_attempt_task --competition s4e1 --stage reflexion --queue-id <id>
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).parent.parent
_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", required=True)
    parser.add_argument("--stage", "-s", required=True)
    parser.add_argument("--queue-id", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))

    from store.db import connect
    from cycle.run import CycleConfig, run_attempt_core

    conn = connect(apply_schema=False)

    ctx_row = conn.execute(
        """
        SELECT super_cycle_id, competition_id, prev_best_cv, lessons
        FROM raw.super_cycle_context
        WHERE queue_id = %s
        """,
        [args.queue_id],
    ).fetchone()
    if not ctx_row:
        print(f"[run_attempt_task] no context for queue_id={args.queue_id}", file=sys.stderr)
        sys.exit(1)

    super_cycle_id, competition_id, prev_best_cv, lessons_raw = ctx_row
    lessons = json.loads(lessons_raw) if isinstance(lessons_raw, str) else lessons_raw

    try:
        comp = importlib.import_module(f"config.competitions.{args.competition}")
    except ModuleNotFoundError as exc:
        print(f"[run_attempt_task] competition config not found: {exc}", file=sys.stderr)
        sys.exit(1)

    s3_path = getattr(comp, "S3_DATA_PATH", None)
    if s3_path and _MINIO_ENDPOINT:
        train = pl.read_csv(f"{_MINIO_ENDPOINT}/kaggle/{s3_path}train.csv").drop(comp.DROP_COLS)
    else:
        train = pl.read_csv(comp.DATA_DIR / "train.csv").drop(comp.DROP_COLS)

    config = CycleConfig(
        competition_id=comp.COMPETITION_ID,
        train=train,
        target_col=comp.TARGET,
        metric=comp.METRIC,
        stage=args.stage,
        eda_card=comp.EDA_CARD,
        n_splits=getattr(comp, "N_SPLITS", 5),
        seed=42,
        k_retrieve=5,
        is_classification=comp.IS_CLASSIFICATION,
        slug=args.competition,
    )

    data = run_attempt_core(
        conn, config, lessons, prev_best_cv,
        super_cycle_id=super_cycle_id,
    )
    conn.close()

    print(
        f"[run_attempt_task] queue_id={args.queue_id}"
        f" attempt={data.attempt_id[:8]}"
        f" cv={data.cv_score} label={data.label}"
        f" gain={data.gain_vs_best}"
    )


if __name__ == "__main__":
    main()
