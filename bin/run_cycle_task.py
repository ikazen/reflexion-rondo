"""Airflow DockerOperator 진입점 — cycle 1회 실행.

Usage (컨테이너 내부):
    uv run python -m bin.run_cycle_task --competition s4e1 --stage bootstrap --queue-id <id>

성공 시 exit 0. DB 연결·설정 오류 시 exit 1 (DAG run failed).
error_trace (LLM/eval 소프트 에러)는 attempt에 기록되지만 exit 0 — 다음 사이클 계속.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).parent.parent


def _s3_storage_options() -> dict[str, str]:
    return {
        "aws_access_key_id": os.environ["MINIO_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["MINIO_SECRET_ACCESS_KEY"],
        "aws_endpoint_url": os.environ["MINIO_ENDPOINT"],
        "aws_region": "us-east-1",
        "aws_allow_http": "true",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", required=True)
    parser.add_argument("--stage", "-s", required=True)
    parser.add_argument("--queue-id", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))

    from store.db import connect, ensure_competition
    from cycle.run import CycleConfig, run_cycle

    conn = connect(apply_schema=False)

    try:
        comp = importlib.import_module(f"config.competitions.{args.competition}")
    except ModuleNotFoundError as exc:
        print(f"[run_cycle_task] competition config not found: {exc}", file=sys.stderr)
        sys.exit(1)

    s3_data_dir = getattr(comp, "S3_DATA_DIR", None)
    if s3_data_dir:
        train = pl.read_csv(
            f"{s3_data_dir}train.csv",
            storage_options=_s3_storage_options(),
        ).drop(comp.DROP_COLS)
    else:
        train = pl.read_csv(comp.DATA_DIR / "train.csv").drop(comp.DROP_COLS)

    ensure_competition(
        conn,
        competition_id=comp.COMPETITION_ID,
        name=comp.NAME,
        task_type=comp.TASK_TYPE,
        metric=comp.METRIC,
        metric_sign=comp.METRIC_SIGN,
    )

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

    result = run_cycle(conn, config)
    conn.close()

    print(
        f"[run_cycle_task] queue_id={args.queue_id} "
        f"attempt={result.attempt_id[:8]} "
        f"cv={result.cv_score} label={result.label}"
    )


if __name__ == "__main__":
    main()
