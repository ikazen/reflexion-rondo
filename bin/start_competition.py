"""Cold-start: fingerprint 계산 → raw.competitions insert.

Usage:
    uv run python bin/start_competition.py \\
        --id playground-series-s4e1 \\
        --name "Binary Classification with a Bank Churn Dataset" \\
        --task binary --metric auc
"""
import argparse
import json
from pathlib import Path

import polars as pl

from store.db import connect, ensure_competition
from store.fingerprint import compute as compute_fingerprint
from evaluator.metrics import get as get_metric

DATA_ROOT = Path(__file__).parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",     required=True, help="Kaggle competition slug")
    parser.add_argument("--name",   required=True)
    parser.add_argument("--task",   required=True, choices=["binary", "multiclass", "regression"])
    parser.add_argument("--metric", required=True, help="auc / rmse / accuracy / ...")
    parser.add_argument("--target", default=None,  help="target column name (기본: 마지막 컬럼)")
    args = parser.parse_args()

    _, metric_sign, _ = get_metric(args.metric)

    train_path = DATA_ROOT / args.id / "train.csv"
    if not train_path.exists():
        print(f"train.csv not found: {train_path}")
        print(f"run: uv run kaggle competitions download {args.id} -p data/{args.id}")
        return

    train = pl.read_csv(train_path)
    target = args.target or train.columns[-1]
    print(f"train shape: {train.shape}  target: '{target}'")

    fp = compute_fingerprint(train, target, args.task, args.metric, metric_sign)
    print("fingerprint:", json.dumps(fp, indent=2))

    conn = connect()
    ensure_competition(
        conn,
        competition_id=args.id,
        name=args.name,
        task_type=args.task,
        metric=args.metric,
        metric_sign=metric_sign,
        fingerprint=fp,
    )
    conn.close()
    print(f"\nraw.competitions 등록 완료: {args.id}")


if __name__ == "__main__":
    main()
