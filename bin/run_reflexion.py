"""Reflexion 루프 한 사이클 실행.

Usage:
    uv run python -m bin.run_reflexion
    uv run python -m bin.run_reflexion --stage bootstrap --cycles 1
"""
import argparse
from pathlib import Path

import polars as pl

from cycle.run import CycleConfig, run_cycle
from store.db import connect

COMPETITION_ID = "playground-series-s4e1"
TARGET = "Exited"
METRIC = "auc"
DATA_DIR = Path(__file__).parent.parent / "data" / COMPETITION_ID

EDA_CARD = """competition: playground-series-s4e1 (Bank Churn)
task: binary classification  metric: AUC  target: Exited
rows: 165034  cols: 13 (10 numeric, 3 categorical)
categorical: Geography, Gender, Surname
numeric: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
target rate: 21.2% (mild imbalance)
no missing values"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",  default="bootstrap", choices=["bootstrap", "reflexion", "exploitation"])
    parser.add_argument("--cycles", type=int, default=1)
    args = parser.parse_args()

    train = pl.read_csv(DATA_DIR / "train.csv")
    conn = connect()

    failed = 0
    for i in range(args.cycles):
        print(f"\n--- cycle {i + 1}/{args.cycles} (stage={args.stage}) ---")
        config = CycleConfig(
            competition_id=COMPETITION_ID,
            train=train,
            target_col=TARGET,
            metric=METRIC,
            stage=args.stage,
            eda_card=EDA_CARD,
            n_splits=5,
            seed=42,
            k_retrieve=5,
            is_classification=True,
        )
        try:
            result = run_cycle(conn, config)
        except Exception as exc:
            failed += 1
            print(f"[cycle {i + 1} FAILED] {exc}")
            continue

        print(f"attempt_id:    {result.attempt_id}")
        print(f"cv_score:      {result.cv_score}")
        print(f"label:         {result.label}")
        print(f"gain_vs_best:  {result.gain_vs_best}")
        print(f"reflection_id: {result.reflection_id}")
        print(f"code:          {result.code_path}")
        if result.error_trace:
            print(f"error:\n{result.error_trace}")

    if failed:
        print(f"\n{failed}/{args.cycles} cycle(s) failed.")

    conn.close()


if __name__ == "__main__":
    main()
