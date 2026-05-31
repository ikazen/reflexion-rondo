"""Phase 0 — full-train predict + Kaggle submission.

Usage:
    uv run python bin/submit.py           # 파일만 생성
    uv run python bin/submit.py --submit  # 생성 후 Kaggle 제출
"""
import argparse
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from bin.run_cycle import (
    COMPETITION_ID, DATA_DIR, TARGET, PARAMS,
    feature_fn, model_fn,
)

RUNS_DIR = Path(__file__).parent.parent / "runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Kaggle에 바로 제출")
    args = parser.parse_args()

    train = pl.read_csv(DATA_DIR / "train.csv")
    test  = pl.read_csv(DATA_DIR / "test.csv")
    ids   = test["id"]

    X_train, X_test = feature_fn(train, test, TARGET)
    y_train = train[TARGET].to_numpy()

    model = model_fn(PARAMS)
    model.fit(X_train.to_numpy(), y_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        preds = model.predict_proba(X_test.to_numpy())[:, 1]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RUNS_DIR / f"submission_{ts}.csv"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"id": ids, TARGET: preds}).write_csv(out)
    print(f"submission saved: {out}")

    if args.submit:
        result = subprocess.run(
            ["uv", "run", "kaggle", "competitions", "submit",
             COMPETITION_ID, "-f", str(out), "-m", f"Phase0 baseline {ts}"],
            capture_output=True, text=True,
        )
        print(result.stdout or result.stderr)
    else:
        print("제출하려면: PYTHONPATH=. uv run python bin/submit.py --submit")


if __name__ == "__main__":
    main()
