"""best attempt 코드로 전체 train 학습 → test 예측 → Kaggle 제출.

Usage:
    uv run python -m bin.submit --competition s4e1            # CSV만 생성
    uv run python -m bin.submit --competition s4e1 --submit   # 생성 후 Kaggle 제출
    uv run python -m bin.submit --competition s4e1 --attempt-id <id>  # 특정 attempt 지정
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

ROOT = Path(__file__).parent.parent
RUNS_DIR = ROOT / "runs"
CODE_SEP = "# " + "-" * 60


def _load_best_code(competition_id: str, attempt_id: str | None) -> tuple[str, float, str]:
    """(code_source, cv_score, attempt_id) 반환."""
    import sys
    sys.path.insert(0, str(ROOT))
    from store.db import connect
    conn = connect(apply_schema=False)
    if attempt_id:
        row = conn.execute(
            "select code_path, cv_score, attempt_id from raw.attempts where attempt_id like %s",
            [f"{attempt_id}%"],
        ).fetchone()
    else:
        row = conn.execute(
            """
            select a.code_path, a.cv_score, a.attempt_id
            from raw.attempts a
            join raw.competitions c using (competition_id)
            where a.competition_id = %s
              and a.cv_score is not null
              and a.error_trace is null
            order by c.metric_sign * a.cv_score desc
            limit 1
            """,
            [competition_id],
        ).fetchone()
    conn.close()

    if not row:
        raise ValueError(f"No valid attempt found for {competition_id}")

    code_path, cv_score, aid = row
    from store.s3_code import download as _code_download
    content = _code_download(code_path)
    if not content:
        raise FileNotFoundError(f"code not found: {code_path}")
    sep = CODE_SEP + "\n"
    source = content.split(sep, 1)[1].strip() if sep in content else content.strip()
    return source, cv_score, aid


def _exec_code(source: str) -> tuple:
    ns: dict = {}
    exec(compile(source, "<best_code>", "exec"), ns)  # noqa: S102
    return ns["feature_fn"], ns["model_fn"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", required=True)
    parser.add_argument("--attempt-id", default=None, help="특정 attempt ID 앞 8자리")
    parser.add_argument("--submit", action="store_true", help="Kaggle에 바로 제출")
    parser.add_argument("--message", "-m", default=None, help="제출 메시지")
    args = parser.parse_args()

    comp = importlib.import_module(f"config.competitions.{args.competition}")

    source, cv_score, attempt_id = _load_best_code(comp.COMPETITION_ID, args.attempt_id)
    print(f"best attempt: {attempt_id[:8]}  cv={cv_score:.5f}")

    train = pl.read_csv(comp.DATA_DIR / "train.csv").drop(comp.DROP_COLS)
    test  = pl.read_csv(comp.DATA_DIR / "test.csv")

    # test에 있는 id 계열 컬럼 보존 (sample_submission 형식 맞추기)
    sample = pl.read_csv(comp.DATA_DIR / "sample_submission.csv")
    id_col = sample.columns[0]
    test_ids = test[id_col]
    test_feat = test.drop([c for c in comp.DROP_COLS if c in test.columns])

    # feature_fn은 (train_fold, valid_fold, target) 시그니처
    # 제출 시: train 전체로 fit, test에 apply
    # target이 test에 없으므로 dummy 추가
    test_with_dummy = test_feat.with_columns(pl.lit(0).cast(train[comp.TARGET].dtype).alias(comp.TARGET))

    feature_fn, model_fn = _exec_code(source)

    X_train, X_test = feature_fn(train, test_with_dummy, comp.TARGET)
    y_train = train[comp.TARGET].to_numpy()

    model = model_fn({})
    model.fit(X_train.to_numpy(), y_train)

    import numpy as np
    X_test_np = X_test.to_numpy().astype(float)
    if np.isnan(X_test_np).any():
        col_medians = np.nanmedian(X_train.to_numpy().astype(float), axis=0)
        nan_mask = np.isnan(X_test_np)
        X_test_np = np.where(nan_mask, col_medians, X_test_np)
        print(f"  NaN {nan_mask.sum()}개 → 훈련 중앙값으로 대체")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if comp.IS_CLASSIFICATION:
            preds = model.predict_proba(X_test_np)[:, 1]
        else:
            preds = model.predict(X_test_np)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RUNS_DIR / f"submission_{comp.competition_id if hasattr(comp, 'competition_id') else args.competition}_{ts}.csv"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({id_col: test_ids, comp.TARGET: preds}).write_csv(out)
    print(f"submission saved: {out}")

    if args.submit:
        msg = args.message or f"reflexion best cv={cv_score:.5f} attempt={attempt_id[:8]}"
        result = subprocess.run(
            ["uv", "run", "kaggle", "competitions", "submit",
             "-c", comp.COMPETITION_ID, "-f", str(out), "-m", msg],
            capture_output=True, text=True,
        )
        print(result.stdout or result.stderr)
    else:
        print(f"\n제출하려면: uv run python -m bin.submit --competition {args.competition} --submit")


if __name__ == "__main__":
    main()
