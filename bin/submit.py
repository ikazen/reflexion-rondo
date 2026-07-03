"""best attempt 코드로 전체 train 학습 → test 예측 → Kaggle 제출.

Usage:
    uv run python -m bin.submit --competition s4e1            # CSV만 생성
    uv run python -m bin.submit --competition s4e1 --submit   # 생성 후 Kaggle 제출
    uv run python -m bin.submit --competition s4e1 --attempt-id <id>  # 특정 attempt 지정
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import requests

ROOT = Path(__file__).parent.parent
_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")
RUNS_DIR = ROOT / "runs"
CODE_SEP = "# " + "-" * 60


def _load_best_code(competition_id: str, attempt_id: str | None) -> tuple[str, float, str]:
    """(code_source, cv_score, attempt_id) 반환.

    attempt_id 지정 시: 사용자가 명시적으로 고른 attempt(미확정이어도 허용) — 의도된 escape hatch.
    미지정(자동 선택) 시: BON-245(a) — confirmed 파이프라인(raw.pipelines, cross-seed 통과분)만
    소스로 쓴다. _load_pipeline()이 실제 제출하는 모델도 confirmed 소스에서 materialize된
    것이므로, 리포팅되는 cv_score/attempt_id도 같은 소스여야 한다. raw.attempts all-time
    max는 미확정 attempt를 가리킬 수 있어 리포팅 불일치를 낳았다.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from store.db import connect
    conn = connect(apply_schema=False)

    if attempt_id:
        row = conn.execute(
            "select code_path, cv_score, attempt_id from raw.attempts where attempt_id like %s",
            [f"{attempt_id}%"],
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

    row = conn.execute(
        """
        select p.code, p.cv_score, p.attempt_id
        from raw.pipelines p
        join raw.competitions c using (competition_id)
        where p.competition_id = %s
          and p.cv_score is not null
        order by c.metric_sign * p.cv_score desc
        limit 1
        """,
        [competition_id],
    ).fetchone()
    conn.close()
    if not row:
        raise ValueError(
            f"No confirmed pipeline for {competition_id} — "
            "use --attempt-id to submit an unconfirmed attempt explicitly"
        )
    source, cv_score, aid = row
    return source.strip(), cv_score, aid


def _read_csv(comp: object, name: str) -> pl.DataFrame:
    s3 = getattr(comp, "S3_DATA_PATH", None)
    if s3 and _MINIO_ENDPOINT:
        try:
            url = f"{_MINIO_ENDPOINT}/kaggle/{s3}{name}"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return pl.read_csv(resp.content)
        except Exception as exc:
            raise FileNotFoundError(
                f"MinIO read failed for {name} ({url}): {exc}"
            ) from exc
    return pl.read_csv(getattr(comp, "DATA_DIR") / name)


_HOOK_NAMES = (
    "preprocess", "feature_transform", "param_candidates",
    "build_model", "postprocess_predictions",
)


def _load_pipeline(competition_id: str, extra_source: str | None = None) -> object:
    """Load the materialized best pipeline from MinIO.

    extra_source: attempt source exec'd first so helper classes defined there
    (e.g. WeightedEnsemble) are available when the stored best pipeline runs.
    Needed for pipelines materialized before BON-184 fix (missing ClassDef support).
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from store.s3_code import download_best_pipeline
    from evaluator.harness import BasePipeline, PipelineContext

    best_source = download_best_pipeline(competition_id)
    if not best_source:
        return BasePipeline()

    ns: dict = {}
    if extra_source:
        exec(compile(extra_source, "<attempt_ns>", "exec"), ns)  # noqa: S102
    exec(compile(best_source, "<best_pipeline>", "exec"), ns)  # noqa: S102
    patch_cls = ns.get("Patch")
    if not patch_cls:
        return BasePipeline()
    methods = {h: getattr(patch_cls, h) for h in _HOOK_NAMES if hasattr(patch_cls, h)}
    BestPipelineCls = type("BestPipeline", (BasePipeline,), methods)
    return BestPipelineCls()


def _predict_raw(model, X, metric_class: str):
    """BON-245(b): metric_class(evaluator/metrics.get 3번째 값) 기준으로 proba/label
    분기 — comp.IS_CLASSIFICATION 기준이면 accuracy/f1/qwk 같은 label-metric
    classification 대회에서 predict_proba를 잘못 쓰게 된다.
    """
    if metric_class == "binary_proba":
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _submission_value_col(sample_columns: list[str], fallback: str) -> str:
    """BON-245(c): 제출 값 컬럼명은 sample_submission.csv의 실제 2번째 컬럼을 따른다."""
    return sample_columns[1] if len(sample_columns) > 1 else fallback


def _impute_train_test_median(train_np, test_np):
    """BON-245(d): NaN 중앙값 대치를 train/test 대칭 적용. medians는 train 기준."""
    col_medians = np.nanmedian(train_np, axis=0)
    train_mask, test_mask = np.isnan(train_np), np.isnan(test_np)
    if train_mask.any():
        train_np = np.where(train_mask, col_medians, train_np)
        print(f"  NaN {train_mask.sum()}개(train) → 훈련 중앙값으로 대체")
    if test_mask.any():
        test_np = np.where(test_mask, col_medians, test_np)
        print(f"  NaN {test_mask.sum()}개(test) → 훈련 중앙값으로 대체")
    return train_np, test_np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", required=True)
    parser.add_argument("--attempt-id", default=None, help="특정 attempt ID 앞 8자리 (참고용)")
    parser.add_argument("--submit", action="store_true", help="Kaggle에 바로 제출")
    parser.add_argument("--message", "-m", default=None, help="제출 메시지")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))

    comp = importlib.import_module(f"config.competitions.{args.competition}")

    source, cv_score, attempt_id = _load_best_code(comp.COMPETITION_ID, args.attempt_id)
    print(f"best attempt: {attempt_id[:8]}  cv={cv_score:.5f}")

    from evaluator.harness import PipelineContext, preselect_params
    pipeline = _load_pipeline(comp.COMPETITION_ID, extra_source=source)
    ctx = PipelineContext(
        target_col=comp.TARGET,
        metric=comp.METRIC,
        n_splits=5,
        seed=42,
        is_classification=comp.IS_CLASSIFICATION,
    )

    train = _read_csv(comp, "train.csv").drop(comp.DROP_COLS)
    test  = _read_csv(comp, "test.csv")

    sample = _read_csv(comp, "sample_submission.csv")
    id_col = sample.columns[0]
    test_ids = test[id_col]
    test_feat = test.drop([c for c in comp.DROP_COLS if c in test.columns])

    # dummy target in test so preprocess/feature_transform work
    test_with_dummy = test_feat.with_columns(pl.lit(0).cast(train[comp.TARGET].dtype).alias(comp.TARGET))

    params = preselect_params(pipeline, train, ctx)
    train_proc, test_proc = pipeline.preprocess(train, test_with_dummy, comp.TARGET, ctx)
    X_train, X_test = pipeline.feature_transform(train_proc, test_proc, comp.TARGET, ctx)
    y_train = train_proc[comp.TARGET].to_numpy()

    X_train_np, X_test_np = _impute_train_test_median(
        X_train.to_numpy().astype(float), X_test.to_numpy().astype(float)
    )

    model = pipeline.build_model(params, ctx)
    model.fit(X_train_np, y_train)

    from evaluator.metrics import get as get_metric
    _, _, metric_class = get_metric(comp.METRIC)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_preds = _predict_raw(model, X_test_np, metric_class)
    preds = pipeline.postprocess_predictions(raw_preds, ctx)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RUNS_DIR / f"submission_{comp.competition_id if hasattr(comp, 'competition_id') else args.competition}_{ts}.csv"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    value_col = _submission_value_col(sample.columns, comp.TARGET)
    pl.DataFrame({id_col: test_ids, value_col: preds}).write_csv(out)
    print(f"submission saved: {out}")

    if args.submit:
        import sys
        msg = args.message or f"reflexion best cv={cv_score:.5f} attempt={attempt_id[:8]}"
        result = subprocess.run(
            ["uv", "run", "kaggle", "competitions", "submit",
             "-c", comp.COMPETITION_ID, "-f", str(out), "-m", msg],
            capture_output=True, text=True,
        )
        print(result.stdout or result.stderr)
        if result.returncode != 0:
            print(f"kaggle submit failed (rc={result.returncode})", file=sys.stderr)
            sys.exit(result.returncode)
    else:
        print(f"\n제출하려면: uv run python -m bin.submit --competition {args.competition} --submit")


if __name__ == "__main__":
    main()
