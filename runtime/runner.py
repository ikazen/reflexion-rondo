"""Evaluation runner — Docker 컨테이너 안에서 실행된다.

/workspace/source.py, /workspace/input.json, /workspace/train.parquet 를 읽고
/workspace/output.json 에 결과를 쓴다.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import polars as pl

WS = Path("/workspace")


def _write(payload: dict) -> None:
    (WS / "output.json").write_text(json.dumps(payload))


def main() -> None:
    try:
        source = (WS / "source.py").read_text()
        inp: dict = json.loads((WS / "input.json").read_text())
        train = pl.read_parquet(WS / "train.parquet")
    except Exception as exc:
        _write({"error_trace": f"runner setup failed: {exc}"})
        sys.exit(0)

    ns: dict = {}
    try:
        exec(compile(source, "<generated>", "exec"), ns)  # noqa: S102
    except Exception:
        _write({"error_trace": traceback.format_exc()})
        sys.exit(0)

    missing = [n for n in ("feature_fn", "model_fn") if n not in ns]
    if missing:
        _write({"error_trace": f"missing after exec: {missing}"})
        sys.exit(0)

    feature_fn = ns["feature_fn"]
    model_fn = ns["model_fn"]

    from evaluator.harness import run as eval_run
    try:
        result = eval_run(
            train=train,
            target_col=inp["target_col"],
            metric=inp["metric"],
            feature_fn=feature_fn,
            model_fn=model_fn,
            params={},
            prev_best=inp.get("prev_best"),
            n_splits=inp["n_splits"],
            seed=inp["seed"],
            is_classification=inp["is_classification"],
        )
        _write({
            "cv_score": result.cv_score,
            "cv_fold_var": result.cv_fold_var,
            "fold_scores": result.fold_scores,
            "label": result.label,
            "gain_vs_best": result.gain_vs_best,
            "error_trace": None,
        })
    except Exception:
        _write({"error_trace": traceback.format_exc()})


if __name__ == "__main__":
    main()
