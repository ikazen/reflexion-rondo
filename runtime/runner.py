"""Evaluation runner — executed inside the worker container.

Reads source.py, input.json, train.parquet from /workspace (or argv[1]).
Optionally reads best_pipeline.py as the base pipeline.
Writes output.json.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import polars as pl

WS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace")

_HOOK_NAMES = (
    "preprocess", "feature_transform", "param_candidates",
    "build_model", "postprocess_predictions",
)


def _write(payload: dict) -> None:
    (WS / "output.json").write_text(json.dumps(payload))


def _load_best_pipeline_class(best_source: str, BasePipeline: type) -> type:
    ns: dict = {}
    try:
        exec(compile(best_source, "<best_pipeline>", "exec"), ns)  # noqa: S102
    except Exception:
        return BasePipeline
    patch_cls = ns.get("Patch")
    if not patch_cls:
        return BasePipeline
    methods = {h: getattr(patch_cls, h) for h in _HOOK_NAMES if hasattr(patch_cls, h)}
    return type("BestPipeline", (BasePipeline,), methods)


def main() -> None:
    try:
        source = (WS / "source.py").read_text()
        inp: dict = json.loads((WS / "input.json").read_text())
        train = pl.read_parquet(WS / "train.parquet")
        best_path = WS / "best_pipeline.py"
        best_source = best_path.read_text() if best_path.exists() else None
    except Exception as exc:
        _write({"error_trace": f"runner setup failed: {exc}"})
        sys.exit(0)

    from evaluator.harness import BasePipeline, PatchedPipeline, PipelineContext, evaluate_pipeline

    BestPipelineCls = (
        _load_best_pipeline_class(best_source, BasePipeline) if best_source else BasePipeline
    )
    base = BestPipelineCls()

    patch_ns: dict = {}
    try:
        exec(compile(source, "<generated>", "exec"), patch_ns)  # noqa: S102
    except Exception:
        _write({"error_trace": traceback.format_exc()})
        sys.exit(0)

    patch_cls = patch_ns.get("Patch")
    if patch_cls is None:
        _write({"error_trace": "missing Patch class after exec"})
        sys.exit(0)

    try:
        patch = patch_cls()
    except Exception:
        _write({"error_trace": f"Patch() instantiation failed:\n{traceback.format_exc()}"})
        sys.exit(0)

    pipeline = PatchedPipeline(base, patch)

    ctx = PipelineContext(
        target_col=inp["target_col"],
        metric=inp["metric"],
        n_splits=inp["n_splits"],
        seed=inp["seed"],
        is_classification=inp["is_classification"],
        prev_best=inp.get("prev_best"),
        action_type=inp.get("action_type", ""),
    )

    try:
        result = evaluate_pipeline(pipeline, train, ctx)
        _write({
            "cv_score": result.cv_score,
            "cv_fold_var": result.cv_fold_var,
            "fold_scores": result.fold_scores,
            "label": result.label,
            "gain_vs_best": result.gain_vs_best,
            "feature_importance": result.feature_importance,
            "error_trace": None,
        })
    except Exception:
        _write({"error_trace": traceback.format_exc()})


if __name__ == "__main__":
    main()
