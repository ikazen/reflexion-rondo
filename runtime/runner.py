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


def _eval_holdout(
    pipeline: object,
    train90: "pl.DataFrame",
    holdout10: "pl.DataFrame",
    ctx: object,
) -> float | None:
    """train90으로 fit, holdout10으로 1회 측정. audit holdout 전용.

    CV 결과와 독립적으로 일반화 성능을 추정한다. 실패 시 None 반환(caller가 무시).
    """
    import warnings
    from evaluator.harness import _mask_target, _strip_target, _encode_residual_categoricals, preselect_params
    from evaluator.metrics import get as get_metric

    fn, _, metric_class = get_metric(ctx.metric)
    params = preselect_params(pipeline, train90, ctx)
    # preprocess가 타깃을 변환(log1p 등)할 수 있으므로 채점은 변환 이전의 raw
    # 타깃(yho_raw)으로 한다 — evaluator/harness.py의 evaluate_pipeline/preselect_params와
    # 동일 계약.
    yho_raw = holdout10[ctx.target_col].to_numpy()
    tr2, ho2 = pipeline.preprocess(train90, holdout10, ctx.target_col, ctx)
    ytr = tr2[ctx.target_col].to_numpy()
    Xtr, Xho = pipeline.feature_transform(tr2, _mask_target(ho2, ctx.target_col), ctx.target_col, ctx)
    Xtr = _strip_target(Xtr, ctx.target_col)
    Xho = _strip_target(Xho, ctx.target_col)
    Xtr, Xho = _encode_residual_categoricals(Xtr, Xho)
    model = pipeline.build_model(params, ctx)
    model.fit(Xtr.to_numpy(), ytr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if metric_class == "binary_proba":
            raw_preds = model.predict_proba(Xho.to_numpy())[:, 1]
        else:
            raw_preds = model.predict(Xho.to_numpy())
    preds = pipeline.postprocess_predictions(raw_preds, ctx)
    return float(fn(yho_raw, preds))


def _load_best_pipeline_class(best_source: str, BasePipeline: type) -> type:
    ns: dict = {}
    exec(compile(best_source, "<best_pipeline>", "exec"), ns)  # noqa: S102
    patch_cls = ns.get("Patch")
    if not patch_cls:
        raise RuntimeError("best_pipeline.py has no Patch class")
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

    try:
        BestPipelineCls = (
            _load_best_pipeline_class(best_source, BasePipeline) if best_source else BasePipeline
        )
    except Exception:
        _write({"error_trace": f"best_pipeline load failed:\n{traceback.format_exc()}"})
        sys.exit(0)
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
        best_params=inp.get("best_params"),
    )

    try:
        result = evaluate_pipeline(pipeline, train, ctx, collect_oof=inp.get("collect_oof", False))
    except Exception:
        _write({"error_trace": traceback.format_exc()})
        return

    holdout_score = None
    holdout_path = WS / "holdout.parquet"
    if holdout_path.exists():
        try:
            holdout = pl.read_parquet(holdout_path)
            holdout_score = _eval_holdout(pipeline, train, holdout, ctx)
        except Exception:
            pass  # holdout 실패는 무시 — CV 결과는 유효

    _write({
        "cv_score": result.cv_score,
        "cv_fold_var": result.cv_fold_var,
        "fold_scores": result.fold_scores,
        "label": result.label,
        "gain_vs_best": result.gain_vs_best,
        "feature_importance": result.feature_importance,
        "is_noop_tie": result.is_noop_tie,
        "selected_params": result.selected_params,
        "oof_preds": result.oof_preds,
        "error_trace": None,
        "holdout_score": holdout_score,
    })


if __name__ == "__main__":
    main()
