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

    holdout10의 타깃은 preprocess에 넘기기 전 bin/submit.py와 동일한 더미 상수로
    치환한다(evaluator.harness.replace_with_dummy_target) — test.csv에 타깃이
    아예 없어 dummy로 채우는 실제 제출 조건을 그대로 재현해야 preprocess 훅의
    valid-target 의존 누수(GH #96)를 이 holdout이 실제로 걸러낼 수 있다(#98).
    이전엔 real target을 그대로 넘겨 누수를 그대로 재현했다 — s5e10 실측:
    holdout_score(0.02135)가 cv_score(0.02151)와 거의 같아 전혀 못 걸렀다.
    dummy는 상수라 feature_transform이 읽어도 정보 누수가 없으므로, 여기서부터는
    submit.py와 동일하게 별도 마스킹 없이 그대로 흘린다.
    """
    import warnings
    from evaluator.harness import (
        _build_model_safe,
        _encode_residual_categoricals,
        _fit_predict_ensemble,
        _strip_target,
        preselect_params,
        replace_with_dummy_target,
    )
    from evaluator.metrics import get as get_metric

    fn, _, metric_class = get_metric(ctx.metric)
    params = preselect_params(pipeline, train90, ctx)
    # preprocess가 타깃을 변환(log1p 등)할 수 있으므로 채점은 변환 이전의 raw
    # 타깃(yho_raw)으로 한다 — evaluator/harness.py의 evaluate_pipeline/preselect_params와
    # 동일 계약.
    yho_raw = holdout10[ctx.target_col].to_numpy()
    holdout10_dummy = replace_with_dummy_target(holdout10, ctx.target_col, train90)
    tr2, ho2 = pipeline.preprocess(train90, holdout10_dummy, ctx.target_col, ctx)
    ytr = tr2[ctx.target_col].to_numpy()
    Xtr, Xho = pipeline.feature_transform(tr2, ho2, ctx.target_col, ctx)
    Xtr = _strip_target(Xtr, ctx.target_col)
    Xho = _strip_target(Xho, ctx.target_col)
    Xtr, Xho = _encode_residual_categoricals(Xtr, Xho)
    Xtr_np, Xho_np = Xtr.to_numpy(), Xho.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = pipeline.ensemble_spec(ctx)
        if spec is not None:
            # ensemble_spec 파이프라인도 이 게이트를 거쳐야 한다(#226) — 이전엔
            # build_model만 호출해 ensemble이 조용히 단일 모델로 대체됐다.
            raw_preds = _fit_predict_ensemble(spec, Xtr_np, ytr, Xho_np, None, ctx, metric_class)
        else:
            model = _build_model_safe(pipeline, params, ctx)
            model.fit(Xtr_np, ytr)
            if metric_class == "binary_proba":
                raw_preds = model.predict_proba(Xho_np)[:, 1]
            else:
                raw_preds = model.predict(Xho_np)
    preds = pipeline.postprocess_predictions(raw_preds, ctx)
    return float(fn(yho_raw, preds))


def _load_best_pipeline(best_source: str, BasePipeline: type, PatchedPipeline: type) -> object:
    """best_pipeline.py의 Patch를 base로 인스턴스화한다.

    이전엔 훅 메서드만 골라 `type(...)`으로 새 클래스에 옮겨 붙였는데(고정 이름
    목록 기준이라 ensemble_spec이 빠져 있었다 — #226), 이는 bin/submit.py가
    이미 겪고 고친 것과 동일한 클래스의 버그(#83: Patch가 훅 밖 클래스 속성/중첩
    클래스에 의존하면 그 상태가 통째로 소실됨)다. submit.py와 동일하게
    PatchedPipeline(BasePipeline(), patch_cls())로 통일해 인스턴스 상태를 보존한다.
    """
    ns: dict = {}
    exec(compile(best_source, "<best_pipeline>", "exec"), ns)  # noqa: S102
    patch_cls = ns.get("Patch")
    if not patch_cls:
        raise RuntimeError("best_pipeline.py has no Patch class")
    return PatchedPipeline(BasePipeline(), patch_cls())


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
        base = (
            _load_best_pipeline(best_source, BasePipeline, PatchedPipeline)
            if best_source else BasePipeline()
        )
    except Exception:
        _write({"error_trace": f"best_pipeline load failed:\n{traceback.format_exc()}"})
        sys.exit(0)

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
        "gain_vs_best_relative": result.gain_vs_best_relative,
        "feature_importance": result.feature_importance,
        "is_noop_tie": result.is_noop_tie,
        "selected_params": result.selected_params,
        "oof_preds": result.oof_preds,
        "error_trace": None,
        "holdout_score": holdout_score,
    })


if __name__ == "__main__":
    main()
