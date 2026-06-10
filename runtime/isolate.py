"""eval runner를 컨테이너 내부 subprocess로 직접 실행한다.

DockerOperator 컨테이너 자체가 격리 환경이므로 DooD 불필요.
timeout은 subprocess.run timeout_sec으로 제어.
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

_RUNNER = Path(__file__).parent / "runner.py"
DEFAULT_TIMEOUT = 600


@dataclass(frozen=True, slots=True)
class IsolatedResult:
    cv_score: float | None
    cv_fold_var: float | None
    fold_scores: list[float] | None
    label: str | None
    gain_vs_best: float | None
    error_trace: str | None
    feature_importance: dict | None = None


def eval_isolated(
    source: str,
    train: pl.DataFrame,
    target_col: str,
    metric: str,
    prev_best: float | None,
    n_splits: int,
    seed: int,
    is_classification: bool,
    action_type: str = "",
    best_source: str | None = None,
    timeout_sec: int = DEFAULT_TIMEOUT,
) -> IsolatedResult:
    with tempfile.TemporaryDirectory(prefix="rondo-eval-") as tmpdir:
        ws = Path(tmpdir)
        (ws / "source.py").write_text(source)
        train.write_parquet(ws / "train.parquet")
        (ws / "input.json").write_text(json.dumps({
            "target_col": target_col,
            "metric": metric,
            "prev_best": prev_best,
            "n_splits": n_splits,
            "seed": seed,
            "is_classification": is_classification,
            "action_type": action_type,
        }))
        if best_source:
            (ws / "best_pipeline.py").write_text(best_source)

        _EVAL_ENV_ALLOWLIST = {
            "PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "LC_CTYPE",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        }
        env = {k: v for k, v in os.environ.items() if k in _EVAL_ENV_ALLOWLIST}
        env["PYTHONPATH"] = str(_RUNNER.parent.parent)
        try:
            proc = subprocess.run(
                [sys.executable, str(_RUNNER), tmpdir],
                timeout=timeout_sec,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return _err(f"timeout after {timeout_sec}s")

        out_path = ws / "output.json"
        if not out_path.exists():
            stderr = (proc.stderr or "")[:2000]
            return _err(
                f"runner exited without output.json (rc={proc.returncode})\n{stderr}"
            )

        try:
            out: dict = json.loads(out_path.read_text())
        except Exception as exc:
            return _err(f"failed to parse output.json: {exc}")

        if out.get("error_trace"):
            return _err(out["error_trace"])

        return IsolatedResult(
            cv_score=out.get("cv_score"),
            cv_fold_var=out.get("cv_fold_var"),
            fold_scores=out.get("fold_scores"),
            label=out.get("label"),
            gain_vs_best=out.get("gain_vs_best"),
            error_trace=None,
            feature_importance=out.get("feature_importance"),
        )


def _err(msg: str) -> IsolatedResult:
    return IsolatedResult(
        cv_score=None, cv_fold_var=None, fold_scores=None,
        label=None, gain_vs_best=None, error_trace=msg,
    )
