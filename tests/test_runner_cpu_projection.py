"""runtime/runner.py — fold-1 CPU 투영 중단(#361)의 에러 표기.

CpuBudgetProjectedError는 traceback 없이 메시지 그대로 error_trace에 써야 한다 —
cycle/run.py:_resource_kill_feedback이 `startswith("cpu budget exceeded")`로 kill을 알아본다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).parent.parent


def _run_runner(tmp_path: Path, **input_overrides) -> dict:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 3))
    pl.DataFrame({
        "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": (x[:, 0] + x[:, 1] > 0).astype(float),
    }).write_parquet(tmp_path / "train.parquet")
    (tmp_path / "source.py").write_text("class Patch:\n    pass\n")
    (tmp_path / "input.json").write_text(json.dumps({
        "target_col": "y", "metric": "auc", "prev_best": None, "n_splits": 3, "seed": 42,
        "is_classification": True, **input_overrides,
    }))
    subprocess.run(
        [sys.executable, str(ROOT / "runtime" / "runner.py"), str(tmp_path)],
        check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    return json.loads((tmp_path / "output.json").read_text())


def test_projected_abort_is_written_as_a_plain_message(tmp_path) -> None:
    out = _run_runner(tmp_path, cpu_budget_sec=0.001)
    assert out["error_trace"].startswith("cpu budget exceeded: projected ")
    assert "Traceback" not in out["error_trace"]


def test_without_a_budget_the_runner_evaluates_normally(tmp_path) -> None:
    out = _run_runner(tmp_path)
    assert out["error_trace"] is None
    assert out["cv_score"] is not None
