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


def test_runner_reuses_a_cached_fold1_match(tmp_path) -> None:
    """#376: input.json의 known_fold1_scores가 실제 서브프로세스 실행에서도 fold 재계산을 대체한다.
    먼저 캐시 없이 한 번 돌려 진짜 fold-1 점수를 얻고, fold 2+ 값만 가짜 sentinel로 바꿔 캐시로
    준다 — 결정적 seed라 "그냥 다시 계산해도 같은 값"이 나오는 걸 배제하고, 캐시가 실제로
    fold 2+ 재계산을 건너뛰었는지 증명한다."""
    real = _run_runner(tmp_path)
    assert real["error_trace"] is None
    poisoned = [real["fold_scores"][0], 0.123456789, 0.987654321]
    cached = _run_runner(tmp_path, known_fold1_scores=[poisoned])
    assert cached["fold_scores"] == poisoned
    assert cached["cv_score"] == sum(poisoned) / len(poisoned)


def test_runner_writes_a_progress_log_for_the_watchdog(tmp_path) -> None:
    """#421: eval_isolated의 워치독은 kill 시 이 파일의 마지막 줄을 error_trace에 붙여 kill 위치를 남긴다."""
    _run_runner(tmp_path)

    lines = (tmp_path / "_progress.log").read_text().splitlines()
    assert lines[0].startswith("stage=eval_start")
    assert [line.split()[1] for line in lines if "fold_done" in line] == ["fold=1/3", "fold=2/3", "fold=3/3"]
    assert lines[-1].startswith("stage=cv_done")
