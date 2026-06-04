"""Host-side Docker 격리 실행기 (ADR-013).

eval_isolated()는 생성 코드를 Docker 컨테이너 안에서 실행해 결과를 반환한다.
컨테이너는 --network none, 메모리/CPU/pids 상한, 타임아웃 아래에서 돌아간다.
OOM·무한루프·격리 위반은 모두 error_trace로 기록되어 워커 프로세스를 죽이지 않는다.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

EVAL_IMAGE = os.environ.get("EVAL_DOCKER_IMAGE", "reflexion-eval:latest")
_MEMORY = "8g"
_CPUS = "2"
_PIDS = "512"
DEFAULT_TIMEOUT = 300


@dataclass(frozen=True, slots=True)
class IsolatedResult:
    cv_score: float | None
    cv_fold_var: float | None
    fold_scores: list[float] | None
    label: str | None
    gain_vs_best: float | None
    error_trace: str | None


def eval_isolated(
    source: str,
    train: pl.DataFrame,
    target_col: str,
    metric: str,
    prev_best: float | None,
    n_splits: int,
    seed: int,
    is_classification: bool,
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
        }))

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            f"--memory={_MEMORY}",
            f"--cpus={_CPUS}",
            f"--pids-limit={_PIDS}",
            "-v", f"{tmpdir}:/workspace",
            EVAL_IMAGE,
        ]

        try:
            proc = subprocess.run(
                cmd,
                timeout=timeout_sec,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return _err(f"timeout after {timeout_sec}s")
        except FileNotFoundError:
            return _err("docker not found — is Docker installed and in PATH?")

        out_path = ws / "output.json"
        if not out_path.exists():
            stderr = (proc.stderr or "")[:2000]
            return _err(
                f"container exited without output.json (rc={proc.returncode})\n{stderr}"
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
        )


def _err(msg: str) -> IsolatedResult:
    return IsolatedResult(
        cv_score=None, cv_fold_var=None, fold_scores=None,
        label=None, gain_vs_best=None, error_trace=msg,
    )
