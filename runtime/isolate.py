"""LLM 생성 코드(class Patch)를 격리 subprocess로 실행한다.

프로덕션(Linux, CAP_SYS_ADMIN 있음): preexec_fn에서 os.unshare(CLONE_NEWNET)으로
network namespace를 분리해 subprocess egress 차단. rlimit(AS/CPU) + timeout 병행.

폴백(SYS_ADMIN 없음, 로컬 mac): env allowlist + rlimit + timeout만 적용.
네트워크 샌드박스 없음 — EVAL_SANDBOX=none 으로 명시적 비활성도 가능.

DockerOperator에 cap_add=["SYS_ADMIN"] 필요(컨테이너 자체 네트워크는 유지,
차단은 subprocess preexec_fn 레벨에서만).
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

_HAVE_NEWNET = sys.platform == "linux" and hasattr(os, "CLONE_NEWNET")

# 메모리/CPU 제한 — attempt 하나의 OOM이 컨테이너 전체를 죽이지 않도록.
# 제한 초과 시 subprocess가 SIGKILL(RLIMIT_AS) 또는 SIGXCPU(RLIMIT_CPU)로 종료되고
# runner output.json 없음 → _err() 경로로 error_trace에 기록.
try:
    import resource as _resource

    def _preexec_fn() -> None:
        if _HAVE_NEWNET and os.environ.get("EVAL_SANDBOX") != "none":
            try:
                os.unshare(os.CLONE_NEWNET)
            except OSError:
                pass  # CAP_SYS_ADMIN 없으면 조용히 스킵 (로컬 개발 환경 등)
        mem = int(os.environ.get("EVAL_MEM_LIMIT_BYTES", str(6 * 1024 ** 3)))
        cpu = int(os.environ.get("EVAL_CPU_LIMIT_SECS", "900"))
        _resource.setrlimit(_resource.RLIMIT_AS, (mem, mem))
        _resource.setrlimit(_resource.RLIMIT_CPU, (cpu, cpu))

    _PREEXEC = _preexec_fn
except (ImportError, AttributeError):
    _PREEXEC = None  # Windows/non-Linux fallback


@dataclass(frozen=True, slots=True)
class IsolatedResult:
    cv_score: float | None
    cv_fold_var: float | None
    fold_scores: list[float] | None
    label: str | None
    gain_vs_best: float | None
    error_trace: str | None
    feature_importance: dict | None = None
    holdout_score: float | None = None
    is_noop_tie: bool = False


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
    best_params: dict | None = None,
    timeout_sec: int = DEFAULT_TIMEOUT,
    holdout_data: pl.DataFrame | None = None,
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
            "best_params": best_params,
        }))
        if best_source:
            (ws / "best_pipeline.py").write_text(best_source)
        if holdout_data is not None:
            holdout_data.write_parquet(ws / "holdout.parquet")

        _EVAL_ENV_ALLOWLIST = {
            "PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "LC_CTYPE",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        }
        env = {k: v for k, v in os.environ.items() if k in _EVAL_ENV_ALLOWLIST}
        env["PYTHONPATH"] = str(_RUNNER.parent.parent)
        env["HOME"] = tmpdir  # catboost_info 등 홈 쓰기를 tmpdir로 격리

        try:
            proc = subprocess.run(
                [sys.executable, str(_RUNNER), tmpdir],
                timeout=timeout_sec,
                capture_output=True,
                text=True,
                env=env,
                preexec_fn=_PREEXEC,
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
            holdout_score=out.get("holdout_score"),
            is_noop_tie=out.get("is_noop_tie", False),
        )


def _err(msg: str) -> IsolatedResult:
    return IsolatedResult(
        cv_score=None, cv_fold_var=None, fold_scores=None,
        label=None, gain_vs_best=None, error_trace=msg,
    )
