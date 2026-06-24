"""LLM 생성 코드(class Patch)를 샌드박스 subprocess로 실행한다.

프로덕션(Linux, bwrap 존재): bubblewrap --unshare-net으로 네트워크 차단 +
--ro-bind/--tmpfs FS 화이트리스트. rlimit(AS/CPU) + timeout 병행.

폴백(bwrap 부재, 로컬 mac): env allowlist + rlimit + timeout만 적용.
네트워크/FS 샌드박스 없음.

EVAL_SANDBOX=none 으로 강제 폴백, =bwrap 으로 bwrap 강제(테스트용).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

_RUNNER = Path(__file__).parent / "runner.py"
DEFAULT_TIMEOUT = 600

# 메모리/CPU 제한 — attempt 하나의 OOM이 컨테이너 전체를 죽이지 않도록.
# 제한 초과 시 subprocess가 SIGKILL(RLIMIT_AS) 또는 SIGXCPU(RLIMIT_CPU)로 종료되고
# runner output.json 없음 → _err() 경로로 error_trace에 기록.
try:
    import resource as _resource

    def _set_eval_limits() -> None:
        mem = int(os.environ.get("EVAL_MEM_LIMIT_BYTES", str(6 * 1024 ** 3)))
        cpu = int(os.environ.get("EVAL_CPU_LIMIT_SECS", "900"))
        _resource.setrlimit(_resource.RLIMIT_AS, (mem, mem))
        _resource.setrlimit(_resource.RLIMIT_CPU, (cpu, cpu))

    _PREEXEC = _set_eval_limits
except (ImportError, AttributeError):
    _PREEXEC = None  # Windows/non-Linux fallback


def _bwrap_cmd(tmpdir: str, inner_cmd: list[str]) -> list[str] | None:
    """bwrap 래핑 커맨드 반환. 사용 불가/비활성 시 None.

    네트워크 차단(--unshare-net)과 FS 화이트리스트(--ro-bind/--tmpfs)를
    한 번에 실현한다. container-level --network none 대신 subprocess 레벨에서
    적용하는 이유: task 컨테이너 자체는 Postgres/MinIO/Ollama 접근에 네트워크 필요.
    """
    sandbox = os.environ.get("EVAL_SANDBOX", "")
    if sandbox == "none":
        return None
    use_bwrap = sandbox == "bwrap" or (sys.platform == "linux" and shutil.which("bwrap"))
    if not use_bwrap:
        return None

    app_root = str(_RUNNER.parent.parent)
    ro_bind_paths = ["/usr", "/lib", "/etc"]
    for extra in ("/lib64", "/lib32"):
        if Path(extra).exists():
            ro_bind_paths.append(extra)
    # /app = 컨테이너 내 repo 루트 (.venv 포함). 로컬 경로에선 실제 repo 루트.
    if app_root not in ("", "/") and Path(app_root).exists():
        ro_bind_paths.append(app_root)

    cmd = [
        "bwrap",
        "--unshare-net", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--die-with-parent",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--chdir", tmpdir,
    ]
    for p in ro_bind_paths:
        cmd += ["--ro-bind", p, p]
    cmd += ["--bind", tmpdir, tmpdir]
    cmd += inner_cmd
    return cmd


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

        inner_cmd = [sys.executable, str(_RUNNER), tmpdir]
        cmd = _bwrap_cmd(tmpdir, inner_cmd) or inner_cmd

        try:
            proc = subprocess.run(
                cmd,
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
        )


def _err(msg: str) -> IsolatedResult:
    return IsolatedResult(
        cv_score=None, cv_fold_var=None, fold_scores=None,
        label=None, gain_vs_best=None, error_trace=msg,
    )
