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
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

_RUNNER = Path(__file__).parent / "runner.py"
DEFAULT_TIMEOUT = 1200

# RLIMIT_AS(가상 주소공간, VSZ)는 numpy/BLAS/부스팅 라이브러리가 실사용보다
# 훨씬 큰 주소공간을 예약해서 물리 메모리가 남아도 import만으로 실패할 수
# 있다(ADR-027) — 그래서 그대로 6GiB로 둔다. 대신 이 워치독은 물리
# RSS(/proc/<pid>/status VmRSS)를 폴링해서 실제 메모리 고갈로 인한 커널
# OOM kill(rc=-9, 평균 13분을 태운 뒤 죽음, 2026-08 실측 계산의 37%)을
# 훨씬 싸게(폴링 주기 이내) 선제 차단한다. 컨테이너 mem_limit(백스톱)보다
# 낮게 잡아야 이 워치독이 먼저 죽여 원인이 명시된 error_signature를 남긴다.
_DEFAULT_RSS_LIMIT_BYTES = 4 * 1024 ** 3
_RSS_POLL_INTERVAL_SEC = 2.0

_HAVE_NEWNET = sys.platform == "linux" and hasattr(os, "CLONE_NEWNET")

# RLIMIT_AS는 물리 RSS가 아닌 가상 주소공간(VSZ) 상한 — numpy/BLAS/부스팅
# 라이브러리가 실사용보다 훨씬 큰 주소공간을 예약해 이 값이 너무 낮으면 물리
# 메모리가 남아도 import만으로 실패한다(decisions.md ADR-027). EVAL_MEM_LIMIT_BYTES
# env var로 대회/큐별 override 가능.
_DEFAULT_MEM_LIMIT_BYTES = 6 * 1024 ** 3

try:
    import resource as _resource

    def _set_resource_limits() -> None:
        mem = int(os.environ.get("EVAL_MEM_LIMIT_BYTES", str(_DEFAULT_MEM_LIMIT_BYTES)))
        cpu = int(os.environ.get("EVAL_CPU_LIMIT_SECS", "900"))
        _resource.setrlimit(_resource.RLIMIT_AS, (mem, mem))
        _resource.setrlimit(_resource.RLIMIT_CPU, (cpu, cpu))

    def _preexec_fn() -> None:
        if _HAVE_NEWNET and os.environ.get("EVAL_SANDBOX") != "none":
            try:
                os.unshare(os.CLONE_NEWNET)
            except OSError:
                pass  # CAP_SYS_ADMIN 없으면 조용히 스킵 (로컬 개발 환경 등)
        _set_resource_limits()

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
    selected_params: dict | None = None
    oof_preds: list[float] | None = None
    gain_vs_best_relative: float | None = None
    peak_rss_bytes: int | None = None


def _read_rss_bytes(pid: int) -> int | None:
    """/proc/<pid>/status의 VmRSS(KB)를 바이트로 읽는다. 프로세스가 이미 종료됐으면 None."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None
    return None


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
    collect_oof: bool = False,
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
            "collect_oof": collect_oof,
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

        rss_limit = int(os.environ.get("EVAL_RSS_LIMIT_BYTES", str(_DEFAULT_RSS_LIMIT_BYTES)))
        stdout_path = ws / "_stdout.log"
        stderr_path = ws / "_stderr.log"
        peak_rss = 0
        killed_reason: str | None = None
        start = time.monotonic()

        # stdout/stderr는 PIPE 대신 파일로 리다이렉트 — RSS 폴링 중 자식이
        # 파이프 버퍼를 채우면 부모가 안 읽어가는 동안 자식이 write()에서
        # 블로킹돼 데드락(watchdog이 kill할 기회조차 없이 멈춤)이 난다.
        with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
            proc = subprocess.Popen(
                [sys.executable, str(_RUNNER), tmpdir],
                stdout=out_f,
                stderr=err_f,
                env=env,
                preexec_fn=_PREEXEC,
            )
            while True:
                rss = _read_rss_bytes(proc.pid)
                if rss is not None:
                    peak_rss = max(peak_rss, rss)
                    if rss > rss_limit:
                        killed_reason = (
                            f"memory watchdog: peak RSS {rss // (1024 ** 2)}MB "
                            f"> limit {rss_limit // (1024 ** 2)}MB"
                        )
                        proc.kill()
                        proc.wait()
                        break
                remaining = timeout_sec - (time.monotonic() - start)
                if remaining <= 0:
                    proc.kill()
                    proc.wait()
                    killed_reason = f"timeout after {timeout_sec}s"
                    break
                try:
                    proc.wait(timeout=min(_RSS_POLL_INTERVAL_SEC, remaining))
                    break  # 자식이 스스로 종료
                except subprocess.TimeoutExpired:
                    continue

        peak_rss_bytes = peak_rss or None

        if killed_reason:
            return _err(killed_reason, peak_rss_bytes=peak_rss_bytes)

        out_path = ws / "output.json"
        if not out_path.exists():
            stderr = stderr_path.read_text(errors="replace")[:2000]
            return _err(
                f"runner exited without output.json (rc={proc.returncode})\n{stderr}",
                peak_rss_bytes=peak_rss_bytes,
            )

        try:
            out: dict = json.loads(out_path.read_text())
        except Exception as exc:
            return _err(f"failed to parse output.json: {exc}", peak_rss_bytes=peak_rss_bytes)

        if out.get("error_trace"):
            return _err(out["error_trace"], peak_rss_bytes=peak_rss_bytes)

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
            selected_params=out.get("selected_params"),
            oof_preds=out.get("oof_preds"),
            gain_vs_best_relative=out.get("gain_vs_best_relative"),
            peak_rss_bytes=peak_rss_bytes,
        )


def _err(msg: str, peak_rss_bytes: int | None = None) -> IsolatedResult:
    return IsolatedResult(
        cv_score=None, cv_fold_var=None, fold_scores=None,
        label=None, gain_vs_best=None, error_trace=msg,
        peak_rss_bytes=peak_rss_bytes,
    )
