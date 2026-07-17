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
DEFAULT_TIMEOUT = 1200  # BON-275: s5e5/s6e6(기존 최대 165k행 대비 3.5~4.5배 큰 데이터)에서
# 600s 타임아웃 실측 발생 확인 후 상향

_HAVE_NEWNET = sys.platform == "linux" and hasattr(os, "CLONE_NEWNET")

# 메모리/CPU 제한 — attempt 하나의 OOM이 컨테이너 전체를 죽이지 않도록.
# 제한 초과 시 subprocess가 SIGKILL(RLIMIT_AS) 또는 SIGXCPU(RLIMIT_CPU)로 종료되고
# runner output.json 없음 → _err() 경로로 error_trace에 기록.
#
# issue #20/#27: big 큐의 mac-server-big 워커는 Docker가 Colima VM 안에서 도는데,
# 원래 VM이 8GiB 고정이라 6GiB 기본값이 이론상 오버서브스크립션을 만들 수 있다고 보고
# #20에서 1.5GiB로 낮췄었다. 하지만 RLIMIT_AS는 물리 RSS가 아니라 가상 주소공간(VSZ)
# 상한이라 numpy/scipy/sklearn/lightgbm/catboost/xgboost 같은 라이브러리는 실제 쓰는
# 물리 메모리가 적어도 공유 라이브러리 mmap·BLAS 스레드풀 등으로 VSZ를 널찍하게 예약한다
# — 1.5GiB는 이런 라이브러리를 import하는 것만으로도 부족해서, 물리 메모리가 12GB나
# 남는 worker-vm에서도 신규 대회 부트스트랩 attempt 전체가 실패하는 회귀를 냈다(#27).
#
# Airflow 실측 히스토리(super-cycle 3678건 전수 + attempt task instance 표본 1200건)로
# 재확인한 실제 동시성은 mac-server-big 최대 3(설정 concurrency=4가 실제로 4까지 찬 적
# 없음), worker-vm-big 최대 1 — #20이 가정한 "4개 동시" 자체가 실측과 달랐다. 근본
# 해결로 mac-server의 Colima VM을 8→16GiB로 증설했고(vz 백엔드, demand-paged라 즉시
# 선점 아님), 이제 6GiB 기본값으로 되돌린다 — 3×6GiB=18GiB가 VSZ 기준으로는 16GiB를
# 근소 초과하지만 실제 물리 RSS는 훨씬 낮게 쓰이므로 무해할 것으로 판단. 필요하면
# EVAL_MEM_LIMIT_BYTES env var로 대회/큐별 override 가능(기존 메커니즘, 변경 없음).
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
            selected_params=out.get("selected_params"),
            oof_preds=out.get("oof_preds"),
        )


def _err(msg: str) -> IsolatedResult:
    return IsolatedResult(
        cv_score=None, cv_fold_var=None, fold_scores=None,
        label=None, gain_vs_best=None, error_trace=msg,
    )
