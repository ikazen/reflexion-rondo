"""runtime/isolate.py 리소스 상한 회귀 테스트.

mac-server-big(당시 Colima VM 8GiB 고정)의 이론상 오버서브스크립션을 막으려고
RLIMIT_AS를 6GiB→1.5GiB로 낮췄으나, RLIMIT_AS는 물리 RSS가 아니라 가상 주소공간(VSZ)
상한이라 numpy/scipy/sklearn 등 라이브러리를 import하는 것만으로도 부족해 신규 대회
부트스트랩 전체가 실패하는 회귀를 냈다 — 물리 메모리가 남는 worker-vm에서도 실패.

이후 mac-server Colima VM을 8→16GiB로 증설하고(실측 최대 동시성도 3이지 4가 아님을
확인), RLIMIT_AS를 원래 값 6GiB로 복원했다.

RLIMIT_CPU는 과거 soft==hard==900으로 걸었으나, 리눅스가 hard를 먼저 검사해
SIGXCPU 없이 곧장 SIGKILL(rc=-9)로 죽여 OOM killer 사망과 구분이 안 됐다
(2026-08 실측: 계산의 40%, 전부 재시도까지 태워 attempt당 최대 16분). 지금은
eval_isolated의 폴링 루프가 CPU 시간을 직접 감시해서 명시적 원인으로 선제
kill하고, RLIMIT_CPU는 폴링이 놓쳤을 때만 발동하는 soft<hard 백스톱으로 강등했다.

os.unshare(CLONE_NEWNET)는 CAP_SYS_ADMIN을 요구하고 테스트 프로세스 자체의
네트워크 namespace에 영향을 줄 수 있어 건드리지 않는다 — _set_resource_limits()는
그 로직과 분리돼 있어 안전하게 단위 테스트 가능하다.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.isolate import (
    _CPU_BACKSTOP_HARD_MARGIN_SECS,
    _CPU_BACKSTOP_SOFT_MARGIN_SECS,
    _DEFAULT_MEM_LIMIT_BYTES,
    _make_preexec,
    _set_resource_limits,
)

_EXPECTED_DEFAULT = 6 * 1024 ** 3


def test_default_mem_limit_is_6_gib() -> None:
    assert _DEFAULT_MEM_LIMIT_BYTES == _EXPECTED_DEFAULT


def test_set_resource_limits_uses_new_default_when_no_override() -> None:
    with patch.dict("os.environ", {}, clear=True), \
         patch("runtime.isolate._resource.setrlimit") as mock_setrlimit:
        _set_resource_limits(cpu_budget=900)

    import resource
    calls = {c.args[0]: c.args[1] for c in mock_setrlimit.call_args_list}
    assert calls[resource.RLIMIT_AS] == (_EXPECTED_DEFAULT, _EXPECTED_DEFAULT)
    assert calls[resource.RLIMIT_CPU] == (
        900 + _CPU_BACKSTOP_SOFT_MARGIN_SECS, 900 + _CPU_BACKSTOP_HARD_MARGIN_SECS,
    )


def test_set_resource_limits_respects_mem_env_override() -> None:
    override_bytes = str(3 * 1024 ** 3)
    with patch.dict("os.environ", {"EVAL_MEM_LIMIT_BYTES": override_bytes}), \
         patch("runtime.isolate._resource.setrlimit") as mock_setrlimit:
        _set_resource_limits(cpu_budget=900)

    import resource
    calls = {c.args[0]: c.args[1] for c in mock_setrlimit.call_args_list}
    assert calls[resource.RLIMIT_AS] == (3 * 1024 ** 3, 3 * 1024 ** 3)


def test_set_resource_limits_cpu_backstop_tracks_budget_argument() -> None:
    """RLIMIT_CPU 백스톱은 env가 아니라 호출자가 넘긴 cpu_budget에서 파생된다 —
    eval_isolated가 attempt 단위로 남은 예산을 계산해 매 재시도마다 다르게 넘기므로."""
    with patch.dict("os.environ", {}, clear=True), \
         patch("runtime.isolate._resource.setrlimit") as mock_setrlimit:
        _set_resource_limits(cpu_budget=200)

    import resource
    calls = {c.args[0]: c.args[1] for c in mock_setrlimit.call_args_list}
    assert calls[resource.RLIMIT_CPU] == (
        200 + _CPU_BACKSTOP_SOFT_MARGIN_SECS, 200 + _CPU_BACKSTOP_HARD_MARGIN_SECS,
    )


def test_make_preexec_returns_callable_that_applies_budget() -> None:
    with patch("runtime.isolate._resource.setrlimit") as mock_setrlimit, \
         patch("runtime.isolate._HAVE_NEWNET", False):
        preexec = _make_preexec(cpu_budget=900)
        assert preexec is not None
        preexec()

    import resource
    calls = {c.args[0]: c.args[1] for c in mock_setrlimit.call_args_list}
    assert calls[resource.RLIMIT_CPU] == (
        900 + _CPU_BACKSTOP_SOFT_MARGIN_SECS, 900 + _CPU_BACKSTOP_HARD_MARGIN_SECS,
    )


#
# eval_isolated는 내부적으로 subprocess.run 대신 Popen + RSS 폴링 루프를 쓴다
# (runner exited rc=-9 OOM kill이 평균 773초를 태우고서야 죽는 것을 워치독으로
# 선제 차단하기 위함 — 2026-08 처리량 진단). 그래서 아래 fake는 subprocess.run이
# 아니라 Popen을 대체하고, poll 루프가 기대하는 pid/wait()/kill() 인터페이스를
# 갖춘다.

class _FakePopen:
    """subprocess.Popen 대역 — pid는 실존하지 않아 _read_rss_bytes가 항상 None을
    반환하므로(실제 프로세스 없음), RSS 워치독 분기를 타지 않고 wait()에서 바로
    종료한다."""

    def __init__(self, cmd, **kwargs):
        self.pid = 2 ** 30  # 실존 가능성이 사실상 0인 pid
        self.returncode = 0
        self._on_init(cmd)

    def _on_init(self, cmd) -> None:
        pass

    def wait(self, timeout=None):
        return self.returncode

    def kill(self) -> None:
        pass


def test_eval_isolated_passes_through_gain_vs_best_relative() -> None:
    """subprocess(runner.py)가 쓴 output.json의 gain_vs_best_relative가 IsolatedResult로
    그대로 전달되는지 확인 — metric 스케일 정규화 필드가 격리 경계를 넘어야 한다."""
    import json as _json

    import polars as pl

    from runtime.isolate import eval_isolated

    class _FakeProc(_FakePopen):
        def _on_init(self, cmd) -> None:
            tmpdir = cmd[2]
            (Path(tmpdir) / "output.json").write_text(_json.dumps({
                "cv_score": 0.9, "cv_fold_var": 0.01, "fold_scores": [0.89, 0.9, 0.91],
                "label": "jump", "gain_vs_best": 0.05, "gain_vs_best_relative": 0.06,
                "error_trace": None,
            }))

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    with patch("runtime.isolate.subprocess.Popen", side_effect=_FakeProc):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
        )
    assert result.gain_vs_best_relative == 0.06
    assert result.peak_rss_bytes is None  # pid가 실존하지 않아 RSS를 못 읽음
    assert result.peak_cpu_sec is None  # 마찬가지로 CPU 시간도 못 읽음


def test_eval_isolated_kills_on_rss_over_limit() -> None:
    """peak RSS가 EVAL_RSS_LIMIT_BYTES를 넘으면 output.json을 기다리지 않고
    즉시 kill + 원인이 명시된 error_trace를 반환한다."""
    import polars as pl

    from runtime.isolate import eval_isolated

    class _FakeProc(_FakePopen):
        def __init__(self, cmd, **kwargs):
            super().__init__(cmd, **kwargs)
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    over_limit = 5 * 1024 ** 3
    with patch("runtime.isolate.subprocess.Popen", side_effect=_FakeProc), \
         patch("runtime.isolate._read_rss_bytes", return_value=over_limit), \
         patch.dict("os.environ", {"EVAL_RSS_LIMIT_BYTES": str(4 * 1024 ** 3)}):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
        )
    assert result.error_trace is not None
    assert "memory watchdog" in result.error_trace
    assert result.peak_rss_bytes == over_limit


def test_eval_isolated_kills_on_cpu_budget_exceeded() -> None:
    """peak CPU 시간이 예산을 넘으면 output.json을 기다리지 않고 즉시 kill +
    'cpu budget exceeded'가 명시된 error_trace를 반환한다 — 과거엔 커널이
    SIGKILL(rc=-9)로 흔적 없이 죽여 OOM과 구분이 안 됐다."""
    import polars as pl

    from runtime.isolate import eval_isolated

    class _FakeProc(_FakePopen):
        def __init__(self, cmd, **kwargs):
            super().__init__(cmd, **kwargs)
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    over_budget = 950.0
    with patch("runtime.isolate.subprocess.Popen", side_effect=_FakeProc), \
         patch("runtime.isolate._read_cpu_seconds", return_value=over_budget):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
            cpu_budget_sec=900,
        )
    assert result.error_trace is not None
    assert "cpu budget exceeded" in result.error_trace
    assert result.peak_cpu_sec == over_budget


def test_eval_isolated_cpu_budget_sec_overrides_env_default() -> None:
    """호출자가 넘긴 cpu_budget_sec이 EVAL_CPU_BUDGET_SECS 기본값보다 우선한다 —
    cycle/run.py가 attempt 단위로 남은 예산을 재시도마다 다르게 넘기기 위함."""
    import polars as pl

    from runtime.isolate import eval_isolated

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    with patch("runtime.isolate.subprocess.Popen", side_effect=_FakePopen), \
         patch("runtime.isolate._read_cpu_seconds", return_value=50.0), \
         patch.dict("os.environ", {"EVAL_CPU_BUDGET_SECS": "900"}):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
            cpu_budget_sec=30,
        )
    assert result.error_trace is not None
    assert "cpu budget exceeded" in result.error_trace
    assert "limit 30s" in result.error_trace


def _exits_after(n_polls: int):
    """n_polls번 폴링된 뒤 스스로 종료하는 가짜 subprocess."""
    class _Proc:
        pid = 4321
        returncode = 0

        def __init__(self, *a, **kw) -> None:
            self.calls = 0
            self.killed = False

        def wait(self, timeout=None):
            self.calls += 1
            if timeout is not None and self.calls <= n_polls:
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            return 0

        def kill(self) -> None:
            self.killed = True

    return _Proc


def test_wall_timeout_defaults_to_at_least_cpu_budget() -> None:
    """벽시계 상한이 CPU 예산보다 낮으면 예산을 선점해 무력화한다(#207) — 호출자가
    timeout_sec을 안 주면 CPU 예산이 항상 먼저 걸리도록 벽시계를 그만큼 늘린다."""
    import polars as pl

    from runtime.isolate import DEFAULT_TIMEOUT, eval_isolated

    assert DEFAULT_TIMEOUT < 3000  # 예산이 기본 벽시계보다 커야 의미 있는 케이스

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    clock = iter([0.0] + [float(DEFAULT_TIMEOUT) + 1.0] * 50)
    with patch("runtime.isolate.subprocess.Popen", side_effect=_exits_after(2)), \
         patch("runtime.isolate._read_cpu_seconds", return_value=10.0), \
         patch("runtime.isolate._read_rss_bytes", return_value=1000), \
         patch("runtime.isolate.time.monotonic", side_effect=lambda: next(clock)):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
            cpu_budget_sec=3000,
        )
    assert result.error_trace is not None
    assert "timeout" not in result.error_trace


def test_explicit_timeout_sec_is_respected() -> None:
    """호출자가 timeout_sec을 명시하면 CPU 예산과 무관하게 그 값이 벽시계 상한이다."""
    import polars as pl

    from runtime.isolate import eval_isolated

    train = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    clock = iter([0.0] + [60.0] * 50)
    with patch("runtime.isolate.subprocess.Popen", side_effect=_exits_after(50)), \
         patch("runtime.isolate._read_cpu_seconds", return_value=10.0), \
         patch("runtime.isolate._read_rss_bytes", return_value=1000), \
         patch("runtime.isolate.time.monotonic", side_effect=lambda: next(clock)):
        result = eval_isolated(
            source="class Patch:\n    pass\n", train=train, target_col="y", metric="auc",
            prev_best=0.85, n_splits=3, seed=42, is_classification=True,
            cpu_budget_sec=3000, timeout_sec=30,
        )
    assert result.error_trace is not None
    assert "timeout after 30s" in result.error_trace


def test_err_result_defaults_gain_relative_to_none() -> None:
    from runtime.isolate import _err
    result = _err("some failure")
    assert result.gain_vs_best_relative is None
    assert result.peak_rss_bytes is None
    assert result.peak_cpu_sec is None


def test_err_result_carries_peak_rss_bytes() -> None:
    from runtime.isolate import _err
    result = _err("memory watchdog: ...", peak_rss_bytes=123)
    assert result.peak_rss_bytes == 123


def test_err_result_carries_peak_cpu_sec() -> None:
    from runtime.isolate import _err
    result = _err("cpu budget exceeded: ...", peak_cpu_sec=456.0)
    assert result.peak_cpu_sec == 456.0
