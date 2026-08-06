"""runtime/isolate.py 메모리 기본값 회귀 테스트.

mac-server-big(당시 Colima VM 8GiB 고정)의 이론상 오버서브스크립션을 막으려고
RLIMIT_AS를 6GiB→1.5GiB로 낮췄으나, RLIMIT_AS는 물리 RSS가 아니라 가상 주소공간(VSZ)
상한이라 numpy/scipy/sklearn 등 라이브러리를 import하는 것만으로도 부족해 신규 대회
부트스트랩 전체가 실패하는 회귀를 냈다 — 물리 메모리가 남는 worker-vm에서도 실패.

이후 mac-server Colima VM을 8→16GiB로 증설하고(실측 최대 동시성도 3이지 4가 아님을
확인), RLIMIT_AS를 원래 값 6GiB로 복원했다.

os.unshare(CLONE_NEWNET)는 CAP_SYS_ADMIN을 요구하고 테스트 프로세스 자체의
네트워크 namespace에 영향을 줄 수 있어 건드리지 않는다 — _set_resource_limits()는
그 로직과 분리돼 있어 안전하게 단위 테스트 가능하다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.isolate import _DEFAULT_MEM_LIMIT_BYTES, _set_resource_limits

_EXPECTED_DEFAULT = 6 * 1024 ** 3


def test_default_mem_limit_is_6_gib() -> None:
    assert _DEFAULT_MEM_LIMIT_BYTES == _EXPECTED_DEFAULT


def test_set_resource_limits_uses_new_default_when_no_override() -> None:
    with patch.dict("os.environ", {}, clear=True), \
         patch("runtime.isolate._resource.setrlimit") as mock_setrlimit:
        _set_resource_limits()

    import resource
    calls = {c.args[0]: c.args[1] for c in mock_setrlimit.call_args_list}
    assert calls[resource.RLIMIT_AS] == (_EXPECTED_DEFAULT, _EXPECTED_DEFAULT)
    assert calls[resource.RLIMIT_CPU] == (900, 900)


def test_set_resource_limits_respects_env_override() -> None:
    override_bytes = str(3 * 1024 ** 3)
    with patch.dict("os.environ", {"EVAL_MEM_LIMIT_BYTES": override_bytes, "EVAL_CPU_LIMIT_SECS": "600"}), \
         patch("runtime.isolate._resource.setrlimit") as mock_setrlimit:
        _set_resource_limits()

    import resource
    calls = {c.args[0]: c.args[1] for c in mock_setrlimit.call_args_list}
    assert calls[resource.RLIMIT_AS] == (3 * 1024 ** 3, 3 * 1024 ** 3)
    assert calls[resource.RLIMIT_CPU] == (600, 600)


def test_set_resource_limits_cpu_default_unchanged() -> None:
    with patch.dict("os.environ", {}, clear=True), \
         patch("runtime.isolate._resource.setrlimit") as mock_setrlimit:
        _set_resource_limits()

    import resource
    calls = {c.args[0]: c.args[1] for c in mock_setrlimit.call_args_list}
    assert calls[resource.RLIMIT_CPU] == (900, 900)


# --- eval_isolated: subprocess 격리 경계 필드 전달 ---
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


def test_err_result_defaults_gain_relative_to_none() -> None:
    from runtime.isolate import _err
    result = _err("some failure")
    assert result.gain_vs_best_relative is None
    assert result.peak_rss_bytes is None


def test_err_result_carries_peak_rss_bytes() -> None:
    from runtime.isolate import _err
    result = _err("memory watchdog: ...", peak_rss_bytes=123)
    assert result.peak_rss_bytes == 123
