"""issue #20: runtime/isolate.py 메모리 기본값 회귀 테스트.

근본원인: mac-server-big 워커의 Docker가 Colima VM(8GiB 고정) 안에서 돌고
concurrency=4라, 기존 RLIMIT_AS=6GiB 기본값은 동시 4개 실행 시 이론상 24GiB를
요구해 Python 자신의 RLIMIT_AS(catch 가능한 MemoryError)보다 VM 커널 OOM killer가
먼저 SIGKILL — "runner exited without output.json (rc=-9)"로 원인불명 처리됐다.
1.5GiB로 낮추면 VM 안에서 먼저 catch 가능한 MemoryError가 발생한다.

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

_EXPECTED_DEFAULT = int(1.5 * 1024 ** 3)


def test_default_mem_limit_is_1_5_gib() -> None:
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
