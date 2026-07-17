"""issue #20/#27: runtime/isolate.py 메모리 기본값 회귀 테스트.

#20에서 mac-server-big(당시 Colima VM 8GiB 고정)의 이론상 오버서브스크립션을 막으려고
RLIMIT_AS를 6GiB→1.5GiB로 낮췄으나, RLIMIT_AS는 물리 RSS가 아니라 가상 주소공간(VSZ)
상한이라 numpy/scipy/sklearn 등 라이브러리를 import하는 것만으로도 부족해 신규 대회
부트스트랩 전체가 실패하는 회귀를 냈다(#27) — 물리 메모리가 남는 worker-vm에서도 실패.

#27에서 mac-server Colima VM을 8→16GiB로 증설하고(실측 최대 동시성도 3이지 4가 아님을
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
