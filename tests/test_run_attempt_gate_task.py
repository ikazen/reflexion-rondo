"""bin/run_attempt_gate_task.py — straggler 대기 완화 게이트 (#203, #214).

발화 조건(_should_fire)은 순수 함수라 대부분의 케이스를 mock 없이 검증한다.
main()의 컨텍스트 조회/폴링 루프는 별도로 mock 처리해 exit code와 로그를 검증한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.run_attempt_gate_task import _should_fire, main  # noqa: E402


def test_quorum_fires_on_success_count_regardless_of_span():
    assert _should_fire(
        n_success=2, n_done=2, span_sec=1.0, waited_sec=1.0,
        min_done=2, grace_sec=900, max_wait_sec=3000,
    ) == "quorum"


def test_two_errors_do_not_trigger_quorum():
    """#214 회귀 방지 — 2개 다 에러(성공 0)면 n_done=2라도 quorum이 발동하면 안 된다.
    발동하면 promote가 winner 없이 조기 종료되고, 아직 도는 세 번째(성공할 수도 있는)
    attempt는 이 사이클에서 영영 빠진다."""
    assert _should_fire(
        n_success=0, n_done=2, span_sec=50.0, waited_sec=50.0,
        min_done=2, grace_sec=900, max_wait_sec=3000,
    ) is None


def test_two_errors_still_fall_through_to_grace():
    """quorum은 안 뜨지만, 에러라도 결과가 나온 지 오래면(grace) 무기한 대기하지
    않는다 — n_done(성공+실패) 기준은 그대로 유지."""
    assert _should_fire(
        n_success=0, n_done=2, span_sec=901.0, waited_sec=901.0,
        min_done=2, grace_sec=900, max_wait_sec=3000,
    ) == "grace"


def test_one_success_one_error_below_quorum_and_below_grace_waits():
    assert _should_fire(
        n_success=1, n_done=2, span_sec=100.0, waited_sec=100.0,
        min_done=2, grace_sec=900, max_wait_sec=3000,
    ) is None


def test_below_quorum_past_grace_fires():
    assert _should_fire(
        n_success=1, n_done=1, span_sec=901.0, waited_sec=901.0,
        min_done=2, grace_sec=900, max_wait_sec=3000,
    ) == "grace"


def test_zero_done_past_maxwait_fires():
    assert _should_fire(
        n_success=0, n_done=0, span_sec=None, waited_sec=3001.0,
        min_done=2, grace_sec=900, max_wait_sec=3000,
    ) == "maxwait"


def test_zero_done_before_maxwait_waits():
    """n_done=0이면 span_sec도 NULL(SQL min(run_ts) over 0 rows) — grace 분기가 이걸로
    TypeError 없이 넘어가고 maxwait 전까지는 계속 대기해야 한다."""
    assert _should_fire(
        n_success=0, n_done=0, span_sec=None, waited_sec=100.0,
        min_done=2, grace_sec=900, max_wait_sec=3000,
    ) is None


def test_main_exits_zero_when_context_missing(monkeypatch, capsys):
    """context row가 안 보이면(retrieve 지연 등) 재시도 후 exit 0 — 사이클을 죽이면 안 된다."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    monkeypatch.setattr(sys, "argv", ["run_attempt_gate_task", "--run-id", "rid-1"])
    monkeypatch.setattr("bin.run_attempt_gate_task._CONTEXT_LOOKUP_RETRY_SEC", 0.01)
    with patch("store.db.connect", return_value=conn), patch("time.sleep"):
        try:
            main()
            raised = None
        except SystemExit as e:
            raised = e.code
    assert raised == 0
    assert "no context" in capsys.readouterr().out


def test_main_fires_on_quorum_and_exits_zero(monkeypatch, capsys):
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [
        ("sc-123",),     # context lookup
        (2, 2, 120.0),   # poll: n_success=2, n_done=2 -> quorum
    ]
    monkeypatch.setattr(sys, "argv", ["run_attempt_gate_task", "--run-id", "rid-1"])
    with patch("store.db.connect", return_value=conn), patch("time.sleep"):
        try:
            main()
            raised = None
        except SystemExit as e:
            raised = e.code
    assert raised == 0
    assert "reason=quorum" in capsys.readouterr().out


def test_main_does_not_fire_on_two_errors_then_fires_once_success_lands(monkeypatch, capsys):
    """#214 회귀 방지 — main() 레벨에서도 에러 2개로는 안 멈추고, 세 번째가 성공으로
    도착해야 quorum이 뜬다."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [
        ("sc-123",),     # context lookup
        (0, 2, 50.0),    # poll 1: n_success=0, n_done=2(둘 다 에러) -> 대기 계속
        (1, 3, 80.0),    # poll 2: 세 번째가 성공 -> 여전히 min_done=2 미달, 대기
        (2, 3, 90.0),    # poll 3: 성공 2개(가정상 재시도로) -> quorum
    ]
    monkeypatch.setattr(sys, "argv", ["run_attempt_gate_task", "--run-id", "rid-1"])
    with patch("store.db.connect", return_value=conn), patch("time.sleep"):
        try:
            main()
            raised = None
        except SystemExit as e:
            raised = e.code
    assert raised == 0
    out = capsys.readouterr().out
    assert "reason=quorum" in out
    assert "reason=maxwait" not in out


def test_main_continues_past_transient_poll_error(monkeypatch, capsys):
    """폴링 중 DB 예외가 나도 루프가 죽지 않고 다음 폴에서 계속돼야 한다."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [
        ("sc-123",),           # context lookup
        RuntimeError("boom"),  # first poll fails
        (2, 2, 50.0),          # second poll succeeds -> quorum
    ]
    monkeypatch.setattr(sys, "argv", ["run_attempt_gate_task", "--run-id", "rid-1"])
    with patch("store.db.connect", return_value=conn), patch("time.sleep"):
        try:
            main()
            raised = None
        except SystemExit as e:
            raised = e.code
    assert raised == 0
    assert "reason=quorum" in capsys.readouterr().out
