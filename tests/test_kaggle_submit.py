"""_kaggle_submit 캐시 우선 경로.

promote 시점에 캐싱된 submission CSV(store.s3_code.download_submission_csv)가 있으면
fit(bin.submit 서브프로세스) 없이 캐시된 CSV를 바로 kaggle CLI로 업로드해야 한다 —
매일 06:00 auto-submit이 ops-vm(daemon 상주, 2 OCPU)에서 매번 fit하며 CPU를 포화시키던
경로를 대체하는 것이 이번 변경의 목적. 캐시 미스 시에는 기존 bin.submit 경로로 폴백한다.

_kaggle_submit은 subprocess.run 대신 _run_in_pgroup을 쓴다 — 타임아웃 시
uv run이 spawn한 손자 python 프로세스까지 확실히 죽이기 위해서다(process group kill).
"""
from __future__ import annotations

import contextlib
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.api import _kaggle_submit, _last_submitted_attempt, _run_in_pgroup, _submit_failure_detail


@contextlib.contextmanager
def _fake_kaggle_home_env():
    yield {"HOME": "/fake/home"}


def _run(attempt_id, cached_csv, submit_result=None):
    """공통 mock 세트로 _kaggle_submit 1회 실행.
    (run_in_pgroup_mock, download_mock, upload_mock) 반환."""
    submit_result = submit_result or MagicMock(returncode=0, stdout="submission saved: /tmp/x.csv\n", stderr="")
    conn_mock = MagicMock()
    with patch("store.db.connect", return_value=conn_mock), \
         patch("bin.api._kaggle_home_env", _fake_kaggle_home_env), \
         patch("store.s3_code.download_submission_csv", return_value=cached_csv) as download_mock, \
         patch("store.s3_code.upload_submission_csv") as upload_mock, \
         patch("bin.api._run_in_pgroup", return_value=submit_result) as run_mock:
        _kaggle_submit(
            submission_id="sub-1",
            competition_id="playground-series-s4e1",
            competition_slug="s4e1",
            attempt_id=attempt_id,
            message="test message",
        )
    return run_mock, download_mock, upload_mock


def test_cache_hit_uploads_csv_directly_without_fit() -> None:
    """캐시 히트 시 kaggle CLI를 직접 호출하고 bin.submit 서브프로세스(fit)는 안 뜬다."""
    run_mock, download_mock, upload_mock = _run(attempt_id="attempt-abc", cached_csv=b"id,target\n1,0.5\n")
    download_mock.assert_called_once_with("playground-series-s4e1", "attempt-abc")
    run_mock.assert_called_once()
    cmd = run_mock.call_args.args[0]
    assert cmd[:2] == ["uv", "run"]
    assert "kaggle" in cmd
    assert "bin.submit" not in cmd
    # 이미 캐시 히트였으니 재업로드(안전망)는 불필요.
    upload_mock.assert_not_called()


def test_cache_miss_falls_back_to_fit_subprocess() -> None:
    """캐시 미스면 기존대로 bin.submit 서브프로세스(fit)를 띄운다."""
    run_mock, download_mock, _ = _run(attempt_id="attempt-abc", cached_csv=None)
    download_mock.assert_called_once_with("playground-series-s4e1", "attempt-abc")
    run_mock.assert_called_once()
    cmd = run_mock.call_args.args[0]
    assert "bin.submit" in cmd
    assert "--attempt-id" in cmd
    assert "attempt-abc" in cmd


def test_cache_miss_fit_result_is_uploaded_to_cache_as_safety_net() -> None:
    """캐시 미스로 ops-vm에서 직접 fit한 경우, 그 결과 CSV를 캐시에 올려야 한다 —
    promote task가 미처 선캐싱하지 못한 타이밍 갭에서도 같은 attempt의 다음 제출부턴
    fit 없이 히트하도록 하는 안전망."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.write(b"id,target\n1,0.5\n")
    tmp.close()
    try:
        submit_result = MagicMock(returncode=0, stdout=f"submission saved: {tmp.name}\n", stderr="")
        _, _, upload_mock = _run(attempt_id="attempt-abc", cached_csv=None, submit_result=submit_result)
        upload_mock.assert_called_once_with(
            "playground-series-s4e1", "attempt-abc", b"id,target\n1,0.5\n"
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def test_no_attempt_id_skips_cache_lookup_entirely() -> None:
    """attempt_id 미지정(수동 제출 'best')이면 캐시 조회 자체를 하지 않고 기존 경로로 간다."""
    run_mock, download_mock, upload_mock = _run(attempt_id=None, cached_csv=None)
    download_mock.assert_not_called()
    cmd = run_mock.call_args.args[0]
    assert "bin.submit" in cmd
    assert "--attempt-id" not in cmd
    # attempt_id가 없으면 캐시 키를 만들 수 없어 안전망 업로드도 스킵.
    upload_mock.assert_not_called()


def test_run_in_pgroup_kills_process_group_on_timeout() -> None:
    """타임아웃 시 os.killpg가 자식의 프로세스 그룹 전체에 호출돼야 한다.

    subprocess.run은 직속 자식(uv)만 kill해 uv가 spawn한 손자 python이 고아로 남았다.
    start_new_session=True로 띄운 뒤 os.killpg(pgid, SIGKILL)로 그룹째 죽이는지 검증.
    """
    fake_proc = MagicMock()
    fake_proc.pid = 4242
    fake_proc.communicate.side_effect = [subprocess.TimeoutExpired(cmd="x", timeout=1), ("", "")]

    with patch("bin.api.subprocess.Popen", return_value=fake_proc) as popen_mock, \
         patch("bin.api.os.getpgid", return_value=4242) as getpgid_mock, \
         patch("bin.api.os.killpg") as killpg_mock:
        try:
            _run_in_pgroup(["uv", "run", "python", "-m", "bin.submit"], timeout=1, cwd="/tmp", env={})
            raised = False
        except subprocess.TimeoutExpired:
            raised = True

    assert raised, "timeout이 호출부로 재-raise되어야 기존 except subprocess.TimeoutExpired 처리가 동작한다"
    popen_mock.assert_called_once()
    assert popen_mock.call_args.kwargs["start_new_session"] is True
    getpgid_mock.assert_called_once_with(4242)
    killpg_mock.assert_called_once_with(4242, signal.SIGKILL)
    assert fake_proc.communicate.call_count == 2  # 최초 시도 + kill 후 reap


def test_run_in_pgroup_returns_completed_process_on_success() -> None:
    """정상 종료 시 기존 subprocess.run 계약(.returncode/.stdout/.stderr)과 동일한 형태를 반환."""
    fake_proc = MagicMock()
    fake_proc.pid = 1
    fake_proc.returncode = 0
    fake_proc.communicate.return_value = ("submission saved: /tmp/x.csv\n", "")

    with patch("bin.api.subprocess.Popen", return_value=fake_proc):
        result = _run_in_pgroup(["uv", "run", "kaggle"], timeout=10, cwd="/tmp", env={})

    assert result.returncode == 0
    assert "submission saved" in result.stdout
    assert result.stderr == ""


def test_last_submitted_attempt_query_excludes_stale_submitted() -> None:
    """'submitted'가 폴링 데드라인(30분)보다 한참(1시간) 지나도 안 풀리면
    재제출 후보에서 제외돼야 한다 — s6e6이 2주+ 'submitted'에 고정돼 매일 자동 제출이
    영구 스킵되던 실제 사례 재발 방지. 쿼리에 staleness 조건이 들어갔는지 확인."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("attempt-x",)
    result = _last_submitted_attempt(conn, "playground-series-s6e6")
    sql = conn.execute.call_args[0][0]
    assert "checked_at" in sql
    assert "'submitted'" in sql
    assert "interval '1 hour'" in sql
    assert result == "attempt-x"


def test_last_submitted_attempt_query_excludes_null_attempt_id() -> None:
    """attempt_id가 NULL인 제출행(수동 probe 등)이 쿼리에서 제외돼야 한다 — 안 그러면
    auto_submit이 last=None으로 착각해 유의성 게이트를 건너뛴다(#198)."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    _last_submitted_attempt(conn, "playground-series-s6e8")
    sql = conn.execute.call_args[0][0]
    assert "attempt_id is not null" in sql


def test_cache_miss_fit_is_serialized_by_gate() -> None:
    """일일 예산이 대회당 2건이 되면서(ADR-038) 같은 대회 후보 2개가 동시에 전체 train을
    fit해 서로를 굶겨 둘 다 타임아웃난 실측(2026-08-26 s6e8) — fit 경로는 한 번에 하나만."""
    import threading

    import bin.api as api_mod

    seen: list[list[str]] = []

    def _record(cmd, **kw):
        seen.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    conn_mock = MagicMock()
    with patch("store.db.connect", return_value=conn_mock), \
         patch("bin.api._kaggle_home_env", _fake_kaggle_home_env), \
         patch("store.s3_code.download_submission_csv", return_value=None), \
         patch("store.s3_code.upload_submission_csv"), \
         patch("bin.api._run_in_pgroup", side_effect=_record):
        assert api_mod._submit_fit_gate.acquire(blocking=False)
        t = threading.Thread(target=_kaggle_submit, args=(
            "sub-1", "playground-series-s4e1", "s4e1", "attempt-abc", "msg",
        ), daemon=True)
        t.start()
        t.join(timeout=1.0)
        # 게이트를 다른 쪽이 잡고 있는 동안엔 fit 서브프로세스가 시작되지 않는다.
        assert seen == []
        assert t.is_alive()

        api_mod._submit_fit_gate.release()
        t.join(timeout=5.0)

    assert not t.is_alive()
    assert any("bin.submit" in c for c in seen)


def test_cache_hit_does_not_take_fit_gate() -> None:
    """캐시 히트는 fit이 없어 직렬화 대상이 아니다 — 게이트가 잠겨 있어도 바로 업로드한다."""
    import bin.api as api_mod

    assert api_mod._submit_fit_gate.acquire(blocking=False)
    try:
        run_mock, _, _ = _run(attempt_id="attempt-abc", cached_csv=b"id,target\n1,0.5\n")
        run_mock.assert_called_once()
    finally:
        api_mod._submit_fit_gate.release()


def test_submit_failure_detail_keeps_stderr_tail_not_head() -> None:
    """chatty warning이 stderr 앞을 채워도 뒤쪽 진짜 예외가 살아남아야 한다 —
    2026-08-26에 "Patch member collision"으로 하루 오진한 절단 함정."""
    warnings = "Patch member collision (patch overrides base): ['build_model']\n" * 60
    real_exc = "RuntimeError: replay_best_pipeline: 재생 결과 sha256이 마지막 승격분의 신뢰 해시와 다르다"
    result = MagicMock(stdout="", stderr=warnings + real_exc)
    detail = _submit_failure_detail(result)
    assert real_exc in detail
    assert len(detail) <= 2000


def test_submit_failure_detail_prepends_base_origin_line() -> None:
    """stdout의 base 로드 결과 줄을 복구해 앞에 붙인다 — 어느 base 경로를 탔는지가 1차 진단."""
    result = MagicMock(
        stdout="best attempt: 1034d895  cv=0.95986\nbase pipeline loaded: replay fallback (3 promoted pipeline(s), sha verified)\n",
        stderr="RuntimeError: boom",
    )
    detail = _submit_failure_detail(result)
    assert detail.startswith("base pipeline loaded: replay fallback")
    assert "RuntimeError: boom" in detail


def test_submit_failure_detail_no_origin_line_when_base_load_raised() -> None:
    """load_base_snapshot이 raise하면 origin 줄이 안 찍히고, 그 부재가 곧 신호다."""
    result = MagicMock(stdout="best attempt: abc  cv=0.5\n", stderr="RuntimeError: snapshot corrupt")
    detail = _submit_failure_detail(result)
    assert detail == "RuntimeError: snapshot corrupt"


def test_submit_failure_writes_detail_to_error_column() -> None:
    """returncode != 0이면 _submit_failure_detail 결과가 raw.kaggle_submissions.error로 들어간다."""
    conn_mock = MagicMock()
    fail = MagicMock(
        returncode=1,
        stdout="base pipeline loaded: materialized_code snapshot\n",
        stderr="W\n" * 2000 + "RuntimeError: real cause here",
    )
    with patch("store.db.connect", return_value=conn_mock), \
         patch("bin.api._kaggle_home_env", _fake_kaggle_home_env), \
         patch("store.s3_code.download_submission_csv", return_value=None), \
         patch("store.s3_code.upload_submission_csv"), \
         patch("bin.api._run_in_pgroup", return_value=fail):
        _kaggle_submit("sub-1", "playground-series-s4e1", "s4e1", "attempt-abc", "msg")

    error_updates = [
        c.args[1] for c in conn_mock.execute.call_args_list
        if "set status = %s, error = %s" in c.args[0]
    ]
    assert error_updates, "error 업데이트가 실행돼야 한다"
    _status, err, *_ = error_updates[-1]
    assert "RuntimeError: real cause here" in err
    assert err.startswith("base pipeline loaded: materialized_code snapshot")
