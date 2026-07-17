"""GH issue #31: _kaggle_submit 캐시 우선 경로.

promote 시점에 캐싱된 submission CSV(store.s3_code.download_submission_csv)가 있으면
fit(bin.submit 서브프로세스) 없이 캐시된 CSV를 바로 kaggle CLI로 업로드해야 한다 —
매일 06:00 auto-submit이 ops-vm(daemon 상주, 2 OCPU)에서 매번 fit하며 CPU를 포화시키던
경로를 대체하는 것이 이번 변경의 목적. 캐시 미스 시에는 기존 bin.submit 경로로 폴백한다.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.api import _kaggle_submit


@contextlib.contextmanager
def _fake_kaggle_home_env():
    yield {"HOME": "/fake/home"}


def _run(attempt_id, cached_csv, submit_result=None):
    """공통 mock 세트로 _kaggle_submit 1회 실행. (subprocess_run_mock, download_mock) 반환."""
    submit_result = submit_result or MagicMock(returncode=0, stdout="submission saved: /tmp/x.csv\n", stderr="")
    conn_mock = MagicMock()
    with patch("store.db.connect", return_value=conn_mock), \
         patch("bin.api._kaggle_home_env", _fake_kaggle_home_env), \
         patch("store.s3_code.download_submission_csv", return_value=cached_csv) as download_mock, \
         patch("subprocess.run", return_value=submit_result) as run_mock:
        _kaggle_submit(
            submission_id="sub-1",
            competition_id="playground-series-s4e1",
            competition_slug="s4e1",
            attempt_id=attempt_id,
            message="test message",
        )
    return run_mock, download_mock


def test_cache_hit_uploads_csv_directly_without_fit() -> None:
    """캐시 히트 시 kaggle CLI를 직접 호출하고 bin.submit 서브프로세스(fit)는 안 뜬다."""
    run_mock, download_mock = _run(attempt_id="attempt-abc", cached_csv=b"id,target\n1,0.5\n")
    download_mock.assert_called_once_with("playground-series-s4e1", "attempt-abc")
    run_mock.assert_called_once()
    cmd = run_mock.call_args.args[0]
    assert cmd[:2] == ["uv", "run"]
    assert "kaggle" in cmd
    assert "bin.submit" not in cmd


def test_cache_miss_falls_back_to_fit_subprocess() -> None:
    """캐시 미스면 기존대로 bin.submit 서브프로세스(fit)를 띄운다."""
    run_mock, download_mock = _run(attempt_id="attempt-abc", cached_csv=None)
    download_mock.assert_called_once_with("playground-series-s4e1", "attempt-abc")
    run_mock.assert_called_once()
    cmd = run_mock.call_args.args[0]
    assert "bin.submit" in cmd
    assert "--attempt-id" in cmd
    assert "attempt-abc" in cmd


def test_no_attempt_id_skips_cache_lookup_entirely() -> None:
    """attempt_id 미지정(수동 제출 'best')이면 캐시 조회 자체를 하지 않고 기존 경로로 간다."""
    run_mock, download_mock = _run(attempt_id=None, cached_csv=None)
    download_mock.assert_not_called()
    cmd = run_mock.call_args.args[0]
    assert "bin.submit" in cmd
    assert "--attempt-id" not in cmd
