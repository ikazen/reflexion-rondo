"""bin.run_daemon._final_status의 사이클 배치 결과 -> 최종 상태 매핑 단위 테스트."""
from unittest.mock import patch

import pytest

from bin.run_daemon import _final_status, main


def test_all_success():
    assert _final_status(successes=3, failed_cycles=0) == ("done", None)


def test_partial_failure_still_done():
    assert _final_status(successes=2, failed_cycles=1) == ("done", None)


def test_all_failed():
    status, err = _final_status(successes=0, failed_cycles=3)
    assert status == "failed"
    assert "3 failed" in err


def test_empty_batch():
    assert _final_status(successes=0, failed_cycles=0) == ("done", None)


def test_main_refuses_to_start_without_airflow():
    """AIRFLOW_URL이 없으면 조용히 다른 모드로 도는 대신 시작을 거부한다(direct 모드 제거, #474)."""
    with (
        patch("config.settings.require_llm_env"),
        patch("bin.run_daemon.airflow_client.available", return_value=False),
        patch("bin.run_daemon.connect") as mock_connect,
    ):
        with pytest.raises(SystemExit, match="AIRFLOW_URL"):
            main()
    mock_connect.assert_not_called()
