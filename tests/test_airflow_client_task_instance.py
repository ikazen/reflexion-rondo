"""bin/airflow_client.py — task instance 폴링 (#204).

daemon이 DAG run 전체가 아니라 promote task instance 하나의 완료만 기다리도록 전환하는
근거. get_task_instance_state가 404(아직 안 잡힘)와 state=null(아직 안 스케줄됨)을 모두
non-terminal("")로 정규화하는지, wait_for_task_instance가 terminal 상태에서 즉시 반환하고
timeout에서 "timeout"을 주는지 검증한다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import bin.airflow_client as airflow_client


def _resp(status_code=200, json_body=None):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_body or {}
    r.raise_for_status = MagicMock()
    return r


def test_get_task_instance_state_normalizes_404_to_empty():
    with (
        patch("bin.airflow_client._headers", return_value={}),
        patch("bin.airflow_client.requests.get", return_value=_resp(status_code=404)),
    ):
        assert airflow_client.get_task_instance_state("run-1", "promote") == ""


def test_get_task_instance_state_normalizes_null_state_to_empty():
    """아직 스케줄 안 된 task는 200 + state:null을 준다 — non-terminal로 취급해야 한다."""
    with (
        patch("bin.airflow_client._headers", return_value={}),
        patch("bin.airflow_client.requests.get", return_value=_resp(json_body={"state": None})),
    ):
        assert airflow_client.get_task_instance_state("run-1", "promote") == ""


def test_get_task_instance_state_returns_actual_state():
    with (
        patch("bin.airflow_client._headers", return_value={}),
        patch("bin.airflow_client.requests.get", return_value=_resp(json_body={"state": "success"})),
    ):
        assert airflow_client.get_task_instance_state("run-1", "promote") == "success"


def test_wait_for_task_instance_returns_on_terminal_state(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with patch("bin.airflow_client.get_task_instance_state", side_effect=["", "running", "success"]):
        result = airflow_client.wait_for_task_instance("run-1", "promote", poll_interval=0)
    assert result == "success"


def test_wait_for_task_instance_times_out(monkeypatch):
    fake_time = [0.0]
    monkeypatch.setattr("time.time", lambda: fake_time[0])

    def fake_sleep(_):
        fake_time[0] += 100

    monkeypatch.setattr("time.sleep", fake_sleep)
    with patch("bin.airflow_client.get_task_instance_state", return_value="running"):
        result = airflow_client.wait_for_task_instance("run-1", "promote", poll_interval=15, timeout=50)
    assert result == "timeout"
