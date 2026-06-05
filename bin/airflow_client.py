"""Airflow 3 REST API thin client — DAG trigger + polling.

환경변수:
    AIRFLOW_URL       Airflow api-server URL (e.g. http://<ops-tailnet-ip>:8080)
    AIRFLOW_USER      Airflow 사용자 (기본 admin)
    AIRFLOW_PASSWORD  Airflow 비밀번호
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import requests

_AIRFLOW_URL = os.getenv("AIRFLOW_URL", "").rstrip("/")
_AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
_AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "")

DAG_ID = "reflexion_rondo_cycle"
_TERMINAL = {"success", "failed", "cancelled"}


def available() -> bool:
    return bool(_AIRFLOW_URL)


def _auth() -> tuple[str, str]:
    return (_AIRFLOW_USER, _AIRFLOW_PASSWORD)


def trigger_dag_run(competition_id: str, stage: str, queue_id: str) -> str:
    """DAG run 1개 트리거. dag_run_id 반환."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"rondo_{queue_id[:8]}_{ts}"
    resp = requests.post(
        f"{_AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns",
        json={
            "dag_run_id": run_id,
            "conf": {
                "competition_id": competition_id,
                "stage": stage,
                "queue_id": queue_id,
            },
        },
        auth=_auth(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["dag_run_id"]


def get_dag_run_state(dag_run_id: str) -> str:
    resp = requests.get(
        f"{_AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns/{dag_run_id}",
        auth=_auth(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("state", "")


def wait_for_dag_run(
    dag_run_id: str,
    poll_interval: int = 15,
    timeout: int = 3600,
) -> str:
    """terminal 상태가 될 때까지 폴링. 최종 state 반환 (success/failed/cancelled/timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = get_dag_run_state(dag_run_id)
        if state in _TERMINAL:
            return state
        time.sleep(poll_interval)
    return "timeout"
