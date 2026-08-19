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
_TERMINAL = {"success", "failed", "cancelled"}  # dag_run state
_TI_TERMINAL = {"success", "failed", "upstream_failed", "skipped", "removed"}  # task instance state

_token: str | None = None
_token_expires: float = 0.0


def available() -> bool:
    return bool(_AIRFLOW_URL)


def _bearer_token() -> str:
    global _token, _token_expires
    if _token is None or time.time() > _token_expires - 300:
        resp = requests.post(
            f"{_AIRFLOW_URL}/auth/token",
            json={"username": _AIRFLOW_USER, "password": _AIRFLOW_PASSWORD},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        _token = data["access_token"]
        # Airflow 기본 TTL 24h — 보수적으로 23h 캐시
        _token_expires = time.time() + 23 * 3600
    return _token


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_bearer_token()}"}


def trigger_dag_run(competition_id: str, stage: str, queue_id: str) -> str:
    """DAG run 1개 트리거. dag_run_id 반환."""
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S")
    run_id = f"rondo_{queue_id[:8]}_{ts}"
    resp = requests.post(
        f"{_AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns",
        json={
            "dag_run_id": run_id,
            "logical_date": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "conf": {
                "competition_id": competition_id,
                "stage": stage,
                "queue_id": queue_id,
            },
        },
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["dag_run_id"]


def get_dag_run_state(dag_run_id: str) -> str:
    resp = requests.get(
        f"{_AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns/{dag_run_id}",
        headers=_headers(),
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


def get_task_instance_state(dag_run_id: str, task_id: str) -> str:
    """단일 task instance state. 아직 스케줄 안 됐으면(state=null) 또는 아예 아직 DB에
    안 잡혔으면(404, dag_run 트리거 직후 레이스) 둘 다 빈 문자열(non-terminal)로 정규화 —
    호출부가 예외 처리 없이 그대로 폴링 루프에 쓸 수 있게 한다."""
    resp = requests.get(
        f"{_AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns/{dag_run_id}/taskInstances/{task_id}",
        headers=_headers(),
        timeout=15,
    )
    if resp.status_code == 404:
        return ""
    resp.raise_for_status()
    return resp.json().get("state") or ""


def wait_for_task_instance(
    dag_run_id: str,
    task_id: str = "promote",
    poll_interval: int = 15,
    timeout: int = 3600,
) -> str:
    """attempt_gate(#203) 도입 후 daemon은 DAG run 전체가 아니라 promote task 하나의
    완료만 기다리면 된다 — straggler attempt는 DAG run을 계속 running 상태로 붙잡지만
    (#204), 다음 사이클을 시작하는 데는 더 이상 걸림돌이 아니다."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = get_task_instance_state(dag_run_id, task_id)
        if state in _TI_TERMINAL:
            return state
        time.sleep(poll_interval)
    return "timeout"
