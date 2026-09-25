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
_TUNE_DAG_ID = "reflexion_rondo_tune"
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


def trigger_tune_dag_run(
    competition_slug: str,
    n_trials: int = 100,
    timeout_sec: int | None = None,
) -> str:
    """Optuna 튜닝 DAG(reflexion_rondo_tune) 1개 트리거. dag_run_id 반환.

    reflexion_rondo_cycle과 별도 DAG — conf 계약이 다르다(competition은
    COMPETITION_ID가 아니라 config 모듈 slug, 예: "s6e8". airflow-stack
    dags/reflexion_rondo_tune.py 참고). fire-and-forget — 완료를 기다리지
    않는다(별도 컴퓨트 레인이라 promote task/daemon 사이클을 막을 이유가
    없다, #318).

    logical_date는 마이크로초까지 포함한다 — 초 단위였을 때 `_sweep_idle_tuning`이
    같은 스윕 틱에서 대회 여러 개를 연달아 트리거하면(밀리초 간격) 같은 초에 몰려
    Airflow의 (dag_id, logical_date) 유일성 제약에 걸려 두 번째부터 전부 409
    Conflict로 실패했다(실측: s5e2/s6e8 동시 ACTIVE 상태에서 알파벳순으로 먼저
    도는 s5e2는 매번 성공, s6e8은 24회 연속 100% 실패).
    """
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S%f")
    run_id = f"rondo_tune_{competition_slug}_{ts}"
    conf: dict = {"competition": competition_slug, "n_trials": n_trials}
    if timeout_sec is not None:
        conf["timeout_sec"] = timeout_sec
    resp = requests.post(
        f"{_AIRFLOW_URL}/api/v2/dags/{_TUNE_DAG_ID}/dagRuns",
        json={
            "dag_run_id": run_id,
            "logical_date": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "conf": conf,
        },
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["dag_run_id"]


def tune_run_in_flight(competition_slug: str) -> bool:
    """raw.tuned_params는 런이 끝나야(약 3h) 갱신돼 DB만으로는 진행 중인 런을 알 수 없다(#360).
    수동 트리거는 dag_run_id 형식이 달라 conf도 함께 본다."""
    resp = requests.get(
        f"{_AIRFLOW_URL}/api/v2/dags/{_TUNE_DAG_ID}/dagRuns",
        params=[("state", "queued"), ("state", "running"), ("limit", 100)],
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    prefix = f"rondo_tune_{competition_slug}_"
    return any(
        run["dag_run_id"].startswith(prefix) or (run.get("conf") or {}).get("competition") == competition_slug
        for run in resp.json().get("dag_runs", [])
    )


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
