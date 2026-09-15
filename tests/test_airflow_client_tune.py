"""bin/airflow_client.py — Optuna 튜닝 DAG(reflexion_rondo_tune) 트리거 (#318).

conf 계약(competition=slug, n_trials, 선택적 timeout_sec)이 reflexion_rondo_cycle
트리거(competition_id, stage, queue_id)와 다르므로 별도로 검증한다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import bin.airflow_client as airflow_client


def _resp(json_body=None):
    r = MagicMock()
    r.json.return_value = json_body or {"dag_run_id": "rondo_tune_s6e8_20260915T000000"}
    r.raise_for_status = MagicMock()
    return r


def test_trigger_tune_dag_run_posts_to_tune_dag_with_default_trials():
    with (
        patch("bin.airflow_client._headers", return_value={}),
        patch("bin.airflow_client.requests.post", return_value=_resp()) as mock_post,
    ):
        run_id = airflow_client.trigger_tune_dag_run("s6e8")
    assert run_id == "rondo_tune_s6e8_20260915T000000"
    url = mock_post.call_args.args[0]
    assert "/dags/reflexion_rondo_tune/dagRuns" in url
    conf = mock_post.call_args.kwargs["json"]["conf"]
    assert conf == {"competition": "s6e8", "n_trials": 100}


def test_trigger_tune_dag_run_includes_timeout_sec_when_given():
    with (
        patch("bin.airflow_client._headers", return_value={}),
        patch("bin.airflow_client.requests.post", return_value=_resp()) as mock_post,
    ):
        airflow_client.trigger_tune_dag_run("s5e2", n_trials=50, timeout_sec=1800)
    conf = mock_post.call_args.kwargs["json"]["conf"]
    assert conf == {"competition": "s5e2", "n_trials": 50, "timeout_sec": 1800}


def test_trigger_tune_dag_run_omits_timeout_sec_when_not_given():
    """timeout_sec 생략 시 bin/tune_pipeline.py 자체 기본값(무제한, DAG
    execution_timeout이 바깥 상한)을 쓰게 conf에 키 자체를 안 넣는다."""
    with (
        patch("bin.airflow_client._headers", return_value={}),
        patch("bin.airflow_client.requests.post", return_value=_resp()) as mock_post,
    ):
        airflow_client.trigger_tune_dag_run("s6e8")
    conf = mock_post.call_args.kwargs["json"]["conf"]
    assert "timeout_sec" not in conf
