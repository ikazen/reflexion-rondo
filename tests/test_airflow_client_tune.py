"""bin/airflow_client.py — Optuna 튜닝 DAG(reflexion_rondo_tune) 트리거 (#318)와 진행 중 런 조회 (#360).

conf 계약(competition=slug, n_trials, 선택적 timeout_sec)이 reflexion_rondo_cycle
트리거(competition_id, stage, queue_id)와 다르므로 별도로 검증한다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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


def test_trigger_tune_dag_run_back_to_back_calls_do_not_collide():
    """실측 재현(#318 배포 후 회귀): _sweep_idle_tuning이 같은 스윕 틱에서 대회
    여러 개를 연달아 트리거하면 밀리초 간격이라 초 단위 logical_date로는 같은
    초에 몰려 Airflow (dag_id, logical_date) 유일성 제약에 걸린다 — 두 번째
    호출부터 100% 409 Conflict(24회 연속 실측). 마이크로초까지 포함하면
    같은 초 안에서도 각 호출이 서로 다른 logical_date를 받아야 한다."""
    posts = []
    with (
        patch("bin.airflow_client._headers", return_value={}),
        patch("bin.airflow_client.requests.post", return_value=_resp()) as mock_post,
    ):
        airflow_client.trigger_tune_dag_run("s5e2")
        airflow_client.trigger_tune_dag_run("s6e8")
    logical_dates = [c.kwargs["json"]["logical_date"] for c in mock_post.call_args_list]
    run_ids = [c.kwargs["json"]["dag_run_id"] for c in mock_post.call_args_list]
    assert logical_dates[0] != logical_dates[1]
    assert run_ids[0] != run_ids[1]
    # 마이크로초 성분이 실제로 들어있는지(초 단위로 되돌아가는 회귀 방지)
    assert "." in logical_dates[0] and logical_dates[0].endswith("Z")


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


def _runs(*runs: dict) -> MagicMock:
    return _resp({"dag_runs": list(runs), "total_entries": len(runs)})


def _in_flight(slug: str, resp: MagicMock) -> bool:
    with (
        patch("bin.airflow_client._headers", return_value={}),
        patch("bin.airflow_client.requests.get", return_value=resp),
    ):
        return airflow_client.tune_run_in_flight(slug)


def test_tune_run_in_flight_queries_queued_and_running_runs_of_the_tune_dag():
    with (
        patch("bin.airflow_client._headers", return_value={}),
        patch("bin.airflow_client.requests.get", return_value=_runs()) as mock_get,
    ):
        airflow_client.tune_run_in_flight("s5e4")
    assert mock_get.call_args.args[0].endswith("/api/v2/dags/reflexion_rondo_tune/dagRuns")
    params = mock_get.call_args.kwargs["params"]
    assert ("state", "queued") in params and ("state", "running") in params


def test_tune_run_in_flight_true_when_run_id_carries_the_slug():
    resp = _runs({"dag_run_id": "rondo_tune_s5e4_20260924T011100123456", "conf": {"competition": "s5e4"}})
    assert _in_flight("s5e4", resp) is True


def test_tune_run_in_flight_false_for_other_competitions_and_no_runs():
    resp = _runs({"dag_run_id": "rondo_tune_s6e8_20260924T011100123456", "conf": {"competition": "s6e8"}})
    assert _in_flight("s5e4", resp) is False
    assert _in_flight("s5e4", _runs()) is False
    assert _in_flight("s5e4", _resp({})) is False


def test_tune_run_in_flight_does_not_confuse_slugs_sharing_a_prefix():
    """s4e1과 s4e11/s4e12는 실제로 공존한다 — 접두 비교는 구분자(_)까지 포함해야 한다."""
    resp = _runs(
        {"dag_run_id": "rondo_tune_s4e11_20260924T011100123456", "conf": {"competition": "s4e11"}},
        {"dag_run_id": "rondo_tune_s4e12_20260924T011100123456", "conf": None},
    )
    assert _in_flight("s4e1", resp) is False


def test_tune_run_in_flight_matches_manually_triggered_run_by_conf():
    """수동 트리거는 dag_run_id 형식이 다를 수 있어 conf.competition도 본다."""
    resp = _runs({"dag_run_id": "manual__2026-09-24T01:11:00+00:00", "conf": {"competition": "s5e4"}})
    assert _in_flight("s5e4", resp) is True


def test_tune_run_in_flight_propagates_http_errors():
    """호출부(daemon)가 예외를 받아 이번 스윕을 건너뛰어야 하므로 삼키지 않는다."""
    resp = _runs()
    resp.raise_for_status.side_effect = RuntimeError("503")
    with pytest.raises(RuntimeError, match="503"):
        _in_flight("s5e4", resp)
