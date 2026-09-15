"""bin/run_daemon.py — Optuna 튜닝 레인 자동 트리거 (#318).

주 트리거(_maybe_trigger_tune, 승격 성공 직후)와 보조 트리거(_sweep_idle_tuning,
48h 이상 튜닝 안 된 ACTIVE 대회)를 각각 검증한다. 둘 다 best-effort — 트리거
실패가 daemon 사이클/루프 자체를 죽이면 안 된다.
"""
from __future__ import annotations

import datetime as dt
import time
from unittest.mock import MagicMock, patch

import bin.run_daemon as run_daemon
from bin.run_daemon import (
    _TUNE_SWEEP_INTERVAL_SEC,
    _maybe_trigger_tune,
    _sweep_idle_tuning,
)


def _long_ago() -> float:
    return time.monotonic() - _TUNE_SWEEP_INTERVAL_SEC - 1


# ---- _maybe_trigger_tune (주 트리거) ----


def test_maybe_trigger_tune_skips_when_airflow_unavailable():
    conn = MagicMock()
    with patch("bin.run_daemon.airflow_client.available", return_value=False):
        _maybe_trigger_tune(conn, "s6e8", "playground-series-s6e8", "attempt-1")
    conn.execute.assert_not_called()


def test_maybe_trigger_tune_skips_when_attempt_not_a_confirmed_pipeline():
    """was_promoted=True는 승자 선정 단계에서 전부 찍히므로(run_promote_task.py) 실제
    확정(merge-verify 통과, raw.pipelines 존재) 여부를 별도로 확인해야 한다."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    with (
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_tune_dag_run") as mock_trigger,
    ):
        _maybe_trigger_tune(conn, "s6e8", "playground-series-s6e8", "attempt-1")
    mock_trigger.assert_not_called()


def test_maybe_trigger_tune_triggers_when_pipeline_confirmed():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (1,)
    with (
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_tune_dag_run", return_value="run-1") as mock_trigger,
    ):
        _maybe_trigger_tune(conn, "s6e8", "playground-series-s6e8", "attempt-1")
    mock_trigger.assert_called_once_with("s6e8")


def test_maybe_trigger_tune_swallows_trigger_exception():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (1,)
    with (
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_tune_dag_run", side_effect=RuntimeError("boom")),
    ):
        _maybe_trigger_tune(conn, "s6e8", "playground-series-s6e8", "attempt-1")  # 예외 전파되면 실패


# ---- _sweep_idle_tuning (보조 트리거) ----


def _patch_scan(comp_slugs: dict[str, str], active: set[str] | None = None):
    return (
        patch("bin.run_daemon.competition_id_to_slug", return_value=comp_slugs),
        patch(
            "bin.run_daemon.active_competition_ids",
            return_value=set(comp_slugs) if active is None else active,
        ),
    )


def test_idle_tuning_sweep_respects_rate_gate(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_tune_sweep", time.monotonic())
    conn = MagicMock()
    with patch("bin.run_daemon.airflow_client.available", return_value=True):
        _sweep_idle_tuning(conn)
    conn.execute.assert_not_called()


def test_idle_tuning_sweep_skips_when_airflow_unavailable(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_tune_sweep", _long_ago())
    conn = MagicMock()
    with patch("bin.run_daemon.airflow_client.available", return_value=False):
        _sweep_idle_tuning(conn)
    conn.execute.assert_not_called()


def test_idle_tuning_sweep_skips_competition_with_no_confirmed_pipeline(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_tune_sweep", _long_ago())
    conn = MagicMock()
    conn.execute.return_value.fetchall.side_effect = [
        [],  # last_tune: 아무 기록 없음
        [],  # has_confirmed: 확정 pipeline 있는 대회 없음
    ]
    comp_slugs = {"playground-series-s6e8": "s6e8"}
    scan, active = _patch_scan(comp_slugs)
    with (
        scan, active,
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_tune_dag_run") as mock_trigger,
    ):
        _sweep_idle_tuning(conn)
    mock_trigger.assert_not_called()


def test_idle_tuning_sweep_skips_recently_tuned_competition(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_tune_sweep", _long_ago())
    conn = MagicMock()
    recent = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=1)
    conn.execute.return_value.fetchall.side_effect = [
        [("playground-series-s6e8", recent)],  # 1시간 전 튜닝 — 아직 idle 아님
        [("playground-series-s6e8",)],  # 확정 pipeline 있음
    ]
    comp_slugs = {"playground-series-s6e8": "s6e8"}
    scan, active = _patch_scan(comp_slugs)
    with (
        scan, active,
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_tune_dag_run") as mock_trigger,
    ):
        _sweep_idle_tuning(conn)
    mock_trigger.assert_not_called()


def test_idle_tuning_sweep_triggers_stale_or_never_tuned_confirmed_competitions(monkeypatch):
    """s6e8: 확정 pipeline 있고 마지막 튜닝이 48h보다 오래됨 → 트리거.
    s5e2: 확정 pipeline 있고 튜닝 이력 자체 없음(신규 편입) → 트리거.
    s4e1: 확정 pipeline 자체 없음 → 스킵."""
    monkeypatch.setattr(run_daemon, "_last_tune_sweep", _long_ago())
    conn = MagicMock()
    stale = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=100)
    conn.execute.return_value.fetchall.side_effect = [
        [("playground-series-s6e8", stale)],  # last_tune — s5e2/s4e1은 기록 없음
        [("playground-series-s6e8",), ("playground-series-s5e2",)],  # has_confirmed
    ]
    comp_slugs = {
        "playground-series-s6e8": "s6e8",
        "playground-series-s5e2": "s5e2",
        "playground-series-s4e1": "s4e1",
    }
    scan, active = _patch_scan(comp_slugs)
    with (
        scan, active,
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch("bin.run_daemon.airflow_client.trigger_tune_dag_run", return_value="run-1") as mock_trigger,
    ):
        _sweep_idle_tuning(conn)
    triggered_slugs = {c.args[0] for c in mock_trigger.call_args_list}
    assert triggered_slugs == {"s6e8", "s5e2"}


def test_idle_tuning_sweep_swallows_trigger_exception_and_continues(monkeypatch):
    """한 대회 트리거가 실패해도 나머지 대회는 계속 처리한다."""
    monkeypatch.setattr(run_daemon, "_last_tune_sweep", _long_ago())
    conn = MagicMock()
    conn.execute.return_value.fetchall.side_effect = [
        [],  # last_tune 없음 — 둘 다 신규 취급
        [("playground-series-s6e8",), ("playground-series-s5e2",)],  # has_confirmed
    ]
    comp_slugs = {"playground-series-s6e8": "s6e8", "playground-series-s5e2": "s5e2"}
    scan, active = _patch_scan(comp_slugs)
    with (
        scan, active,
        patch("bin.run_daemon.airflow_client.available", return_value=True),
        patch(
            "bin.run_daemon.airflow_client.trigger_tune_dag_run",
            side_effect=[RuntimeError("boom"), "run-2"],
        ) as mock_trigger,
    ):
        _sweep_idle_tuning(conn)  # 예외가 여기까지 전파되면 실패
    assert mock_trigger.call_count == 2
