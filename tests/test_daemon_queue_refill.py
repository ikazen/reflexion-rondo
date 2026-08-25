"""bin/run_daemon.py — cycle_queue 고갈 시 자동 재보급 스윕 (#196).

2026-08-17~18 실측: 큐가 완전히 비면 daemon이 idle에 멈추고 사람이 enqueue할
때까지 27시간 attempt 0건이었다. _sweep_queue_refill이 pending/running이
하나도 없을 때만, 오래 idle한(또는 한 번도 안 돈) 대회를 재큐잉하는지 검증한다.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import bin.run_daemon as run_daemon
from bin.run_daemon import _QUEUE_REFILL_SWEEP_INTERVAL_SEC, _sweep_queue_refill


def _long_ago() -> float:
    return time.monotonic() - _QUEUE_REFILL_SWEEP_INTERVAL_SEC - 1


def _patch_scan(comp_slugs: dict[str, str], active: set[str] | None = None):
    """idle-detection 로직 자체를 검증하는 테스트는 ACTIVE 필터(#227)와 무관하게
    통과해야 하므로, 실제 config/competitions/*.py 값에 기대는 대신 스캔을 스텁한다."""
    return (
        patch("bin.run_daemon.competition_id_to_slug", return_value=comp_slugs),
        patch(
            "bin.run_daemon.active_competition_ids",
            return_value=set(comp_slugs) if active is None else active,
        ),
    )


def test_sweep_respects_rate_gate(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_queue_refill_sweep", time.monotonic())
    conn = MagicMock()
    _sweep_queue_refill(conn)
    conn.execute.assert_not_called()


def test_sweep_skips_when_queue_not_empty(monkeypatch):
    """pending/running이 하나라도 있으면 재큐잉하지 않는다 — 정상 순환 중에
    끼어들어 우선순위를 어지럽히면 안 된다."""
    monkeypatch.setattr(run_daemon, "_last_queue_refill_sweep", _long_ago())
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (1,)
    with patch("bin.run_daemon.competition_id_to_slug") as mock_scan:
        _sweep_queue_refill(conn)
    mock_scan.assert_not_called()


def test_sweep_reenqueues_idle_and_never_run_competitions(monkeypatch):
    """max(run_ts)가 임계값보다 오래됐거나(idle) attempt가 아예 없던(신규) 대회만
    재큐잉하고, 최근에 돈 대회는 건드리지 않는다."""
    monkeypatch.setattr(run_daemon, "_last_queue_refill_sweep", _long_ago())
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None  # 큐 비어있음

    import datetime as dt
    # raw.attempts.run_ts는 naive(#223) — psycopg2 실반환 형태. 프로덕션 idle_cutoff는
    # UTC 기준(datetime.now(timezone.utc) - 6h)이므로 fixture도 로컬 datetime.now()가
    # 아니라 naive UTC로 만들어야 한다 — 안 그러면 UTC보다 6시간 이상 뒤처진 타임존
    # (미국 서부 등)에서 이 테스트가 허위로 실패한다(#239, adversarial review — #223을
    # 막으려던 테스트 파일에서 같은 클래스의 타임존 버그가 재발할 뻔함).
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    conn.execute.return_value.fetchall.return_value = [
        ("playground-series-s6e8", now),  # 방금 돔 — idle 아님
        ("playground-series-s6e1", now - dt.timedelta(hours=100)),  # idle
    ]

    # bin.api._competition_id_to_slug()의 실제 반환 형태({competition_id: slug}).
    comp_slugs = {
        "playground-series-s6e8": "s6e8",
        "playground-series-s6e1": "s6e1",
        "playground-series-s6e2": "s6e2",
    }
    scan, active = _patch_scan(comp_slugs)
    with scan, active:
        _sweep_queue_refill(conn)

    insert_calls = [
        c for c in conn.execute.call_args_list
        if "insert into raw.cycle_queue" in c.args[0]
    ]
    inserted_slugs = {c.args[1][1] for c in insert_calls}
    # s6e1(idle 100h)과 s6e2(never run, max(run_ts) 자체가 없음)는 재큐잉,
    # s6e8(방금 돔)은 제외.
    assert inserted_slugs == {"s6e1", "s6e2"}


def test_sweep_skips_inactive_competitions(monkeypatch):
    """#227(Milestone v1.6.0 fleet 동결): ACTIVE=False 대회는 idle/never-run이어도
    재큐잉 대상에서 제외한다 — 동결된 대회에 daemon이 다시 컴퓨트를 태우면 안 된다."""
    monkeypatch.setattr(run_daemon, "_last_queue_refill_sweep", _long_ago())
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None  # 큐 비어있음
    conn.execute.return_value.fetchall.return_value = []  # 둘 다 attempt 이력 없음(신규 취급)

    comp_slugs = {
        "playground-series-s6e8": "s6e8",   # ACTIVE=True
        "playground-series-s4e1": "s4e1",   # ACTIVE=False
    }

    scan, active = _patch_scan(comp_slugs, active={"playground-series-s6e8"})
    with scan, active:
        _sweep_queue_refill(conn)

    insert_calls = [
        c for c in conn.execute.call_args_list
        if "insert into raw.cycle_queue" in c.args[0]
    ]
    inserted_slugs = {c.args[1][1] for c in insert_calls}
    assert inserted_slugs == {"s6e8"}


def test_active_scan_skips_slug_when_import_fails(monkeypatch, tmp_path):
    """#239(adversarial review): 대회 config 하나의 import가 실패해도(파일 삭제·오탈자)
    그 슬러그만 건너뛰고 나머지는 정상 반환해야 한다 — daemon 메인 루프 전체가 죽는
    #223과 같은 클래스의 크래시를 이 스캔이 재도입하면 안 된다."""
    import config.competitions as comps

    def _import(name: str):
        slug = name.rsplit(".", 1)[-1]
        if slug == "s4e1":
            raise ModuleNotFoundError(f"no module named {name!r}")
        return type("StubComp", (), {"COMPETITION_ID": f"playground-series-{slug}", "ACTIVE": True})

    monkeypatch.setattr(comps, "importlib", MagicMock(import_module=_import))
    monkeypatch.setattr(comps, "_COMP_DIR", tmp_path)
    (tmp_path / "s6e8.py").touch()
    (tmp_path / "s4e1.py").touch()
    (tmp_path / "__init__.py").touch()

    assert comps.competition_id_to_slug() == {"playground-series-s6e8": "s6e8"}
    assert comps.active_competition_ids() == {"playground-series-s6e8"}


def test_sweep_noop_when_nothing_idle(monkeypatch):
    monkeypatch.setattr(run_daemon, "_last_queue_refill_sweep", _long_ago())
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None

    import datetime as dt
    # raw.attempts.run_ts는 naive(#223) — psycopg2 실반환 형태. 프로덕션 idle_cutoff는
    # UTC 기준(datetime.now(timezone.utc) - 6h)이므로 fixture도 로컬 datetime.now()가
    # 아니라 naive UTC로 만들어야 한다 — 안 그러면 UTC보다 6시간 이상 뒤처진 타임존
    # (미국 서부 등)에서 이 테스트가 허위로 실패한다(#239, adversarial review — #223을
    # 막으려던 테스트 파일에서 같은 클래스의 타임존 버그가 재발할 뻔함).
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    conn.execute.return_value.fetchall.return_value = [("playground-series-s6e8", now)]

    scan, active = _patch_scan({"playground-series-s6e8": "s6e8"})
    with scan, active:
        _sweep_queue_refill(conn)

    insert_calls = [
        c for c in conn.execute.call_args_list
        if "insert into raw.cycle_queue" in c.args[0]
    ]
    assert insert_calls == []
