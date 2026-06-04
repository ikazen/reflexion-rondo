"""무인 사이클 daemon — raw.cycle_queue를 폴링해서 cycle을 순차 실행한다.

Usage:
    uv run python -m bin.run_daemon

SIGTERM/SIGINT를 받으면 현재 사이클이 끝난 뒤 종료한다.
"""
from __future__ import annotations

import importlib
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from cycle.run import CycleConfig, run_cycle
from store.db import connect, ensure_competition

POLL_INTERVAL = 10  # 빈 큐 대기 간격 (초)
ROOT = Path(__file__).parent.parent

_running = True


def _handle_signal(sig, frame) -> None:
    global _running
    _running = False
    print(f"[daemon] signal {sig} received — will stop after current cycle")


def _pop_pending(conn) -> dict | None:
    row = conn.execute(
        """
        select queue_id, competition, stage, n_cycles, priority
        from raw.cycle_queue
        where status = 'pending'
        order by priority desc, created_at asc
        limit 1
        """
    ).fetchone()
    if not row:
        return None
    return {"queue_id": row[0], "competition": row[1], "stage": row[2],
            "n_cycles": row[3], "priority": row[4]}


def _set_status(conn, queue_id: str, status: str, **extra) -> None:
    sets = ["status = ?"]
    vals: list = [status]
    for k, v in extra.items():
        sets.append(f"{k} = ?")
        vals.append(v)
    vals.append(queue_id)
    conn.execute(
        f"update raw.cycle_queue set {', '.join(sets)} where queue_id = ?",
        vals,
    )


def _is_cancelled(conn, queue_id: str) -> bool:
    row = conn.execute(
        "select status from raw.cycle_queue where queue_id = ?",
        [queue_id],
    ).fetchone()
    return row is not None and row[0] == "cancelled"


def _process(conn, item: dict) -> None:
    qid = item["queue_id"]
    competition = item["competition"]
    stage = item["stage"]
    n_cycles = item["n_cycles"]

    print(f"[daemon] starting queue_id={qid} competition={competition} stage={stage} n={n_cycles}")
    _set_status(conn, qid, "running", started_at=datetime.now(timezone.utc))

    try:
        comp = importlib.import_module(f"config.competitions.{competition}")
    except ModuleNotFoundError as exc:
        _set_status(conn, qid, "failed", ended_at=datetime.now(timezone.utc), error=str(exc))
        print(f"[daemon] failed to load competition config: {exc}")
        return

    train = pl.read_csv(comp.DATA_DIR / "train.csv").drop(comp.DROP_COLS)
    ensure_competition(
        conn,
        competition_id=comp.COMPETITION_ID,
        name=comp.NAME,
        task_type=comp.TASK_TYPE,
        metric=comp.METRIC,
        metric_sign=comp.METRIC_SIGN,
    )

    latest_score: float | None = None
    failed = False

    for i in range(n_cycles):
        if not _running or _is_cancelled(conn, qid):
            print(f"[daemon] queue_id={qid} cancelled at cycle {i + 1}")
            _set_status(conn, qid, "cancelled", ended_at=datetime.now(timezone.utc))
            return

        config = CycleConfig(
            competition_id=comp.COMPETITION_ID,
            train=train,
            target_col=comp.TARGET,
            metric=comp.METRIC,
            stage=stage,
            eda_card=comp.EDA_CARD,
            n_splits=getattr(comp, "N_SPLITS", 5),
            seed=42,
            k_retrieve=5,
            is_classification=comp.IS_CLASSIFICATION,
        )

        try:
            result = run_cycle(conn, config)
            if result.cv_score is not None:
                latest_score = result.cv_score
            print(
                f"[daemon] cycle {i + 1}/{n_cycles} attempt={result.attempt_id[:8]}"
                f" cv={result.cv_score} label={result.label}"
            )
        except Exception as exc:
            err_msg = str(exc)
            print(f"[daemon] cycle {i + 1}/{n_cycles} raised: {err_msg}")
            _set_status(
                conn, qid, "failed",
                ended_at=datetime.now(timezone.utc),
                cycles_done=i,
                latest_score=latest_score,
                error=err_msg[:2000],
            )
            failed = True
            break

        _set_status(conn, qid, "running",
                    cycles_done=i + 1, latest_score=latest_score)

    if not failed:
        _set_status(conn, qid, "done",
                    ended_at=datetime.now(timezone.utc),
                    cycles_done=n_cycles, latest_score=latest_score)
        print(f"[daemon] queue_id={qid} done latest_score={latest_score}")


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    conn = connect()
    print("[daemon] started — polling raw.cycle_queue")

    while _running:
        item = _pop_pending(conn)
        if item is None:
            time.sleep(POLL_INTERVAL)
            continue
        _process(conn, item)

    conn.close()
    print("[daemon] stopped")


if __name__ == "__main__":
    main()
