"""무인 사이클 daemon — raw.cycle_queue를 폴링해서 cycle을 순차 실행한다.

Usage:
    uv run python -m bin.run_daemon

SIGTERM/SIGINT를 받으면 현재 사이클이 끝난 뒤 종료한다.

환경변수 (페이싱 — 미설정 시 비활성):
    OLLAMA_CLOUD_SESSION_HOURS   세션 윈도우 길이 (기본 5.0)
    OLLAMA_CLOUD_SESSION_CYCLES  세션당 최대 사이클 수 (0=비활성)
    OLLAMA_CLOUD_WEEKLY_CYCLES   주간 최대 사이클 수 (0=비활성)
"""
from __future__ import annotations

import importlib
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

import bin.airflow_client as airflow_client
from bin.api import DaemonState, create_app
from cycle.run import CycleConfig
from cycle.super_cycle import run_super_cycle
from memory.retriever import EmbeddingUnavailableError
from store.db import connect, ensure_competition

POLL_INTERVAL = 10  # 빈 큐 대기 간격 (초)
ROOT = Path(__file__).parent.parent

_running = True


# ---------------------------------------------------------------------------
# Ollama Cloud 페이싱
# ---------------------------------------------------------------------------

@dataclass
class OllamaPacer:
    """세션(5h) + 주간 사이클 한도를 추적해 Ollama Cloud 과금/429를 방지한다.

    session_cycles / weekly_cycles 중 하나라도 0이면 해당 한도는 비활성.
    한도 초과 시 다음 윈도우 시작까지 sleep 후 반환한다 — 스킵이 아닌 대기.
    """
    session_hours: float
    session_cycles: int
    weekly_cycles: int

    _session_start: float = field(default_factory=time.time, init=False)
    _session_count: int = field(default=0, init=False)
    _week_start: float = field(default_factory=time.time, init=False)
    _week_count: int = field(default=0, init=False)

    @classmethod
    def from_env(cls) -> "OllamaPacer":
        return cls(
            session_hours=float(os.getenv("OLLAMA_CLOUD_SESSION_HOURS", "5.0")),
            session_cycles=int(os.getenv("OLLAMA_CLOUD_SESSION_CYCLES", "0")),
            weekly_cycles=int(os.getenv("OLLAMA_CLOUD_WEEKLY_CYCLES", "0")),
        )

    def restore_from_db(self, conn) -> None:
        """재시작 후 DB의 실제 attempt 수로 카운터를 복원한다."""
        if not self.enabled:
            return
        now = datetime.now(timezone.utc)
        session_cutoff = now - timedelta(hours=self.session_hours)
        week_cutoff = now - timedelta(weeks=1)
        row = conn.execute(
            """
            select
                count(*) filter (where run_ts >= %s) as session_count,
                count(*) filter (where run_ts >= %s) as week_count
            from raw.attempts
            """,
            [session_cutoff, week_cutoff],
        ).fetchone()
        if row:
            self._session_count = row[0]
            self._week_count = row[1]
        print(
            f"[pacer] restored from DB — session={self._session_count}, week={self._week_count}"
        )

    @property
    def enabled(self) -> bool:
        return self.session_cycles > 0 or self.weekly_cycles > 0

    def acquire(self) -> None:
        """한도 초과 시 다음 윈도우까지 대기. 한도 내면 즉시 반환."""
        if not self.enabled:
            return

        now = time.monotonic()
        week_secs = 7 * 24 * 3600
        session_secs = self.session_hours * 3600

        # 주간 리셋
        if now - self._week_start >= week_secs:
            self._week_start = now
            self._week_count = 0

        # 세션 리셋
        if now - self._session_start >= session_secs:
            self._session_start = now
            self._session_count = 0

        # 주간 한도 체크
        if self.weekly_cycles > 0 and self._week_count >= self.weekly_cycles:
            wait = week_secs - (now - self._week_start)
            print(
                f"[pacer] weekly limit {self._week_count}/{self.weekly_cycles} reached"
                f" — sleeping {wait/3600:.1f}h"
            )
            time.sleep(max(wait, 0))
            self._week_start = time.monotonic()
            self._week_count = 0

        # 세션 한도 체크
        if self.session_cycles > 0 and self._session_count >= self.session_cycles:
            wait = session_secs - (time.monotonic() - self._session_start)
            print(
                f"[pacer] session limit {self._session_count}/{self.session_cycles} reached"
                f" — sleeping {wait/60:.0f}min"
            )
            time.sleep(max(wait, 0))
            self._session_start = time.monotonic()
            self._session_count = 0

    def record(self) -> None:
        self._session_count += 1
        self._week_count += 1


# ---------------------------------------------------------------------------
# 큐 헬퍼
# ---------------------------------------------------------------------------

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
    sets = ["status = %s"]
    vals: list = [status]
    for k, v in extra.items():
        sets.append(f"{k} = %s")
        vals.append(v)
    vals.append(queue_id)
    conn.execute(
        f"update raw.cycle_queue set {', '.join(sets)} where queue_id = %s",
        vals,
    )


def _is_cancelled(conn, queue_id: str) -> bool:
    row = conn.execute(
        "select status from raw.cycle_queue where queue_id = %s",
        [queue_id],
    ).fetchone()
    return row is not None and row[0] == "cancelled"


# ---------------------------------------------------------------------------
# 큐 아이템 처리
# ---------------------------------------------------------------------------

def _run_api(state: DaemonState) -> None:
    import uvicorn
    api_conn = connect(apply_schema=False)
    app = create_app(api_conn, state)
    host = os.getenv("DAEMON_API_HOST", "127.0.0.1")
    port = int(os.getenv("DAEMON_API_PORT", "8000"))
    print(f"[api] listening on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _process(conn, item: dict, pacer: OllamaPacer, state: DaemonState) -> None:
    qid = item["queue_id"]
    competition = item["competition"]
    stage = item["stage"]
    n_cycles = item["n_cycles"]

    mode = "airflow" if airflow_client.available() else "direct"
    print(f"[daemon] starting queue_id={qid} competition={competition} stage={stage} n={n_cycles} mode={mode}")
    _set_status(conn, qid, "running", started_at=datetime.now(timezone.utc))
    state.update(
        current_queue_id=qid,
        current_competition=competition,
        current_cycle=0,
        current_n_cycles=n_cycles,
    )

    try:
        comp = importlib.import_module(f"config.competitions.{competition}")
    except ModuleNotFoundError as exc:
        _set_status(conn, qid, "failed", ended_at=datetime.now(timezone.utc), error=str(exc))
        print(f"[daemon] failed to load competition config: {exc}")
        return

    # direct 모드에서만 train 데이터 사전 로드 (airflow 모드는 task 컨테이너 안에서 로드)
    train: pl.DataFrame | None = None
    if mode == "direct":
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
    cycles_done = 0
    skipped = 0
    failed = False

    for i in range(n_cycles):
        if not _running or _is_cancelled(conn, qid):
            print(f"[daemon] queue_id={qid} cancelled at cycle {i + 1}")
            _set_status(conn, qid, "cancelled", ended_at=datetime.now(timezone.utc),
                        cycles_done=cycles_done, latest_score=latest_score)
            return

        # Ollama Cloud 페이싱 — 한도 초과 시 여기서 대기
        pacer.acquire()

        if mode == "airflow":
            try:
                dag_run_id = airflow_client.trigger_dag_run(
                    competition_id=competition,  # 모듈명 (e.g. s4e1), COMPETITION_ID 아님
                    stage=stage,
                    queue_id=qid,
                )
                print(f"[daemon] cycle {i + 1}/{n_cycles} dag_run={dag_run_id}")
                final_state = airflow_client.wait_for_dag_run(dag_run_id)
                if final_state == "success":
                    row = conn.execute(
                        """
                        select attempt_id, cv_score, label from raw.attempts
                        where competition_id = %s
                        order by run_ts desc limit 1
                        """,
                        [comp.COMPETITION_ID],
                    ).fetchone()
                    if row:
                        aid, cv, label = row
                        if cv is not None:
                            latest_score = cv
                        print(f"[daemon] cycle {i + 1}/{n_cycles} attempt={aid[:8]} cv={cv} label={label}")
                    cycles_done += 1
                    pacer.record()
                    state.update(current_cycle=cycles_done, last_cycle_at=datetime.now(timezone.utc))
                else:
                    err_msg = f"dag_run {dag_run_id} ended with state={final_state}"
                    print(f"[daemon] cycle {i + 1}/{n_cycles} {err_msg}")
                    _set_status(conn, qid, "failed",
                                ended_at=datetime.now(timezone.utc),
                                cycles_done=cycles_done,
                                latest_score=latest_score,
                                error=err_msg)
                    failed = True
                    break
            except Exception as exc:
                err_msg = str(exc)
                print(f"[daemon] cycle {i + 1}/{n_cycles} airflow error: {err_msg}")
                _set_status(conn, qid, "failed",
                            ended_at=datetime.now(timezone.utc),
                            cycles_done=cycles_done,
                            latest_score=latest_score,
                            error=err_msg[:2000])
                failed = True
                break

        else:
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
                result = run_super_cycle(conn, config)
                if result.cv_score is not None:
                    latest_score = result.cv_score
                n_attempts = len(result.all_results)
                cycles_done += 1
                pacer.record()
                state.update(current_cycle=cycles_done, last_cycle_at=datetime.now(timezone.utc))
                print(
                    f"[daemon] cycle {i + 1}/{n_cycles} super={result.super_cycle_id[:8]}"
                    f" winner={result.attempt_id[:8]} cv={result.cv_score}"
                    f" label={result.label} n_attempts={n_attempts}"
                )
            except EmbeddingUnavailableError as exc:
                print(f"[daemon] cycle {i + 1}/{n_cycles} skipped — embedding unavailable: {exc}")
                skipped += 1
            except Exception as exc:
                err_msg = str(exc)
                print(f"[daemon] cycle {i + 1}/{n_cycles} raised: {err_msg}")
                _set_status(conn, qid, "failed",
                            ended_at=datetime.now(timezone.utc),
                            cycles_done=cycles_done,
                            latest_score=latest_score,
                            error=err_msg[:2000])
                failed = True
                break

        _set_status(conn, qid, "running",
                    cycles_done=cycles_done, latest_score=latest_score)

    if not failed:
        if _is_cancelled(conn, qid):
            print(f"[daemon] queue_id={qid} cancelled (detected post-cycle)")
        elif cycles_done == 0 and skipped > 0:
            _set_status(conn, qid, "failed",
                        ended_at=datetime.now(timezone.utc),
                        cycles_done=0,
                        error=f"all {skipped} cycles skipped — embedding unavailable")
            print(f"[daemon] queue_id={qid} failed — all {skipped} cycles skipped (embedding)")
        else:
            _set_status(conn, qid, "done",
                        ended_at=datetime.now(timezone.utc),
                        cycles_done=cycles_done, latest_score=latest_score)
            print(f"[daemon] queue_id={qid} done latest_score={latest_score}")

    state.update(current_queue_id=None, current_competition=None,
                 current_cycle=0, current_n_cycles=0)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    pacer = OllamaPacer.from_env()
    if pacer.enabled:
        print(
            f"[daemon] Ollama Cloud pacing enabled — "
            f"session={pacer.session_cycles} cycles/{pacer.session_hours}h, "
            f"weekly={pacer.weekly_cycles} cycles"
        )

    if airflow_client.available():
        print(f"[daemon] airflow mode — {airflow_client._AIRFLOW_URL} dag={airflow_client.DAG_ID}")
    else:
        print("[daemon] direct mode — AIRFLOW_URL not set, running cycles in-process")

    state = DaemonState()
    api_thread = threading.Thread(target=_run_api, args=(state,), daemon=True)
    api_thread.start()

    conn = connect()
    pacer.restore_from_db(conn)
    print("[daemon] started — polling raw.cycle_queue")

    while _running:
        item = _pop_pending(conn)
        if item is None:
            time.sleep(POLL_INTERVAL)
            continue
        _process(conn, item, pacer, state)

    conn.close()
    print("[daemon] stopped")


if __name__ == "__main__":
    main()
