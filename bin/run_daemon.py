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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

import bin.airflow_client as airflow_client
from bin.api import _SUBMIT_TIMEOUT_SEC, DaemonState, _competition_id_to_slug, create_app, refresh_submission_row
from bin.archive_lessons import archive_low_gain_lessons
from cycle.run import CycleConfig, establish_bootstrap_baseline, run_cycle
from memory.retriever import EmbeddingUnavailableError
from store.db import connect, ensure_competition
from store.train_data import load_train

POLL_INTERVAL_SEC = 10
MAX_CONSECUTIVE_CYCLE_FAILURES = int(os.getenv("RONDO_MAX_CONSECUTIVE_FAILURES", "5"))
# 큐 항목 하나가 daemon을 통째로 붙잡지 않도록 리스당 최대 사이클 수를 제한한다
# (2026-08 실측: 30~100-cycle 큐 하나가 며칠씩 다른 13개 대회의 실행 기회를 막음).
# 리스 소진 시 pending으로 되돌려 다음으로 오래 기다린 항목에 순서를 넘긴다 —
# 큐 슬롯 자체(big=3)는 그대로라 처리량 이득은 없고 대회 간 커버리지만 회복한다.
DAEMON_CYCLES_PER_LEASE = int(os.getenv("DAEMON_CYCLES_PER_LEASE", "5"))
# attempt_gate(#203)와 짝을 이루는 전환 — off가 기본값이라 gate만 배포해도 아무 동작
# 변화가 없다. gate가 며칠 안정화된 뒤에만 on으로 바꾼다(이미지 재빌드 없이 되돌릴
# 수 있는 롤백 레버, #204).
_WAIT_ON_PROMOTE_TI = os.getenv("RONDO_WAIT_ON_PROMOTE_TI", "") not in ("", "0", "false", "False")
ROOT = Path(__file__).parent.parent

_running = True


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

        if now - self._week_start >= week_secs:
            self._week_start = now
            self._week_count = 0

        if now - self._session_start >= session_secs:
            self._session_start = now
            self._session_count = 0

        if self.weekly_cycles > 0 and self._week_count >= self.weekly_cycles:
            wait = week_secs - (now - self._week_start)
            print(
                f"[pacer] weekly limit {self._week_count}/{self.weekly_cycles} reached"
                f" — sleeping {wait/3600:.1f}h"
            )
            time.sleep(max(wait, 0))
            self._week_start = time.monotonic()
            self._week_count = 0

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


def _handle_signal(sig, frame) -> None:
    global _running
    _running = False
    print(f"[daemon] signal {sig} received — will stop after current cycle")


def _pop_pending(conn) -> dict | None:
    # last_leased_at 기준 오름차순 — 리스를 방금 마친 항목은 last_leased_at이 now()로
    # 갱신돼 뒤로 밀리고, 한 번도 리스된 적 없는 항목은 NULL이라 created_at으로 폴백해
    # 기존 "가장 오래 기다린 것부터" 순서를 유지한다.
    row = conn.execute(
        """
        select queue_id, competition, stage, n_cycles, priority, cycles_done, latest_score
        from raw.cycle_queue
        where status = 'pending'
        order by priority desc, coalesce(last_leased_at, created_at) asc
        limit 1
        """
    ).fetchone()
    if not row:
        return None
    return {"queue_id": row[0], "competition": row[1], "stage": row[2],
            "n_cycles": row[3], "priority": row[4], "cycles_done": row[5],
            "latest_score": row[6]}


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


def _final_status(cycles_done: int, skipped: int, failed_cycles: int) -> tuple[str, str | None]:
    if cycles_done == 0 and (skipped + failed_cycles) > 0:
        return "failed", f"all cycles unsuccessful — {failed_cycles} failed, {skipped} skipped"
    return "done", None


# LB 자동 재폴링 — daemon 루프가 유휴 틱마다 raw.kaggle_submissions를 훑어
# 스스로 재확인한다(수동 `POST /api/submissions/{id}/refresh` 호출 없이도 동작).
_SUBMISSION_SWEEP_INTERVAL_SEC = 60  # 이 주기로만 훑는다 — daemon 루프 자체는 10s poll
_last_submission_sweep: float = 0.0

_REFRESH_WINDOW_10MIN_SEC = 600
_REFRESH_WINDOW_1H_SEC = 3600
_REFRESH_WINDOW_6H_SEC = 6 * 3600
_REFRESH_INTERVAL_UNDER_10MIN_SEC = 120
_REFRESH_INTERVAL_UNDER_1H_SEC = 600
_REFRESH_INTERVAL_UNDER_6H_SEC = 1800
_REFRESH_INTERVAL_OVER_6H_SEC = 7200


def _submission_refresh_due(submitted_at: datetime, checked_at, now: datetime) -> bool:
    """제출 경과 시간에 따라 재확인 간격을 늘리는 백오프.

    Kaggle 채점은 보통 분 단위지만 드물게 몇 시간 걸린다 — 갓 제출된 건 자주,
    오래 pending인 건 뜸하게 확인해 불필요한 kaggle CLI 호출을 아낀다.

    raw.kaggle_submissions.submitted_at/checked_at는 timezone 없는 `timestamp`
    컬럼이라 psycopg2가 naive datetime으로 돌려준다 — 이 코드베이스는 DB의 naive
    timestamp를 암묵적으로 UTC로 취급하는 관례라, aware datetime과 그대로 빼면
    TypeError이므로 셋 다 naive로 정규화한 뒤 비교한다.
    """
    now = now.replace(tzinfo=None) if now.tzinfo is not None else now
    submitted_at = submitted_at.replace(tzinfo=None) if submitted_at.tzinfo is not None else submitted_at
    if checked_at is not None and checked_at.tzinfo is not None:
        checked_at = checked_at.replace(tzinfo=None)
    if checked_at is None:
        return True
    elapsed_since_submit = (now - submitted_at).total_seconds()
    if elapsed_since_submit < _REFRESH_WINDOW_10MIN_SEC:
        interval = _REFRESH_INTERVAL_UNDER_10MIN_SEC
    elif elapsed_since_submit < _REFRESH_WINDOW_1H_SEC:
        interval = _REFRESH_INTERVAL_UNDER_1H_SEC
    elif elapsed_since_submit < _REFRESH_WINDOW_6H_SEC:
        interval = _REFRESH_INTERVAL_UNDER_6H_SEC
    else:
        interval = _REFRESH_INTERVAL_OVER_6H_SEC
    return (now - checked_at).total_seconds() >= interval


def _sweep_stale_submissions(conn) -> None:
    global _last_submission_sweep
    now_mono = time.monotonic()
    if now_mono - _last_submission_sweep < _SUBMISSION_SWEEP_INTERVAL_SEC:
        return
    _last_submission_sweep = now_mono

    # 'submitted'/'pending'은 실제로 쓰인 적 없는(과도 상태이거나 미사용) 값이라
    # 이 필터로는 아무것도 안 걸리는 게 정상 케이스였다(#146) — 실제로 걸리는
    # raw.kaggle_submissions.status는 'queued'/'submitting'/'complete'/'error'.
    # 'queued'/'submitting'을 빼먹으면 daemon 재시작 등으로 갱신이 끊긴 제출이
    # 영구히 미확인 상태로 남는다(실측: 10일·7일째 방치된 건 2건).
    rows = conn.execute(
        """
        select submission_id, submitted_at, checked_at, status
        from raw.kaggle_submissions
        where status in ('submitted', 'pending', 'queued', 'submitting')
        """
    ).fetchall()

    now = datetime.now(timezone.utc)
    for submission_id, submitted_at, checked_at, status in rows:
        if status == "submitting":
            # kaggle CLI가 지금 이 순간 실제로 업로드 중일 수 있다 — 진행 중인
            # 업로드와 경합하지 않도록, 업로드 타임아웃(bin/api.py의
            # _SUBMIT_TIMEOUT_SEC)보다 오래 이 상태면(=daemon 재시작 등으로
            # 갱신이 끊긴 것으로 판단) 재확인, 그 전엔 스킵.
            submitted_naive = submitted_at.replace(tzinfo=None) if submitted_at.tzinfo else submitted_at
            now_naive = now.replace(tzinfo=None)
            if (now_naive - submitted_naive).total_seconds() < _SUBMIT_TIMEOUT_SEC:
                continue
        if not _submission_refresh_due(submitted_at, checked_at, now):
            continue
        try:
            rec = refresh_submission_row(conn, submission_id)
            if rec and rec.get("status") == "complete":
                print(f"[daemon] submission {submission_id[:8]} refreshed → complete lb={rec.get('lb_score')}")
            elif rec and rec.get("status") in ("error", "invalid"):
                print(f"[daemon] submission {submission_id[:8]} refreshed → {rec['status']}")
        except Exception as exc:
            print(f"[daemon] submission {submission_id[:8]} refresh failed: {exc}")


# 저효율 교훈 자동 archive — 반복 인용됐는데(times_applied>=3) 평균 gain이 0
# 이하인 교훈을 검색 후보 풀에서 뺀다. 대회 무관 전역 집계라 자주 훑을 필요
# 없음 — 하루 주기.
_LESSON_ARCHIVE_SWEEP_INTERVAL_SEC = 24 * 3600
_last_lesson_archive_sweep: float = 0.0


def _sweep_low_gain_lessons(conn) -> None:
    global _last_lesson_archive_sweep
    now_mono = time.monotonic()
    if now_mono - _last_lesson_archive_sweep < _LESSON_ARCHIVE_SWEEP_INTERVAL_SEC:
        return
    _last_lesson_archive_sweep = now_mono

    try:
        archived_ids = archive_low_gain_lessons(conn)
        if archived_ids:
            print(f"[daemon] archived {len(archived_ids)} low-gain lesson(s)")
    except Exception as exc:
        print(f"[daemon] lesson archive sweep failed: {exc}")


# cycle_queue 완전 고갈 시 자동 재보급 — 안 그러면 사람이 enqueue할 때까지 daemon 전체가
# idle에 멈춘다(#196 실측: 2026-08-17~18 큐 소진 후 27시간 attempt 0건). pending/running이
# 하나도 없을 때만, config/competitions/*.py 전체 중 최근 _QUEUE_REFILL_IDLE_HOURS시간
# 이상 attempt가 없던(또는 한 번도 없던) 대회를 재큐잉한다.
_QUEUE_REFILL_SWEEP_INTERVAL_SEC = 1800
_QUEUE_REFILL_IDLE_HOURS = 6
_QUEUE_REFILL_N_CYCLES = 20
_last_queue_refill_sweep: float = 0.0


def _sweep_queue_refill(conn) -> None:
    global _last_queue_refill_sweep
    now_mono = time.monotonic()
    if now_mono - _last_queue_refill_sweep < _QUEUE_REFILL_SWEEP_INTERVAL_SEC:
        return
    _last_queue_refill_sweep = now_mono

    active = conn.execute(
        "select 1 from raw.cycle_queue where status in ('pending', 'running') limit 1"
    ).fetchone()
    if active:
        return

    # bin.api._competition_id_to_slug()는 {competition_id: slug} — cycle_queue.competition
    # 컬럼은 slug를 쓰므로(예: "s6e8") 여기서 뒤집는다.
    slug_to_cid = {slug: cid for cid, slug in _competition_id_to_slug().items()}
    # ACTIVE=False(#227, Milestone v1.6.0 — fleet 동결 후 deep tier만 유지)인 대회는
    # 재보급 대상에서 제외한다. 이 헬퍼는 auto-submit 등 다른 소비자와 공유하는
    # _competition_id_to_slug()는 건드리지 않고 여기서만 필터링한다 — 다만 동결 대회의
    # auto-submit이 "계속된다"는 보장은 아니다: bin/api.py:auto_submit()은
    # raw.attempts.run_ts가 최근 window_hours(airflow-stack에서 24h 하드코딩) 이내인
    # 대회만 보므로, 새 attempt가 안 들어오는 동결 대회는 재보급 여부와 무관하게
    # 24~48h 내에 auto-submit 대상에서도 자연히 빠진다(#239, adversarial review) — 그게
    # 실제로 원하는 동작이다(동결 대회에 컴퓨트도 제출 예산도 더 안 쓴다).
    #
    # import_module이 실패하면(예: config 파일 삭제·오탈자) 그 슬러그만 건너뛴다 —
    # 안 그러면 daemon 메인 루프 전체가 죽는 #223과 같은 클래스의 크래시가 된다.
    active_slug_to_cid: dict[str, str] = {}
    for slug, cid in slug_to_cid.items():
        try:
            comp = importlib.import_module(f"config.competitions.{slug}")
        except Exception as exc:
            print(f"[daemon] queue refill — config.competitions.{slug} import 실패, 건너뜀: {exc}")
            continue
        if getattr(comp, "ACTIVE", True):
            active_slug_to_cid[slug] = cid
    slug_to_cid = active_slug_to_cid
    if not slug_to_cid:
        return

    last_run = dict(conn.execute(
        "select competition_id, max(run_ts) from raw.attempts where competition_id = any(%s) group by 1",
        [list(slug_to_cid.values())],
    ).fetchall())

    now = datetime.now(timezone.utc)
    # raw.attempts.run_ts는 timezone 없는 컬럼이라 psycopg2가 naive datetime으로
    # 반환한다 — aware idle_cutoff와 그대로 비교하면 TypeError(#223). 이 repo에서
    # timestamp 컬럼에 쓰는 aware datetime은 전부 UTC 기준으로 저장되므로 naive로
    # 맞춰서 비교한다.
    idle_cutoff = (now - timedelta(hours=_QUEUE_REFILL_IDLE_HOURS)).replace(tzinfo=None)
    idle_slugs = sorted(
        slug for slug, cid in slug_to_cid.items()
        if last_run.get(cid) is None or last_run[cid] < idle_cutoff
    )
    if not idle_slugs:
        return

    for slug in idle_slugs:
        conn.execute(
            """
            insert into raw.cycle_queue
                (queue_id, competition, stage, n_cycles, priority, status, created_at)
            values (%s, %s, 'reflexion', %s, 0, 'pending', %s)
            """,
            [str(uuid.uuid4()), slug, _QUEUE_REFILL_N_CYCLES, now],
        )
    print(f"[daemon] queue refill — {len(idle_slugs)} idle competition(s) re-enqueued: {idle_slugs}")


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
    # 리스 재개 — cycles_done/latest_score는 이전 리스가 pending으로 되돌리며
    # 남겨둔 진행 상태(store/schema.sql cycle_queue). 최초 리스면 둘 다 0/None.
    cycles_done = item.get("cycles_done") or 0
    latest_score: float | None = item.get("latest_score")

    mode = "airflow" if airflow_client.available() else "direct"
    print(
        f"[daemon] starting queue_id={qid} competition={competition} stage={stage} "
        f"progress={cycles_done}/{n_cycles} mode={mode}"
    )
    # started_at은 "최초 시작"으로 dashboard/api가 읽으므로 리스 재개 시에는 건드리지
    # 않는다 — last_leased_at만 매 리스마다 갱신해 _pop_pending의 라운드로빈 정렬에 쓴다.
    lease_status_kwargs = {"last_leased_at": datetime.now(timezone.utc)}
    if cycles_done == 0:
        lease_status_kwargs["started_at"] = datetime.now(timezone.utc)
    _set_status(conn, qid, "running", **lease_status_kwargs)
    state.update(
        current_queue_id=qid,
        current_competition=competition,
        current_cycle=cycles_done,
        current_n_cycles=n_cycles,
    )

    try:
        comp = importlib.import_module(f"config.competitions.{competition}")
    except ModuleNotFoundError as exc:
        _set_status(conn, qid, "failed", ended_at=datetime.now(timezone.utc), error=str(exc))
        print(f"[daemon] failed to load competition config: {exc}")
        return

    # direct 모드는 로컬 smoke/test용 단일 attempt 경로다.
    # airflow 모드는 task 컨테이너 안에서 데이터를 로드하고 super-cycle을 실행한다.
    train: pl.DataFrame | None = None
    if mode == "direct":
        train = load_train(comp)
        ensure_competition(
            conn,
            competition_id=comp.COMPETITION_ID,
            name=comp.NAME,
            task_type=comp.TASK_TYPE,
            metric=comp.METRIC,
            metric_sign=comp.METRIC_SIGN,
        )

    # successes는 "이번 리스에서 실제로 성공한 cycle 수"(_final_status의 "전량 실패"
    # 판정용, 기존 cycles_done의 원래 의미). cycles_done은 이제 성공/실패/스킵을
    # 가리지 않고 소비한 예산 전체를 가리킨다 — 리스 경계를 넘어 n_cycles 소진 여부를
    # 정확히 재개 판단하려면 "시도한 총량"이 필요하기 때문(성공만 세면 계속 실패하는
    # 대회가 예산을 영영 못 채워 무한정 리스를 반복하게 된다).
    successes = 0
    skipped = 0
    failed_cycles = 0
    consecutive_failures = 0
    aborted = False

    lease_end = min(n_cycles, cycles_done + DAEMON_CYCLES_PER_LEASE)
    while cycles_done < lease_end:
        if not _running or _is_cancelled(conn, qid):
            print(f"[daemon] queue_id={qid} cancelled at cycle {cycles_done + 1}")
            _set_status(conn, qid, "cancelled", ended_at=datetime.now(timezone.utc),
                        cycles_done=cycles_done, latest_score=latest_score)
            return

        pacer.acquire()

        cycle_failed = False
        cycle_skipped = False
        err_msg = None

        if mode == "airflow":
            try:
                dag_run_id = airflow_client.trigger_dag_run(
                    competition_id=competition,  # 모듈명 (e.g. s4e1), COMPETITION_ID 아님
                    stage=stage,
                    queue_id=qid,
                )
                print(f"[daemon] cycle {cycles_done + 1}/{n_cycles} dag_run={dag_run_id}")
                if _WAIT_ON_PROMOTE_TI:
                    # attempt_gate(#203) 도입 후 straggler attempt는 여전히 45분
                    # execution_timeout까지 DAG run을 running 상태로 붙잡는다 — DAG run
                    # 전체를 기다리면 gate를 추가한 의미가 없다(#204). promote task
                    # instance 하나의 완료만 기다린다.
                    final_state = airflow_client.wait_for_task_instance(dag_run_id, "promote")
                else:
                    final_state = airflow_client.wait_for_dag_run(dag_run_id)
            except Exception as exc:
                final_state, err_msg = "error", str(exc)
                print(f"[daemon] cycle {cycles_done + 1}/{n_cycles} airflow error: {err_msg}")

            if final_state == "success":
                row = conn.execute(
                    """
                    select attempt_id, cv_score, label from raw.attempts
                    where competition_id = %s
                      and was_promoted is not false
                    order by run_ts desc limit 1
                    """,
                    [comp.COMPETITION_ID],
                ).fetchone()
                if row:
                    aid, cv, label = row
                    if cv is not None:
                        latest_score = cv
                    print(f"[daemon] cycle {cycles_done + 1}/{n_cycles} winner={aid[:8]} cv={cv} label={label}")
                successes += 1
                consecutive_failures = 0
                pacer.record()
            else:
                cycle_failed = True
                err_msg = err_msg or f"dag_run {dag_run_id} ended with state={final_state}"

        else:
            if train is None:
                raise RuntimeError("direct mode requires train data to be loaded")
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
                cpu_budget_secs=getattr(comp, "CPU_BUDGET_SECS", None),
            )
            try:
                result = run_cycle(conn, config)
                if result.cv_score is not None:
                    latest_score = result.cv_score
                successes += 1
                consecutive_failures = 0
                pacer.record()
                print(
                    f"[daemon] cycle {cycles_done + 1}/{n_cycles} attempt={result.attempt_id[:8]}"
                    f" cv={result.cv_score} label={result.label}"
                )
            except EmbeddingUnavailableError as exc:
                print(f"[daemon] cycle {cycles_done + 1}/{n_cycles} skipped — embedding unavailable: {exc}")
                skipped += 1
                cycle_skipped = True
            except Exception as exc:
                cycle_failed = True
                err_msg = str(exc)

        cycles_done += 1
        state.update(current_cycle=cycles_done, last_cycle_at=datetime.now(timezone.utc))

        if cycle_failed:
            failed_cycles += 1
            consecutive_failures += 1
            print(
                f"[daemon] cycle {cycles_done}/{n_cycles} failed ({(err_msg or '')[:120]})"
                f" — {consecutive_failures}/{MAX_CONSECUTIVE_CYCLE_FAILURES} consecutive"
            )
            if consecutive_failures >= MAX_CONSECUTIVE_CYCLE_FAILURES:
                _set_status(conn, qid, "failed",
                            ended_at=datetime.now(timezone.utc),
                            cycles_done=cycles_done,
                            latest_score=latest_score,
                            error=(
                                f"{consecutive_failures} consecutive cycle failures; "
                                f"last: {err_msg}"
                            )[:2000])
                print(f"[daemon] queue_id={qid} aborted — {consecutive_failures} consecutive failures")
                aborted = True
                break
            continue

        if cycle_skipped:
            continue

        # 사이클이 성공한 직후 무조건 "running"으로 되돌아가면, 이 사이클이
        # 진행되는 동안(대부분의 시간, ~2분) 걸린 외부 PATCH cancelled 요청이 여기서
        # 조용히 지워지고 다음 반복의 취소 체크는 이미 복구된 "running"만 보게 된다 —
        # 사실상 취소가 실패 분기(continue로 이 호출을 건너뜀)가 아니면 절대 반영 안 됨.
        # 루프 맨 위 체크와 대칭적으로 여기서도 확인한다.
        if _is_cancelled(conn, qid):
            print(f"[daemon] queue_id={qid} cancelled at cycle {cycles_done} (detected post-cycle)")
            _set_status(conn, qid, "cancelled", ended_at=datetime.now(timezone.utc),
                        cycles_done=cycles_done, latest_score=latest_score)
            return

        _set_status(conn, qid, "running",
                    cycles_done=cycles_done, latest_score=latest_score)

    if aborted:
        pass  # circuit breaker가 이미 failed로 설정
    elif _is_cancelled(conn, qid):
        print(f"[daemon] queue_id={qid} cancelled (detected post-cycle)")
    elif cycles_done < n_cycles:
        # 리스 소진, 예산 남음 — pending으로 되돌려 다른 큐 항목에 순서를 넘긴다.
        # last_leased_at은 _process 진입 시 이미 now()로 갱신돼 있어 라운드로빈
        # 정렬에서 자연히 뒤로 밀린다.
        _set_status(conn, qid, "pending", cycles_done=cycles_done, latest_score=latest_score)
        print(f"[daemon] queue_id={qid} lease exhausted at {cycles_done}/{n_cycles} — requeued")
    else:
        status, err = _final_status(successes, skipped, failed_cycles)
        _set_status(conn, qid, status,
                    ended_at=datetime.now(timezone.utc),
                    cycles_done=cycles_done,
                    latest_score=latest_score,
                    **({"error": err} if err else {}))
        suffix = f" ({failed_cycles} failed)" if failed_cycles else ""
        print(f"[daemon] queue_id={qid} {status} latest_score={latest_score}{suffix}")

        # bootstrap 배치가 최소 1 cycle이라도 성공했으면 baseline 확립을 시도한다.
        # 이미 확정 파이프라인이 있으면(재부트스트랩 등) establish_bootstrap_baseline이
        # 내부에서 스킵한다 — airflow 모드는 train이 로드 안 돼 있으므로 여기서 새로 읽는다.
        # 실패해도 daemon 루프 자체는 계속돼야 하므로 예외를 여기서 흡수한다.
        if stage == "bootstrap" and successes > 0:
            try:
                bootstrap_train = train if train is not None else load_train(comp)
                established = establish_bootstrap_baseline(
                    conn,
                    competition_id=comp.COMPETITION_ID,
                    train=bootstrap_train,
                    target_col=comp.TARGET,
                    metric=comp.METRIC,
                    n_splits=getattr(comp, "N_SPLITS", 5),
                    is_classification=comp.IS_CLASSIFICATION,
                    cpu_budget_secs=getattr(comp, "CPU_BUDGET_SECS", None),
                )
                print(
                    f"[daemon] queue_id={qid} bootstrap baseline "
                    f"{'established' if established else 'not established (existing baseline or not confirmed)'}"
                )
            except Exception as exc:
                print(f"[daemon] queue_id={qid} bootstrap baseline establishment failed: {exc}")

    state.update(current_queue_id=None, current_competition=None,
                 current_cycle=0, current_n_cycles=0)


def main() -> None:
    from config.settings import require_llm_env
    require_llm_env()

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
        print("[daemon] direct mode — AIRFLOW_URL not set, running single-attempt test cycles in-process")

    state = DaemonState()
    api_thread = threading.Thread(target=_run_api, args=(state,), daemon=True)
    api_thread.start()

    conn = connect()
    stuck = conn.execute(
        "UPDATE raw.cycle_queue SET status = 'pending', started_at = NULL "
        "WHERE status = 'running' RETURNING queue_id"
    ).fetchall()
    if stuck:
        ids = [r[0] for r in stuck]
        print(f"[daemon] reset {len(ids)} stuck 'running' → 'pending': {ids}")
    pacer.restore_from_db(conn)
    print("[daemon] started — polling raw.cycle_queue")

    while _running:
        _sweep_stale_submissions(conn)
        _sweep_low_gain_lessons(conn)
        _sweep_queue_refill(conn)
        item = _pop_pending(conn)
        if item is None:
            time.sleep(POLL_INTERVAL_SEC)
            continue
        _process(conn, item, pacer, state)

    conn.close()
    from store.db import close_pool
    close_pool()
    print("[daemon] stopped")


if __name__ == "__main__":
    main()
