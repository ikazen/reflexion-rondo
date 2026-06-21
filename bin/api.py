"""reflexion-rondo daemon FastAPI 앱.

엔드포인트:
  GET  /api/heartbeat
  GET  /api/health
  GET  /api/competitions
  POST /api/competitions
  GET  /api/attempts
  GET  /api/attempts/{id}
  GET  /api/lessons
  GET  /api/cold-start
  GET  /api/queue
  POST /api/queue
  PATCH /api/queue/{id}
  POST /api/submissions
  POST /api/submissions/auto
  GET  /api/submissions
  GET  /api/submissions/{id}

Postgres 연결은 호출 측(daemon)이 주입한다.
DaemonState는 daemon 메인 루프가 갱신하고 API가 읽는 공유 객체.
"""
from __future__ import annotations

import contextlib
import csv as csv_module
import importlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from store.db import PgConn

ROOT = Path(__file__).resolve().parent.parent
_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")


# ---------------------------------------------------------------------------
# TTL 캐시 (read-only GET 엔드포인트용)
# ---------------------------------------------------------------------------

class _TTLCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, ttl: float) -> tuple[object, bool]:
        with self._lock:
            entry = self._store.get(key)
            if entry and time.monotonic() - entry[0] < ttl:
                return entry[1], True
        return None, False

    def set(self, key: str, val: object) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), val)

    def drop(self, prefix: str) -> None:
        with self._lock:
            for k in [k for k in self._store if k.startswith(prefix)]:
                del self._store[k]


_cache = _TTLCache()


# ---------------------------------------------------------------------------
# 공유 상태 (daemon 메인 루프 ↔ API 스레드)
# ---------------------------------------------------------------------------

@dataclass
class DaemonState:
    current_queue_id: str | None = None
    current_competition: str | None = None
    current_cycle: int = 0
    current_n_cycles: int = 0
    last_cycle_at: datetime | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "current_queue_id": self.current_queue_id,
                "current_competition": self.current_competition,
                "current_cycle": self.current_cycle,
                "current_n_cycles": self.current_n_cycles,
                "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            }


# ---------------------------------------------------------------------------
# Pydantic 모델
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    competition: str


class EnqueueRequest(BaseModel):
    competition: str
    stage: str = "reflexion"
    n_cycles: int = 1
    priority: int = 0


class QueuePatchRequest(BaseModel):
    priority: int | None = None
    status: Literal["cancelled"] | None = None


class SubmitRequest(BaseModel):
    competition: str
    attempt_id: str | None = None
    message: str | None = None


class AutoSubmitRequest(BaseModel):
    window_hours: int = 24


# ---------------------------------------------------------------------------
# Kaggle 제출 백그라운드 워커
# ---------------------------------------------------------------------------

_TERMINAL = frozenset({"complete", "error", "invalid"})


def _kaggle_submit(
    submission_id: str,
    competition_id: str,
    competition_slug: str,
    attempt_id: str | None,
    message: str,
) -> None:
    """CSV 생성 + kaggle 제출. 폴링은 /refresh 엔드포인트(DAG)가 담당."""
    from store.db import connect

    def _update(fields: dict) -> None:
        c = connect(apply_schema=False)
        sets = ", ".join(f"{k} = %s" for k in fields)
        c.execute(
            f"update raw.kaggle_submissions set {sets} where submission_id = %s",
            list(fields.values()) + [submission_id],
        )
        c.close()

    cmd = [
        "uv", "run", "python", "-m", "bin.submit",
        "--competition", competition_slug,
        "--submit", "--message", message,
    ]
    if attempt_id:
        cmd += ["--attempt-id", attempt_id]

    _update({"status": "submitting", "checked_at": datetime.now(timezone.utc)})

    try:
        with _kaggle_home_env() as env:
            if env is None:
                _update({"status": "error", "error": "kaggle token unavailable", "checked_at": datetime.now(timezone.utc)})
                return
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT),
                env=env,
            )
    except subprocess.TimeoutExpired:
        _update({"status": "error", "error": "submit timed out (600s)", "checked_at": datetime.now(timezone.utc)})
        return

    if result.returncode != 0:
        _update({"status": "error", "error": result.stderr[:2000], "checked_at": datetime.now(timezone.utc)})
        return

    csv_path: str | None = None
    for line in result.stdout.splitlines():
        if "submission saved:" in line:
            csv_path = line.split("submission saved:", 1)[1].strip()
            break

    fields: dict = {"status": "submitted", "checked_at": datetime.now(timezone.utc)}
    if csv_path:
        fields["csv_path"] = csv_path
    _update(fields)


_kaggle_token_cache: dict = {}
_kaggle_token_lock = threading.Lock()


def _get_fresh_kaggle_token() -> str | None:
    """Return valid OAuth access token, refreshing via kagglesdk if needed.

    credentials.json is bind-mounted read-only in the daemon container.
    kagglesdk's refresh_access_token() always calls save() → OSError on ro mount.
    We call generate_access_token() directly (no save()) and cache the result.
    """
    with _kaggle_token_lock:
        now = datetime.now(timezone.utc)
        cached_token = _kaggle_token_cache.get("token")
        cached_exp = _kaggle_token_cache.get("expires_at")
        if cached_token and cached_exp and cached_exp > now + timedelta(minutes=5):
            return cached_token
        try:
            import kagglesdk.kaggle_creds as _kc
            from kagglesdk.kaggle_client import KaggleClient

            creds_path = Path.home() / ".kaggle" / "credentials.json"
            if not creds_path.exists():
                return None
            creds_data = json.loads(creds_path.read_text())
            client = KaggleClient()
            creds = _kc.KaggleCredentials(client=client, refresh_token=creds_data.get("refresh_token"))
            resp = creds.generate_access_token()
            _kaggle_token_cache["token"] = resp.token
            _kaggle_token_cache["expires_at"] = now + timedelta(seconds=resp.expires_in)
            return resp.token
        except Exception as exc:
            print(f"  [kaggle] token refresh failed: {exc}")
            return None


@contextlib.contextmanager
def _kaggle_home_env():
    """kaggle CLI가 OAuth credentials.json 대신 access_token 파일을 쓰도록 HOME 오버라이드.

    kagglesdk가 credentials.json의 save()를 항상 호출 → read-only mount에서 OSError.
    OAuth를 완전히 우회하기 위해: temp HOME dir에 .kaggle/access_token만 두고
    HOME 환경변수를 해당 경로로 설정 → kaggle CLI가 credentials.json을 찾지 못해
    OAuth 건너뜀 → access_token 파일로 직접 인증.
    """
    token = _get_fresh_kaggle_token()
    if not token:
        yield None
        return
    tmp = Path(tempfile.mkdtemp(prefix="kaggle_home_"))
    try:
        (tmp / ".kaggle").mkdir()
        (tmp / ".kaggle" / "access_token").write_text(token)
        yield {**os.environ, "HOME": str(tmp)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _poll_kaggle_once(
    competition_id: str,
    message: str,
) -> tuple[str, float | None]:
    """kaggle submissions 1회 조회 → (status, lb_score).

    반환 status: 'complete' | 'error' | 'invalid' | 'pending'
    kaggle CLI 오류 시 'pending' 반환 (재시도 위임).
    """
    try:
        with _kaggle_home_env() as env:
            if env is None:
                print("  [poll/warn] kaggle token unavailable, skipping")
                return "pending", None
            poll = subprocess.run(
                ["uv", "run", "kaggle", "competitions", "submissions",
                 "-c", competition_id, "--csv"],
                capture_output=True, text=True, timeout=30, cwd=str(ROOT),
                env=env,
            )
        if poll.returncode != 0:
            print(f"  [poll/warn] kaggle CLI rc={poll.returncode}: {(poll.stderr or poll.stdout)[:200]!r}")
            return "pending", None

        # kaggle CLI may print Warning lines to stdout before CSV — skip them
        csv_lines = [l for l in poll.stdout.splitlines() if not l.startswith("Warning:")]
        reader = csv_module.DictReader(io.StringIO("\n".join(csv_lines)))
        matched = False
        for row in reader:
            if (row.get("description") or "").strip() != message:
                continue
            matched = True
            status = (row.get("status") or "").lower()
            if status.endswith("complete"):
                raw_score = row.get("publicScore") or row.get("public score") or ""
                lb_score: float | None = None
                try:
                    lb_score = float(raw_score)
                except (ValueError, TypeError):
                    pass
                return "complete", lb_score
            if status.endswith("error") or status.endswith("invalid"):
                return status.rsplit(".", 1)[-1], None
            return "pending", None  # 아직 채점 중
        if not matched:
            print(f"  [poll/warn] no kaggle row matched description={message!r}")
    except Exception as exc:
        print(f"  [poll/warn] exception: {exc}")
    return "pending", None


# ---------------------------------------------------------------------------
# 자동 제출 헬퍼 (create_app 밖 — 공유 가능)
# ---------------------------------------------------------------------------

def _start_submission(
    conn: PgConn,
    competition_slug: str,
    competition_id: str,
    attempt_id: str | None,
    message: str,
) -> str:
    sid = str(uuid.uuid4())
    conn.execute(
        """
        insert into raw.kaggle_submissions
            (submission_id, competition_id, attempt_id, submitted_at, message, status)
        values (%s, %s, %s, %s, %s, 'queued')
        """,
        [sid, competition_id, attempt_id, datetime.now(timezone.utc), message],
    )
    threading.Thread(
        target=_kaggle_submit,
        args=(sid, competition_id, competition_slug, attempt_id, message),
        daemon=True,
    ).start()
    return sid


def _competition_id_to_slug() -> dict[str, str]:
    """config/competitions/*.py 스캔 → {competition_id: module_slug} 맵."""
    cached, hit = _cache.get("_comp_slug_map", ttl=3600)
    if hit:
        return cached  # type: ignore[return-value]
    result: dict[str, str] = {}
    comp_dir = ROOT / "config" / "competitions"
    for path in comp_dir.glob("*.py"):
        if path.stem.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"config.competitions.{path.stem}")
            cid = getattr(mod, "COMPETITION_ID", None)
            if cid:
                result[cid] = path.stem
        except Exception:
            continue
    _cache.set("_comp_slug_map", result)
    return result


def _best_attempt(conn: PgConn, competition_id: str) -> tuple[str, float] | None:
    row = conn.execute(
        """
        select a.attempt_id, a.cv_score
        from raw.attempts a
        join raw.competitions c using (competition_id)
        where a.competition_id = %s
          and a.cv_score is not null
          and a.error_trace is null
        order by c.metric_sign * a.cv_score desc
        limit 1
        """,
        [competition_id],
    ).fetchone()
    return (row[0], row[1]) if row else None


def _last_submitted_attempt(conn: PgConn, competition_id: str) -> str | None:
    row = conn.execute(
        """
        select attempt_id from raw.kaggle_submissions
        where competition_id = %s
          and status not in ('error', 'invalid', 'timeout')
        order by submitted_at desc
        limit 1
        """,
        [competition_id],
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# 앱 팩토리
# ---------------------------------------------------------------------------

def create_app(conn: PgConn, state: DaemonState) -> FastAPI:
    app = FastAPI(title="reflexion-rondo", version="v1")

    # ---- read endpoints -----------------------------------------------

    @app.get("/api/heartbeat")
    def heartbeat():
        snap = state.snapshot()
        return {"status": "running" if snap["current_queue_id"] else "idle", **snap}

    @app.get("/api/health")
    def health():
        from bin.healthcheck import run_checks
        checks = run_checks()
        overall = "ok" if all(v["status"] != "fail" for v in checks.values()) else "degraded"
        return {"overall": overall, "checks": checks}

    @app.get("/api/competitions")
    def get_competitions():
        cached, hit = _cache.get("competitions", ttl=60)
        if hit:
            return cached
        rows = conn.execute(
            "select competition_id, name, task_type, metric, metric_sign, start_ts from raw.competitions"
        ).fetchall()
        cols = ["competition_id", "name", "task_type", "metric", "metric_sign", "start_ts"]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set("competitions", result)
        return result

    @app.get("/api/attempts")
    def get_attempts(competition: str | None = None, limit: int = 50):
        limit = min(limit, 500)
        cache_key = f"attempts:{competition}:{limit}"
        cached, hit = _cache.get(cache_key, ttl=30)
        if hit:
            return cached
        where = "where a.competition_id = %s" if competition else ""
        params = [competition] if competition else []
        rows = conn.execute(
            f"""
            select
                a.attempt_id, a.competition_id, a.run_ts, a.stage,
                a.hypothesis, a.action_type, a.cv_score, a.cv_fold_var,
                a.label, a.gain_vs_best, a.error_trace, a.duration_sec,
                a.retries, a.code_path,
                s.best_so_far
            from raw.attempts a
            left join score_progression s using (attempt_id)
            {where}
            order by a.run_ts desc
            limit %s
            """,
            params + [limit],
        ).fetchall()
        cols = [
            "attempt_id", "competition_id", "run_ts", "stage",
            "hypothesis", "action_type", "cv_score", "cv_fold_var",
            "label", "gain_vs_best", "error_trace", "duration_sec",
            "retries", "code_path", "best_so_far",
        ]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/attempts/{attempt_id}")
    def get_attempt(attempt_id: str):
        row = conn.execute(
            """
            select
                attempt_id, competition_id, run_ts, stage,
                hypothesis, action_type, cv_score, cv_fold_var,
                label, gain_vs_best, error_trace, duration_sec,
                retries, code_path, reflection_ids, retrieval_scores
            from raw.attempts
            where attempt_id = %s
            """,
            [attempt_id],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="attempt not found")
        cols = [
            "attempt_id", "competition_id", "run_ts", "stage",
            "hypothesis", "action_type", "cv_score", "cv_fold_var",
            "label", "gain_vs_best", "error_trace", "duration_sec",
            "retries", "code_path", "reflection_ids", "retrieval_scores",
        ]
        return dict(zip(cols, row))

    @app.get("/api/lessons")
    def get_lessons(
        competition: str | None = None,
        generality: str | None = None,
        limit: int = 50,
    ):
        limit = min(limit, 500)
        cache_key = f"lessons:{competition}:{generality}:{limit}"
        cached, hit = _cache.get(cache_key, ttl=30)
        if hit:
            return cached
        conditions = ["r.archived = false"]
        params: list = []
        if competition:
            conditions.append("(r.competition_id = %s or r.generality in ('L2_class', 'L3_general'))")
            params.append(competition)
        if generality:
            conditions.append("r.generality = %s")
            params.append(generality)
        where = "where " + " and ".join(conditions)
        rows = conn.execute(
            f"""
            select
                r.reflection_id, r.created_at, r.competition_id,
                r.embedded_text, r.full_lesson, r.generality,
                r.label, r.gain_vs_best,
                coalesce(i.times_applied, 0) as times_applied,
                coalesce(i.avg_gain, 0.0) as avg_gain
            from raw.reflections r
            left join reflection_impact i using (reflection_id)
            {where}
            order by r.created_at desc
            limit %s
            """,
            params + [limit],
        ).fetchall()
        cols = [
            "reflection_id", "created_at", "competition_id",
            "embedded_text", "full_lesson", "generality",
            "label", "gain_vs_best", "times_applied", "avg_gain",
        ]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    @app.get("/api/cold-start")
    def get_cold_start(competition: str | None = None, limit: int = 200):
        limit = min(limit, 1000)
        cache_key = f"cold_start:{competition}:{limit}"
        cached, hit = _cache.get(cache_key, ttl=30)
        if hit:
            return cached
        where = "where competition_id = %s" if competition else ""
        params = [competition] if competition else []
        rows = conn.execute(
            f"""
            select competition_id, attempt_no, run_ts, stage, cv_score, best_so_far
            from cold_start_progression
            {where}
            order by competition_id, attempt_no
            limit %s
            """,
            params + [limit],
        ).fetchall()
        cols = ["competition_id", "attempt_no", "run_ts", "stage", "cv_score", "best_so_far"]
        result = [dict(zip(cols, r)) for r in rows]
        _cache.set(cache_key, result)
        return result

    # ---- admin endpoints -----------------------------------------------

    @app.get("/api/queue")
    def get_queue(status: str | None = None, limit: int = 100):
        limit = min(limit, 500)
        where = "where status = %s" if status else ""
        params = [status] if status else []
        rows = conn.execute(
            f"""
            select queue_id, competition, stage, n_cycles, priority,
                   status, created_at, started_at, ended_at,
                   cycles_done, latest_score, error
            from raw.cycle_queue
            {where}
            order by priority desc, created_at asc
            limit %s
            """,
            params + [limit],
        ).fetchall()
        cols = [
            "queue_id", "competition", "stage", "n_cycles", "priority",
            "status", "created_at", "started_at", "ended_at",
            "cycles_done", "latest_score", "error",
        ]
        return [dict(zip(cols, r)) for r in rows]

    @app.post("/api/queue", status_code=201)
    def enqueue(body: EnqueueRequest):
        try:
            importlib.import_module(f"config.competitions.{body.competition}")
        except ModuleNotFoundError:
            raise HTTPException(status_code=422, detail=f"unknown competition: {body.competition}")
        qid = str(uuid.uuid4())
        conn.execute(
            """
            insert into raw.cycle_queue
                (queue_id, competition, stage, n_cycles, priority, status, created_at)
            values (%s, %s, %s, %s, %s, 'pending', %s)
            """,
            [qid, body.competition, body.stage, body.n_cycles,
             body.priority, datetime.now(timezone.utc)],
        )
        return {"queue_id": qid}

    @app.patch("/api/queue/{queue_id}")
    def patch_queue(queue_id: str, body: QueuePatchRequest):
        row = conn.execute(
            "select status from raw.cycle_queue where queue_id = %s",
            [queue_id],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="queue item not found")

        sets: list[str] = []
        vals: list = []
        if body.priority is not None:
            sets.append("priority = %s")
            vals.append(body.priority)
        if body.status == "cancelled":
            if row[0] not in ("pending", "running"):
                raise HTTPException(
                    status_code=409,
                    detail=f"cannot cancel item with status '{row[0]}'",
                )
            sets.append("status = %s")
            vals.append("cancelled")

        if not sets:
            raise HTTPException(status_code=422, detail="nothing to update")

        vals.append(queue_id)
        conn.execute(
            f"update raw.cycle_queue set {', '.join(sets)} where queue_id = %s",
            vals,
        )
        return {"queue_id": queue_id, "updated": True}

    # ---- submission endpoints -----------------------------------------

    @app.post("/api/submissions", status_code=201)
    def submit(body: SubmitRequest):
        try:
            comp = importlib.import_module(f"config.competitions.{body.competition}")
        except ModuleNotFoundError:
            raise HTTPException(status_code=422, detail=f"unknown competition: {body.competition}")

        msg = body.message or f"rondo attempt={body.attempt_id or 'best'}"
        sid = _start_submission(conn, body.competition, comp.COMPETITION_ID, body.attempt_id, msg)
        return {"submission_id": sid, "status": "queued"}

    @app.post("/api/submissions/auto", status_code=200)
    def auto_submit(body: AutoSubmitRequest):
        slug_map = _competition_id_to_slug()

        active_rows = conn.execute(
            """
            select distinct competition_id from raw.attempts
            where run_ts >= now() - make_interval(hours => %s)
              and cv_score is not null
              and error_trace is null
            """,
            [body.window_hours],
        ).fetchall()

        submitted = []
        skipped = []

        for (competition_id,) in active_rows:
            slug = slug_map.get(competition_id)
            if not slug:
                skipped.append({"competition": competition_id, "reason": "no config"})
                continue

            best = _best_attempt(conn, competition_id)
            if not best:
                skipped.append({"competition": competition_id, "reason": "no valid attempt"})
                continue
            best_attempt_id, best_cv = best

            last = _last_submitted_attempt(conn, competition_id)
            if last == best_attempt_id:
                skipped.append({"competition": competition_id, "reason": "best unchanged"})
                continue

            msg = f"auto cv={best_cv:.5f} attempt={best_attempt_id[:8]}"
            sid = _start_submission(conn, slug, competition_id, best_attempt_id, msg)
            submitted.append({
                "competition": competition_id,
                "slug": slug,
                "attempt_id": best_attempt_id,
                "submission_id": sid,
            })

        return {"submitted": submitted, "skipped": skipped}

    @app.get("/api/submissions")
    def get_submissions(competition: str | None = None, limit: int = 50):
        limit = min(limit, 200)
        where = "where competition_id = %s" if competition else ""
        params = [competition] if competition else []
        rows = conn.execute(
            f"""
            select submission_id, competition_id, attempt_id, submitted_at,
                   message, csv_path, status, lb_score, error, checked_at
            from raw.kaggle_submissions
            {where}
            order by submitted_at desc
            limit %s
            """,
            params + [limit],
        ).fetchall()
        cols = [
            "submission_id", "competition_id", "attempt_id", "submitted_at",
            "message", "csv_path", "status", "lb_score", "error", "checked_at",
        ]
        return [dict(zip(cols, r)) for r in rows]

    @app.get("/api/submissions/{submission_id}")
    def get_submission(submission_id: str):
        row = conn.execute(
            """
            select submission_id, competition_id, attempt_id, submitted_at,
                   message, csv_path, status, lb_score, error, checked_at
            from raw.kaggle_submissions
            where submission_id = %s
            """,
            [submission_id],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="submission not found")
        cols = [
            "submission_id", "competition_id", "attempt_id", "submitted_at",
            "message", "csv_path", "status", "lb_score", "error", "checked_at",
        ]
        return dict(zip(cols, row))

    @app.post("/api/submissions/{submission_id}/refresh")
    def refresh_submission(submission_id: str):
        row = conn.execute(
            """
            select submission_id, competition_id, attempt_id,
                   submitted_at, message, csv_path, status, lb_score, error, checked_at
            from raw.kaggle_submissions
            where submission_id = %s
            """,
            [submission_id],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="submission not found")
        cols = [
            "submission_id", "competition_id", "attempt_id", "submitted_at",
            "message", "csv_path", "status", "lb_score", "error", "checked_at",
        ]
        rec = dict(zip(cols, row))

        if rec["status"] in _TERMINAL:
            return rec

        kaggle_status, lb_score = _poll_kaggle_once(rec["competition_id"], rec["message"])
        if kaggle_status == "pending":
            conn.execute(
                "update raw.kaggle_submissions set checked_at = %s where submission_id = %s",
                [datetime.now(timezone.utc), submission_id],
            )
            rec["checked_at"] = datetime.now(timezone.utc)
            return rec

        fields: dict = {"status": kaggle_status, "checked_at": datetime.now(timezone.utc)}
        if kaggle_status == "complete" and lb_score is not None:
            fields["lb_score"] = lb_score
        elif kaggle_status in ("error", "invalid"):
            fields["error"] = f"kaggle: {kaggle_status}"

        sets = ", ".join(f"{k} = %s" for k in fields)
        conn.execute(
            f"update raw.kaggle_submissions set {sets} where submission_id = %s",
            list(fields.values()) + [submission_id],
        )

        if kaggle_status == "complete" and lb_score is not None and rec.get("attempt_id"):
            conn.execute(
                "update raw.attempts set lb_score = %s where attempt_id like %s and lb_score is null",
                [lb_score, f"{rec['attempt_id']}%"],
            )

        rec.update(fields)
        return rec

    return app
