"""reflexion-rondo daemon FastAPI 앱.

엔드포인트:
  GET  /api/heartbeat
  GET  /api/competitions
  POST /api/competitions
  GET  /api/attempts
  GET  /api/attempts/{id}
  GET  /api/lessons
  GET  /api/cold-start
  GET  /api/queue
  POST /api/queue
  PATCH /api/queue/{id}

DuckDB 연결은 호출 측(daemon)이 주입한다.
DaemonState는 daemon 메인 루프가 갱신하고 API가 읽는 공유 객체.
"""
from __future__ import annotations

import importlib
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from store.db import PgConn

_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")


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


# ---------------------------------------------------------------------------
# 앱 팩토리
# ---------------------------------------------------------------------------

def create_app(conn: PgConn, state: DaemonState) -> FastAPI:
    app = FastAPI(title="reflexion-rondo", version="v1")

    # ---- read endpoints -----------------------------------------------

    @app.get("/api/heartbeat")
    def heartbeat():
        snap = state.snapshot()
        status = "running" if snap["current_queue_id"] else "idle"
        return {"status": status, **snap}

    @app.get("/api/competitions")
    def get_competitions():
        rows = conn.execute(
            "select competition_id, name, task_type, metric, metric_sign, start_ts from raw.competitions"
        ).fetchall()
        cols = ["competition_id", "name", "task_type", "metric", "metric_sign", "start_ts"]
        return [dict(zip(cols, r)) for r in rows]

    @app.post("/api/competitions", status_code=201)
    def register_competition(body: RegisterRequest):
        try:
            comp = importlib.import_module(f"config.competitions.{body.competition}")
        except ModuleNotFoundError:
            raise HTTPException(status_code=422, detail=f"unknown competition: {body.competition}")

        s3_path = getattr(comp, "S3_DATA_PATH", None)
        if s3_path and _MINIO_ENDPOINT:
            url = f"{_MINIO_ENDPOINT}/kaggle/{s3_path}train.csv"
            try:
                train = pl.read_csv(url).drop(comp.DROP_COLS)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"MinIO read failed: {e}")
        else:
            csv_path = comp.DATA_DIR / "train.csv"
            if not csv_path.exists():
                raise HTTPException(status_code=422, detail=f"train.csv not found: {csv_path}")
            train = pl.read_csv(csv_path).drop(comp.DROP_COLS)

        from evaluator.metrics import get as get_metric
        from store.db import ensure_competition
        from store.fingerprint import compute as compute_fingerprint
        from memory.transfer import find_similar_competitions, cold_start_lessons, bootstrap_seeds

        _, metric_sign, _ = get_metric(comp.METRIC)
        fp = compute_fingerprint(train, comp.TARGET, comp.TASK_TYPE, comp.METRIC, metric_sign)
        ensure_competition(
            conn,
            competition_id=comp.COMPETITION_ID,
            name=comp.NAME,
            task_type=comp.TASK_TYPE,
            metric=comp.METRIC,
            metric_sign=metric_sign,
            fingerprint=fp,
        )

        similar_with_dist = find_similar_competitions(conn, fp, exclude_id=comp.COMPETITION_ID, k=3)
        similar = [c for c, _ in similar_with_dist]
        lessons = cold_start_lessons(conn, similar, k=10)
        seeds = bootstrap_seeds(conn, similar, n=2)

        return {
            "competition_id": comp.COMPETITION_ID,
            "similar_competitions": similar,
            "lessons": len(lessons),
            "seeds": len(seeds),
        }

    @app.get("/api/attempts")
    def get_attempts(competition: str | None = None, limit: int = 50):
        limit = min(limit, 500)
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
        return [dict(zip(cols, r)) for r in rows]

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
        return [dict(zip(cols, r)) for r in rows]

    @app.get("/api/cold-start")
    def get_cold_start(competition: str | None = None):
        where = "where competition_id = %s" if competition else ""
        params = [competition] if competition else []
        rows = conn.execute(
            f"""
            select competition_id, attempt_no, run_ts, stage, cv_score, best_so_far
            from cold_start_progression
            {where}
            order by competition_id, attempt_no
            """,
            params,
        ).fetchall()
        cols = ["competition_id", "attempt_no", "run_ts", "stage", "cv_score", "best_so_far"]
        return [dict(zip(cols, r)) for r in rows]

    # ---- admin endpoints -----------------------------------------------

    @app.get("/api/queue")
    def get_queue():
        rows = conn.execute(
            """
            select queue_id, competition, stage, n_cycles, priority,
                   status, created_at, started_at, ended_at,
                   cycles_done, latest_score, error
            from raw.cycle_queue
            order by priority desc, created_at asc
            """
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

    return app
