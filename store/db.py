"""Postgres 연결 풀 + PgConn(DuckDB 호환 execute().fetchone()/fetchall() 인터페이스).

connect(apply_schema=True)가 기본 경로 — daemon만 이 기본값을 쓰고 Airflow task는
apply_schema=False로 스키마 재적용 경합을 피한다.
"""
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import psycopg2
import psycopg2.pool
from psycopg2 import sql as pgsql
from pgvector.psycopg2 import register_vector

_DSN = os.getenv("RONDO_DB_URL")
_SCHEMA = Path(__file__).parent / "schema.sql"
_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                dsn = _DSN
                if not dsn:
                    raise RuntimeError("RONDO_DB_URL env var is not set")
                _pool = psycopg2.pool.ThreadedConnectionPool(1, 10, dsn=dsn)
    return _pool


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


class _Result:
    def __init__(self, rows: list):
        self._rows = rows

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list:
        return self._rows


class PgConn:
    """Thin psycopg2 wrapper with DuckDB-compatible execute().fetchone()/fetchall() interface."""

    def __init__(self, raw: psycopg2.extensions.connection) -> None:
        self._conn = raw
        self._lock = threading.RLock()

    def execute(self, query: str | pgsql.Composable, params=None) -> _Result:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall() if cur.description else []
            cur.close()
        return _Result(rows)

    def __enter__(self) -> "PgConn":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self):
        """Atomic block: disables autocommit, commits on success, rolls back on error."""
        self._conn.autocommit = False
        try:
            yield self
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._conn.autocommit = True

    def close(self) -> None:
        _get_pool().putconn(self._conn)


_SCHEMA_LOCK_KEY = 7_463_100  # arbitrary stable int for pg_advisory_xact_lock


def _apply_schema(raw: psycopg2.extensions.connection) -> None:
    schema = _SCHEMA.read_text()
    statements = [s.strip() for s in schema.split(";") if s.strip()]
    with raw.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", [_SCHEMA_LOCK_KEY])
        for stmt in statements:
            cur.execute(stmt)
    raw.commit()


def connect(apply_schema: bool = True) -> PgConn:
    raw = _get_pool().getconn()
    raw.autocommit = True
    register_vector(raw)
    if apply_schema:
        raw.autocommit = False
        try:
            _apply_schema(raw)
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.autocommit = True
    return PgConn(raw)


def ensure_competition(
    conn: PgConn,
    competition_id: str,
    name: str,
    task_type: str,
    metric: str,
    metric_sign: int,
    fingerprint: dict | None = None,
) -> None:
    from evaluator.metrics import get as _get_metric
    _get_metric(metric)  # raises ValueError for unknown metric early
    import json
    fp_json = json.dumps(fingerprint or {})
    conn.execute(
        """
        INSERT INTO raw.competitions (competition_id, name, task_type, metric, metric_sign, start_ts, fingerprint)
        VALUES (%s, %s, %s, %s, %s, now(), %s::jsonb)
        ON CONFLICT (competition_id) DO UPDATE SET
            fingerprint = CASE
                WHEN raw.competitions.fingerprint IS NOT NULL
                 AND raw.competitions.fingerprint != '{}'::jsonb
                THEN raw.competitions.fingerprint
                ELSE EXCLUDED.fingerprint
            END
        """,
        [competition_id, name, task_type, metric, metric_sign, fp_json],
    )


def insert_attempt(conn: PgConn, row: dict) -> None:
    columns = list(row.keys())
    query = pgsql.SQL("INSERT INTO raw.attempts ({}) VALUES ({})").format(
        pgsql.SQL(", ").join(map(pgsql.Identifier, columns)),
        pgsql.SQL(", ").join(pgsql.Placeholder() * len(columns)),
    )
    conn.execute(query, list(row.values()))


def insert_pipeline(
    conn: PgConn,
    pipeline_id: str,
    attempt_id: str,
    competition_id: str,
    fingerprint_snapshot: dict,
    code: str,
    cv_score: float,
    gain_vs_best: float,
    pipeline_sha256: str | None = None,
    oof_preds: list | None = None,
    materialized_code: str | None = None,
) -> None:
    import hashlib
    import json
    # forward 경로에서 두 해시는 같다(둘 다 방금 만든 병합본의 sha) — 다른 건 #254
    # 백필 행뿐이다. materialized_origin='promote'로 백필 행과 구분한다.
    materialized_sha256 = (
        hashlib.sha256(materialized_code.encode()).hexdigest()
        if materialized_code is not None else None
    )
    conn.execute(
        """
        INSERT INTO raw.pipelines
            (pipeline_id, attempt_id, competition_id, fingerprint_snapshot, code, cv_score, gain_vs_best,
             pipeline_sha256, oof_preds, materialized_code, materialized_sha256, materialized_origin)
        VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb, %s, %s, 'promote')
        ON CONFLICT (pipeline_id) DO NOTHING
        """,
        [
            pipeline_id, attempt_id, competition_id, json.dumps(fingerprint_snapshot),
            code, cv_score, gain_vs_best, pipeline_sha256,
            json.dumps(oof_preds) if oof_preds is not None else None,
            materialized_code, materialized_sha256,
        ],
    )


def insert_tuned_params(
    conn: PgConn,
    id_: str,
    tuning_run_id: str,
    competition_id: str,
    model_type: str,
    member_index: int | None,
    params: dict,
    cv_score: float,
    baseline_cv_score: float,
    n_trials: int,
    improved: bool,
) -> None:
    """evaluator/tuner.py(#230)의 TunerResult 1건을 raw.tuned_params에 영속화한다.
    tuning_run_id는 한 번의 bin/tune_pipeline.py 실행(ensemble이면 멤버 여러 건)을
    묶는 키 — cycle/run.py:_latest_tuned_params가 "가장 최근 실행 전체"를 한 번에
    조회할 때 쓴다."""
    import json
    conn.execute(
        """
        INSERT INTO raw.tuned_params
            (id, tuning_run_id, competition_id, model_type, member_index, params,
             cv_score, baseline_cv_score, n_trials, improved)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
        """,
        [
            id_, tuning_run_id, competition_id, model_type, member_index,
            json.dumps(params), cv_score, baseline_cv_score, n_trials, improved,
        ],
    )
