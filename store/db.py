import os
import threading
from pathlib import Path

import psycopg2
import psycopg2.pool
from pgvector.psycopg2 import register_vector

_DSN = os.getenv(
    "RONDO_DB_URL",
    "postgresql://rondo:rondo@localhost:5432/rondo",
)
_SCHEMA = Path(__file__).parent / "schema.sql"
_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(1, 10, dsn=_DSN)
    return _pool


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

    def execute(self, query: str, params=None) -> _Result:
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

    def close(self) -> None:
        _get_pool().putconn(self._conn)


def _apply_schema(raw: psycopg2.extensions.connection) -> None:
    schema = _SCHEMA.read_text()
    statements = [s.strip() for s in schema.split(";") if s.strip()]
    with raw.cursor() as cur:
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
    cols = ", ".join(row.keys())
    placeholders = ", ".join("%s" for _ in row)
    conn.execute(
        f"INSERT INTO raw.attempts ({cols}) VALUES ({placeholders})",
        list(row.values()),
    )


def insert_pipeline(
    conn: PgConn,
    pipeline_id: str,
    attempt_id: str,
    competition_id: str,
    fingerprint_snapshot: dict,
    code: str,
    cv_score: float,
    gain_vs_best: float,
) -> None:
    import json
    conn.execute(
        """
        INSERT INTO raw.pipelines
            (pipeline_id, attempt_id, competition_id, fingerprint_snapshot, code, cv_score, gain_vs_best)
        VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
        ON CONFLICT (pipeline_id) DO NOTHING
        """,
        [pipeline_id, attempt_id, competition_id, json.dumps(fingerprint_snapshot), code, cv_score, gain_vs_best],
    )
