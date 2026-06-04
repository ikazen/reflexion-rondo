import os
from pathlib import Path
import duckdb

_DB_PATH = Path(
    os.getenv("RONDO_DB_PATH", str(Path(__file__).parent.parent / "runs" / "reflexion.duckdb"))
)
_SCHEMA = Path(__file__).parent / "schema.sql"


def connect() -> duckdb.DuckDBPyConnection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(_DB_PATH))
    conn.execute(_SCHEMA.read_text())
    return conn


def ensure_competition(
    conn: duckdb.DuckDBPyConnection,
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
        insert into raw.competitions (competition_id, name, task_type, metric, metric_sign, start_ts, fingerprint)
        values (?, ?, ?, ?, ?, now(), ?)
        on conflict (competition_id) do update set
            fingerprint = case
                when raw.competitions.fingerprint is not null
                 and raw.competitions.fingerprint != '{}'
                then raw.competitions.fingerprint
                else excluded.fingerprint
            end
        """,
        [competition_id, name, task_type, metric, metric_sign, fp_json],
    )


def insert_attempt(conn: duckdb.DuckDBPyConnection, row: dict) -> None:
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    conn.execute(
        f"insert into raw.attempts ({cols}) values ({placeholders})",
        list(row.values()),
    )


def insert_pipeline(
    conn: duckdb.DuckDBPyConnection,
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
        insert into raw.pipelines
            (pipeline_id, attempt_id, competition_id, fingerprint_snapshot, code, cv_score, gain_vs_best)
        values (?, ?, ?, ?, ?, ?, ?)
        on conflict (pipeline_id) do nothing
        """,
        [pipeline_id, attempt_id, competition_id, json.dumps(fingerprint_snapshot), code, cv_score, gain_vs_best],
    )
