from pathlib import Path
import duckdb

_DB_PATH = Path(__file__).parent.parent / "runs" / "reflexion.duckdb"
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
        on conflict (competition_id) do update set fingerprint = excluded.fingerprint
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
