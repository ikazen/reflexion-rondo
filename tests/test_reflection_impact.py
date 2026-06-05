from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

_SCHEMA = (Path(__file__).parent.parent / "store" / "schema.sql").read_text()


@pytest.fixture()
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute(_SCHEMA)
    c.execute(
        """
        insert into raw.competitions (competition_id, name, task_type, metric, metric_sign)
        values ('comp1', 'test', 'binary', 'auc', 1)
        """
    )
    return c


def _insert_attempt(
    conn: duckdb.DuckDBPyConnection,
    attempt_id: str,
    cv_score: float,
    reflection_ids: list[str] | None,
    stage: str = "reflexion",
    ts_offset: str = "0 seconds",
) -> None:
    conn.execute(
        f"""
        insert into raw.attempts
            (attempt_id, competition_id, run_ts, stage, cv_score, reflection_ids)
        values (%s, 'comp1', now() + interval '{ts_offset}', %s, %s,  %s)
        """,
        [attempt_id, stage, cv_score, reflection_ids],
    )


def test_avg_gain_positive_jump(conn: duckdb.DuckDBPyConnection) -> None:
    _insert_attempt(conn, "a1", 0.80, None,          ts_offset="0 seconds")
    _insert_attempt(conn, "a2", 0.85, ["r1"],        ts_offset="1 seconds")

    rows = conn.execute(
        "select reflection_id, avg_gain, jumps from reflection_impact"
    ).fetchall()

    assert len(rows) == 1
    rid, avg_gain, jumps = rows[0]
    assert rid == "r1"
    assert avg_gain > 0
    assert jumps == 1


def test_avg_gain_regression(conn: duckdb.DuckDBPyConnection) -> None:
    _insert_attempt(conn, "a1", 0.85, None,     ts_offset="0 seconds")
    _insert_attempt(conn, "a2", 0.80, ["r1"],   ts_offset="1 seconds")

    rows = conn.execute("select avg_gain from reflection_impact").fetchall()
    assert rows[0][0] < 0


def test_multiple_reflections_aggregated(conn: duckdb.DuckDBPyConnection) -> None:
    _insert_attempt(conn, "a1", 0.80, None,          ts_offset="0 seconds")
    _insert_attempt(conn, "a2", 0.85, ["r1", "r2"],  ts_offset="1 seconds")
    _insert_attempt(conn, "a3", 0.87, ["r1"],         ts_offset="2 seconds")

    result = {
        row[0]: row
        for row in conn.execute(
            "select reflection_id, times_applied, jumps from reflection_impact"
        ).fetchall()
    }

    assert result["r1"][1] == 2  # times_applied
    assert result["r1"][2] == 2  # both jumps
    assert result["r2"][1] == 1


def test_bootstrap_stage_excluded(conn: duckdb.DuckDBPyConnection) -> None:
    _insert_attempt(conn, "a1", 0.80, None,       ts_offset="0 seconds")
    _insert_attempt(conn, "a2", 0.90, ["r1"],     ts_offset="1 seconds", stage="bootstrap")

    rows = conn.execute("select * from reflection_impact").fetchall()
    assert len(rows) == 0


def test_first_attempt_has_no_prev_best(conn: duckdb.DuckDBPyConnection) -> None:
    # first reflexion attempt has no preceding row → gain_vs_best = null → excluded
    _insert_attempt(conn, "a1", 0.80, ["r1"], ts_offset="0 seconds")

    rows = conn.execute("select * from reflection_impact").fetchall()
    assert len(rows) == 0
