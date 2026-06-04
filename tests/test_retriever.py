from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from memory import retriever

_SCHEMA = (Path(__file__).parent.parent / "store" / "schema.sql").read_text()
_DIM = 1024


def _make_vec(val: float) -> list[float]:
    # unit vector along first dimension scaled to val, rest zeros
    v = [0.0] * _DIM
    v[0] = val
    return v


def _norm_vec(val: float) -> list[float]:
    # normalized so cosine similarity is well-defined
    mag = abs(val) if val != 0 else 1.0
    v = [0.0] * _DIM
    v[0] = val / mag
    return v


@pytest.fixture()
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute(_SCHEMA)
    return c


def test_insert_and_search_returns_top_k(conn: duckdb.DuckDBPyConnection) -> None:
    vecs = {
        "r1": _norm_vec(1.0),   # most similar to query
        "r2": _norm_vec(-1.0),  # least similar
        "r3": _norm_vec(0.9),
    }

    def fake_embed(text: str) -> list[float]:
        return vecs.get(text, _norm_vec(1.0))

    with patch.object(retriever, "embed", side_effect=fake_embed):
        for rid, vec in vecs.items():
            conn.execute(
                """
                insert into raw.reflections
                    (reflection_id, created_at, attempt_id, competition_id,
                     embedded_text, embedding, full_lesson, generality,
                     label, gain_vs_best, archived)
                values (?, now(), 'a1', 'comp1', ?, ?, 'lesson', 'L3_general',
                        'neutral', 0.0, false)
                """,
                [rid, rid, vec],
            )

        results = retriever.search(conn, "r1", competition_id="comp1", k=2)

    ids = {r["reflection_id"] for r in results}
    assert len(results) == 2
    assert "r1" in ids   # 가장 유사한 항목은 항상 포함
    assert "r2" not in ids  # 부정 유사도 항목은 제외


def test_archived_excluded(conn: duckdb.DuckDBPyConnection) -> None:
    vec = _norm_vec(1.0)

    with patch.object(retriever, "embed", return_value=vec):
        conn.execute(
            """
            insert into raw.reflections
                (reflection_id, created_at, attempt_id, competition_id,
                 embedded_text, embedding, full_lesson, generality,
                 label, gain_vs_best, archived)
            values ('archived_r', now(), 'a1', 'comp1', 'x', ?, 'lesson',
                    'L3_general', 'neutral', 0.0, true)
            """,
            [vec],
        )
        results = retriever.search(conn, "x", competition_id="comp1", k=5)

    assert all(r["reflection_id"] != "archived_r" for r in results)


def test_meta_filter_l1_other_competition(conn: duckdb.DuckDBPyConnection) -> None:
    vec = _norm_vec(1.0)

    with patch.object(retriever, "embed", return_value=vec):
        conn.execute(
            """
            insert into raw.reflections
                (reflection_id, created_at, attempt_id, competition_id,
                 embedded_text, embedding, full_lesson, generality,
                 label, gain_vs_best, archived)
            values ('l1_other', now(), 'a1', 'other_comp', 'x', ?, 'lesson',
                    'L1_local', 'neutral', 0.0, false)
            """,
            [vec],
        )
        results = retriever.search(conn, "x", competition_id="comp1", k=5)

    assert all(r["reflection_id"] != "l1_other" for r in results)


def test_meta_filter_l1_same_competition(conn: duckdb.DuckDBPyConnection) -> None:
    vec = _norm_vec(1.0)

    with patch.object(retriever, "embed", return_value=vec):
        conn.execute(
            """
            insert into raw.reflections
                (reflection_id, created_at, attempt_id, competition_id,
                 embedded_text, embedding, full_lesson, generality,
                 label, gain_vs_best, archived)
            values ('l1_same', now(), 'a1', 'comp1', 'x', ?, 'lesson',
                    'L1_local', 'neutral', 0.0, false)
            """,
            [vec],
        )
        results = retriever.search(conn, "x", competition_id="comp1", k=5)

    assert any(r["reflection_id"] == "l1_same" for r in results)


def test_rerank_boosts_high_gain(conn: duckdb.DuckDBPyConnection) -> None:
    # r_low_gain has higher cosine sim but lower gain
    # r_high_gain has slightly lower cosine sim but high past gain
    vec_high_sim = _norm_vec(1.0)
    vec_low_sim = [0.0] * _DIM
    vec_low_sim[0] = 0.8
    vec_low_sim[1] = 0.6  # not normalized but that's fine for this test

    with patch.object(retriever, "embed", return_value=vec_high_sim):
        for rid, vec, gain in [
            ("r_low_gain", vec_high_sim, 0.0),
            ("r_high_gain", vec_low_sim, 0.5),
        ]:
            conn.execute(
                """
                insert into raw.reflections
                    (reflection_id, created_at, attempt_id, competition_id,
                     embedded_text, embedding, full_lesson, generality,
                     label, gain_vs_best, archived)
                values (?, now(), 'a1', 'comp1', ?, ?, 'lesson',
                        'L3_general', 'jump', ?, false)
                """,
                [rid, rid, vec, gain],
            )
        conn.execute(
            """
            insert into raw.competitions (competition_id, name, task_type, metric, metric_sign)
            values ('comp1', 'test', 'binary', 'auc', 1)
            """
        )
        # a0 establishes baseline (no reflection); a1 jumps using r_high_gain
        # gain = 1*(0.85 - 0.35) = 0.50 → avg_gain for r_high_gain = 0.50
        # score: r_high_gain = 0.8*(1+0.50)=1.20 > r_low_gain = 1.0*(1+0)=1.0
        conn.execute(
            """
            insert into raw.attempts
                (attempt_id, competition_id, run_ts, stage, cv_score, reflection_ids)
            values
                ('a0', 'comp1', now() - interval '1 second', 'reflexion', 0.35, null),
                ('a1', 'comp1', now(),                       'reflexion', 0.85, ['r_high_gain'])
            """
        )

        results = retriever.search(conn, "r_low_gain", competition_id="comp1", k=2)

    assert results[0]["reflection_id"] == "r_high_gain"
