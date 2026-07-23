"""BON-244: reflection_impact 뷰 baseline이 런타임 _prev_best(확정 파이프라인)와
일치하는지 실제 Postgres에서 검증. RONDO_DB_URL 미설정 시 skip — 이 repo는
DB-backed 테스트 하네스가 없어(전부 mock) 뷰 SQL 자체의 정합은 이 테스트가 유일한 검증.

로컬/CI에서 돌리려면:
    docker run -d --rm -e POSTGRES_PASSWORD=x -e POSTGRES_DB=rondo -p 5433:5432 pgvector/pgvector:pg16
    RONDO_DB_URL=postgresql://postgres:x@localhost:5433/rondo uv run pytest tests/test_reflection_impact_view.py
"""
from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RONDO_DB_URL"),
    reason="RONDO_DB_URL not set — DB-backed view test skipped",
)


@pytest.fixture
def conn():
    from store.db import connect

    c = connect(apply_schema=True)
    yield c
    c.close()


def test_gain_not_penalized_by_unconfirmed_winner(conn) -> None:
    """미확정(cross-seed 미확인) winner가 phantom 천장을 만들어 후속 attempt의
    gain을 깎으면 안 된다 — reflection_impact는 저장된 gain_vs_best(런타임 _prev_best
    기준)를 그대로 써야 한다.
    """
    competition_id = f"bon244-test-{uuid.uuid4().hex[:8]}"
    r_base, r_lucky, r_next = (f"{competition_id}-r_base", f"{competition_id}-r_lucky",
                                f"{competition_id}-r_next")
    try:
        conn.execute(
            "INSERT INTO raw.competitions (competition_id, name, task_type, metric, metric_sign)"
            " VALUES (%s, 'BON-244 test', 'binary', 'auc', 1)",
            [competition_id],
        )
        # a0: baseline attempt, cv=0.80, prev_best 없음 -> gain=0.00
        # a1: super-cycle winner, cv=0.90이지만 cross-seed 미확인 -> raw.pipelines엔 없음.
        #     runtime evaluator/harness.py가 prev_best=0.80 기준으로 gain=0.10 저장,
        #     run_promote_task가 confirm 이전에 was_promoted=TRUE로 마킹.
        # a2: 다음 super-cycle, prev_best는 여전히 0.80(a1 미확정이라 안 바뀜),
        #     cv=0.82 -> gain=0.02 (양수). 옛 뷰는 running-max(0.90 포함) 대비
        #     -0.08로 깎아 phantom 음수를 만들었다.
        # issue #58: reflection_impact가 gain_vs_best_relative를 집계한다. auc는
        # metric_class=binary_proba라 gain_vs_best_relative == gain_vs_best(패스스루).
        conn.execute(
            """
            INSERT INTO raw.attempts
                (attempt_id, competition_id, run_ts, stage, cv_score, gain_vs_best,
                 gain_vs_best_relative, reflection_ids, super_cycle_id, was_promoted)
            VALUES
                (%s, %s, now() - interval '3 hours', 'reflexion', 0.80, 0.00, 0.00, ARRAY[%s], 'sc1', NULL),
                (%s, %s, now() - interval '2 hours', 'reflexion', 0.90, 0.10, 0.10, ARRAY[%s], 'sc1', TRUE),
                (%s, %s, now() - interval '1 hours', 'reflexion', 0.82, 0.02, 0.02, ARRAY[%s], 'sc2', NULL)
            """,
            [
                f"{competition_id}-a0", competition_id, r_base,
                f"{competition_id}-a1", competition_id, r_lucky,
                f"{competition_id}-a2", competition_id, r_next,
            ],
        )

        rows = conn.execute(
            "SELECT reflection_id, avg_gain, jumps FROM reflection_impact"
            " WHERE reflection_id IN (%s, %s, %s)",
            [r_base, r_lucky, r_next],
        ).fetchall()
        by_id = {r[0]: (float(r[1]), r[2]) for r in rows}

        # 핵심 단언: r_next는 확정 baseline(0.80) 기준 +0.02 이며 jump로 카운트돼야 한다.
        # 옛(버그) 뷰였다면 running-max(0.90 포함) 대비 avg_gain=-0.08, jumps=0이었을 것.
        assert by_id[r_next] == (0.02, 1)
        assert by_id[r_lucky] == (0.10, 1)
        assert by_id[r_base] == (0.00, 0)
    finally:
        conn.execute(
            "DELETE FROM raw.attempts WHERE competition_id = %s", [competition_id]
        )
        conn.execute(
            "DELETE FROM raw.competitions WHERE competition_id = %s", [competition_id]
        )


def test_legacy_row_without_relative_gain_excluded(conn) -> None:
    """issue #58: gain_vs_best_relative가 없는(NULL) legacy row는 스케일을 신뢰할 수
    없으므로 reflection_impact 집계에서 제외돼야 한다 — raw gain_vs_best로 폴백하지 않는다."""
    competition_id = f"issue58-test-{uuid.uuid4().hex[:8]}"
    r_legacy = f"{competition_id}-r_legacy"
    try:
        conn.execute(
            "INSERT INTO raw.competitions (competition_id, name, task_type, metric, metric_sign)"
            " VALUES (%s, 'issue #58 test', 'regression', 'rmse', -1)",
            [competition_id],
        )
        conn.execute(
            """
            INSERT INTO raw.attempts
                (attempt_id, competition_id, run_ts, stage, cv_score, gain_vs_best,
                 gain_vs_best_relative, reflection_ids, super_cycle_id, was_promoted)
            VALUES
                (%s, %s, now(), 'reflexion', 100.0, -5000.0, NULL, ARRAY[%s], 'sc1', NULL)
            """,
            [f"{competition_id}-a0", competition_id, r_legacy],
        )
        rows = conn.execute(
            "SELECT reflection_id FROM reflection_impact WHERE reflection_id = %s",
            [r_legacy],
        ).fetchall()
        assert rows == []
    finally:
        conn.execute(
            "DELETE FROM raw.attempts WHERE competition_id = %s", [competition_id]
        )
        conn.execute(
            "DELETE FROM raw.competitions WHERE competition_id = %s", [competition_id]
        )
