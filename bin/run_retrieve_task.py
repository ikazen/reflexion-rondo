"""Super-cycle retrieve step — Airflow task 1 of 4.

Generates super_cycle_id, fetches lessons + prev_best_cv,
stores in raw.super_cycle_context for attempt/promote tasks to read.

context is keyed by --run-id (Airflow dag_run_id, unique per cycle),
not --queue-id (shared by every cycle of the same super-cycle queue item —
keying by queue_id let concurrent cycles clobber/steal each other's context row).

Usage (container):
    uv run python -m bin.run_retrieve_task --competition s4e1 --stage reflexion --queue-id <id> --run-id <run_id>
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent

_ERROR_LESSON_K = 3  # 코사인 무관 강제 포함할 최근 failure 교훈 수


def _merge_lessons(lessons: list[dict], extra: list[dict]) -> list[dict]:
    """extra를 lessons 뒤에 병합, reflection_id 중복은 lessons 쪽을 우선한다."""
    seen = {l["reflection_id"] for l in lessons}
    return lessons + [l for l in extra if l["reflection_id"] not in seen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", required=True)
    parser.add_argument("--stage", "-s", required=True)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--k-retrieve", type=int, default=5)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))

    from store.db import connect, ensure_competition
    from cycle.run import _build_retrieval_query, _prev_best, _recent_failure_summary
    from cycle.action_optimizer import assign_super_cycle_actions
    from memory.retriever import EmbeddingUnavailableError, search, search_failure_lessons

    conn = connect(apply_schema=False)

    try:
        comp = importlib.import_module(f"config.competitions.{args.competition}")
    except ModuleNotFoundError as exc:
        print(f"[run_retrieve_task] competition config not found: {exc}", file=sys.stderr)
        sys.exit(1)

    ensure_competition(
        conn,
        competition_id=comp.COMPETITION_ID,
        name=comp.NAME,
        task_type=comp.TASK_TYPE,
        metric=comp.METRIC,
        metric_sign=comp.METRIC_SIGN,
    )

    competition_id = comp.COMPETITION_ID
    eda_card = comp.EDA_CARD

    super_cycle_id = str(uuid.uuid4())
    prev_best_cv = _prev_best(conn, competition_id)
    fail_summary = _recent_failure_summary(conn, competition_id)
    query = _build_retrieval_query(conn, competition_id, eda_card, fail_summary)
    print(f"[run_retrieve_task] run_id={args.run_id} queue_id={args.queue_id[:8]}"
          f" super_cycle_id={super_cycle_id[:8]} prev_best_cv={prev_best_cv}")
    if fail_summary:
        print(f"  fail_summary: {fail_summary[:120]}")
    try:
        lessons = search(conn, query, competition_id, k=args.k_retrieve)
    except EmbeddingUnavailableError as exc:
        print(f"[run_retrieve_task] embedding unavailable, proceeding with no lessons: {exc}")
        lessons = []
    failure_lessons = search_failure_lessons(conn, competition_id, k=_ERROR_LESSON_K)
    lessons = _merge_lessons(lessons, failure_lessons)
    assigned_actions = assign_super_cycle_actions(conn, competition_id, n_attempts=3)
    print(f"  assigned_actions: {assigned_actions}")

    conn.execute(
        """
        INSERT INTO raw.super_cycle_context
            (run_id, queue_id, super_cycle_id, competition_id, prev_best_cv, lessons, assigned_actions, created_at)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, now())
        ON CONFLICT (run_id) DO UPDATE SET
            queue_id         = EXCLUDED.queue_id,
            super_cycle_id   = EXCLUDED.super_cycle_id,
            prev_best_cv     = EXCLUDED.prev_best_cv,
            lessons          = EXCLUDED.lessons,
            assigned_actions = EXCLUDED.assigned_actions,
            created_at       = EXCLUDED.created_at
        """,
        [args.run_id, args.queue_id, super_cycle_id, competition_id, prev_best_cv,
         json.dumps(lessons), json.dumps(assigned_actions)],
    )

    # promote 스킵(attempt 하드 실패) 시 row 가 영구 누수되므로
    # retrieve 에서 매 사이클 TTL 청소. 실패해도 retrieve 자체는 성공시킨다.
    try:
        conn.execute(
            "DELETE FROM raw.super_cycle_context"
            " WHERE created_at < now() - interval '7 days'"
        )
    except Exception:
        print("[run_retrieve_task] super_cycle_context ttl sweep failed", file=sys.stderr)

    conn.close()
    print(f"[run_retrieve_task] done competition={args.competition} stage={args.stage} n_lessons={len(lessons)}")
    for i, l in enumerate(lessons):
        snippet = (l.get("embedded_text") or "")[:80].replace("\n", " ")
        score = l.get("score")
        score_str = f"{score:.3f}" if score is not None else "n/a(forced-failure)"
        print(f"  lesson[{i}] score={score_str} {snippet}")


if __name__ == "__main__":
    main()
