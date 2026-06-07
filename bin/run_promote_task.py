"""Super-cycle promote step — Airflow task 4 of 4.

Picks winner from 3 attempts, updates was_promoted, reflects winner (BON-96 gate).

Usage (container):
    uv run python -m bin.run_promote_task --queue-id <id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-id", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))

    from store.db import connect
    from agents.reflector import AttemptContext, reflect
    from memory.retriever import EmbeddingUnavailableError
    from store.s3_code import download as _code_download
    from cycle.run import _CODE_HEADER_SEP

    conn = connect(apply_schema=False)

    ctx_row = conn.execute(
        "SELECT super_cycle_id, competition_id FROM raw.super_cycle_context WHERE queue_id = %s",
        [args.queue_id],
    ).fetchone()
    if not ctx_row:
        print(f"[run_promote_task] no context for queue_id={args.queue_id}", file=sys.stderr)
        sys.exit(1)

    super_cycle_id, competition_id = ctx_row

    rows = conn.execute(
        """
        SELECT attempt_id, gain_vs_best, cv_score, label, error_trace,
               hypothesis, action_type, reflection_ids, cv_fold_var, code_path
        FROM raw.attempts
        WHERE super_cycle_id = %s
        ORDER BY run_ts
        """,
        [super_cycle_id],
    ).fetchall()

    if not rows:
        print(f"[run_promote_task] no attempts for super_cycle_id={super_cycle_id[:8]}", file=sys.stderr)
        conn.close()
        return

    # Pick winner: max gain_vs_best (includes negative), fallback to any with cv_score
    with_gain = [(i, r[1]) for i, r in enumerate(rows) if r[1] is not None]
    winner_idx = max(with_gain, key=lambda x: x[1])[0] if with_gain else None
    if winner_idx is None:
        winner_idx = next((i for i, r in enumerate(rows) if r[2] is not None), None)

    for i, r in enumerate(rows):
        conn.execute(
            "UPDATE raw.attempts SET was_promoted = %s WHERE attempt_id = %s",
            [i == winner_idx, r[0]],
        )

    if winner_idx is None:
        print(f"[run_promote_task] super_cycle={super_cycle_id[:8]} — all errored, no winner")
        conn.close()
        return

    (attempt_id, gain_vs_best, cv_score, label, error_trace,
     hypothesis, action_type, reflection_ids, cv_fold_var, code_path) = rows[winner_idx]

    print(
        f"[run_promote_task] super_cycle={super_cycle_id[:8]}"
        f" winner={attempt_id[:8]} cv={cv_score} label={label}"
        f" n_attempts={len(rows)}"
    )

    # BON-96 gate: reflect only for jump/regression/error
    if label not in ("jump", "regression") and error_trace is None:
        conn.close()
        return

    source = ""
    if code_path:
        content = _code_download(code_path) or ""
        sep = _CODE_HEADER_SEP + "\n"
        source = content.split(sep, 1)[1].strip() if sep in content else content

    ctx = AttemptContext(
        hypothesis=hypothesis or "",
        action_type=action_type or "",
        code=source,
        cv_score=cv_score or 0.0,
        cv_fold_var=cv_fold_var or 0.0,
        gain_vs_best=gain_vs_best,
        label=label or "regression",
        retrieved_ids=reflection_ids or [],
        feature_importance=None,
        error_trace=error_trace,
    )
    try:
        output = reflect(conn, attempt_id=attempt_id, competition_id=competition_id, context=ctx)
        print(f"[run_promote_task] reflection_id={output.reflection_id}")
    except EmbeddingUnavailableError as exc:
        print(f"[run_promote_task] reflect skipped — embedding unavailable: {exc}")

    conn.close()


if __name__ == "__main__":
    main()
