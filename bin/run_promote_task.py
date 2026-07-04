"""Super-cycle promote step — Airflow task 4 of 4.

Picks winner from 3 attempts, updates was_promoted, reflects winner (BON-96 gate).
cross-seed confirmation + audit holdout 측정 후 confirmed=True일 때만 승격.

BON-237: context lookup/delete key is --run-id (Airflow dag_run_id), not --queue-id.
queue_id is shared by every cycle of the same super-cycle (max_active_runs=4 lets
several run concurrently) — keying by queue_id let a later cycle's retrieve
overwrite an earlier cycle's context row, and whichever promote ran first would
delete it out from under the other (hard "no context" failure, or worse: silently
promoting the wrong cycle's winner).

Usage (container):
    uv run python -m bin.run_promote_task --queue-id <id> --run-id <run_id>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--competition", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))

    import importlib
    import polars as pl
    from store.db import connect, insert_pipeline
    from agents.reflector import AttemptContext, reflect
    from config.settings import PROMOTE_CONFIRM_SEEDS
    from cycle.materialize import materialize_best_pipeline
    from cycle.promotion import confirm_and_measure
    from evaluator.harness import is_significant_gain, split_audit_holdout
    from memory.retriever import EmbeddingUnavailableError
    from store.s3_code import download as _code_download
    from store.s3_code import download_best_pipeline, upload_best_pipeline
    from cycle.run import _CODE_HEADER_SEP

    conn = connect(apply_schema=False)

    ctx_row = conn.execute(
        "SELECT super_cycle_id, competition_id FROM raw.super_cycle_context WHERE run_id = %s",
        [args.run_id],
    ).fetchone()
    if not ctx_row:
        print(f"[run_promote_task] no context for run_id={args.run_id}", file=sys.stderr)
        sys.exit(1)

    super_cycle_id, competition_id = ctx_row

    # BON-111: 컨텍스트를 다 읽은 시점이라 즉시 삭제 — 이후 모든 return 경로를
    # 일괄 커버(no attempts/no winner 포함). ON CONFLICT UPDATE라 재실행 시 재삽입 정상.
    # BON-237: run_id로 삭제 — queue_id는 같은 super-cycle의 다른(동시 실행) cycle과
    # 공유되므로 그 키로 지우면 다른 cycle의 아직 안 읽은 context까지 지워버린다.
    conn.execute(
        "DELETE FROM raw.super_cycle_context WHERE run_id = %s",
        [args.run_id],
    )

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

    print(f"[run_promote_task] super_cycle={super_cycle_id[:8]} n_attempts={len(rows)}")
    for i, r in enumerate(rows):
        winner_mark = "*" if i == winner_idx else " "
        print(f"  [{winner_mark}{i}] {r[0][:8]} action={r[9]} cv={r[2]} gain={r[1]} label={r[3]}")

    if winner_idx is None:
        print("  -> all errored, no winner")
        conn.close()
        return

    print(f"  -> promoted {rows[winner_idx][0][:8]} (gain={rows[winner_idx][1]})")

    import json as _json
    import uuid as _uuid

    winner_row = rows[winner_idx]
    winner_gain = winner_row[1]
    winner_cv_fold_var = winner_row[8] or 0.0
    winner_error = winner_row[4]
    winner_code_path = winner_row[9]
    if is_significant_gain(winner_gain, winner_cv_fold_var) and not winner_error and winner_code_path:
        winner_content = _code_download(winner_code_path) or ""
        sep = _CODE_HEADER_SEP + "\n"
        winner_source = winner_content.split(sep, 1)[1].strip() if sep in winner_content else winner_content
        if winner_source:
            # train 로드 + split — cross-seed 확인 및 holdout 측정을 위해
            comp_row = conn.execute(
                "select task_type, metric from raw.competitions where competition_id = %s",
                [competition_id],
            ).fetchone()
            train90: pl.DataFrame | None = None
            holdout10: pl.DataFrame | None = None
            try:
                comp = importlib.import_module(f"config.competitions.{args.competition}")
                if getattr(comp, "COMPETITION_ID", None) != competition_id:
                    print(
                        f"[run_promote_task] WARNING: comp.COMPETITION_ID={comp.COMPETITION_ID!r}"
                        f" != DB competition_id={competition_id!r}",
                        file=sys.stderr,
                    )
                s3_path = getattr(comp, "S3_DATA_PATH", None)
                if s3_path and _MINIO_ENDPOINT:
                    full_train = pl.read_csv(
                        f"{_MINIO_ENDPOINT}/kaggle/{s3_path}train.csv"
                    ).drop(comp.DROP_COLS)
                else:
                    full_train = pl.read_csv(comp.DATA_DIR / "train.csv").drop(comp.DROP_COLS)
                train90, holdout10 = split_audit_holdout(
                    full_train, comp.TARGET, comp.IS_CLASSIFICATION
                )
            except Exception as exc:
                print(f"[run_promote_task] train 로드 실패 — holdout/confirm 스킵: {exc}")
                train90 = None

            current_best = download_best_pipeline(competition_id)
            if train90 is not None:
                is_classification = comp.IS_CLASSIFICATION if comp_row else True
                n_splits = getattr(comp, "N_SPLITS", 5) if comp_row else 5
                confirm = confirm_and_measure(
                    source=winner_source,
                    best_source=current_best,
                    train90=train90,
                    holdout10=holdout10,
                    target_col=comp.TARGET,
                    metric=comp.METRIC,
                    n_splits=n_splits,
                    seed=42,
                    is_classification=is_classification,
                    confirm_seeds=PROMOTE_CONFIRM_SEEDS,
                )
                if confirm.holdout_score is not None:
                    conn.execute(
                        "UPDATE raw.attempts SET holdout_score = %s WHERE attempt_id = %s",
                        [confirm.holdout_score, winner_row[0]],
                    )
                if confirm.seed_gains:
                    conn.execute(
                        "UPDATE raw.attempts SET confirm_seed_gains = %s WHERE attempt_id = %s",
                        [_json.dumps(confirm.seed_gains), winner_row[0]],
                    )
                if not confirm.confirmed:
                    print(f"[run_promote_task] cross-seed 미확인 — 승격 스킵 winner={winner_row[0][:8]}")
                    # 승격만 스킵 — 아래 promotion 가드(confirm.confirmed)가 막고, reflect 루프는 계속 실행
            else:
                confirm = None

            if train90 is None or (confirm is not None and confirm.confirmed):
                fp_row = conn.execute(
                    "select fingerprint from raw.competitions where competition_id = %s",
                    [competition_id],
                ).fetchone()
                fp_val = fp_row[0] if fp_row and fp_row[0] else {}
                fp_dict = fp_val if isinstance(fp_val, dict) else _json.loads(fp_val)
                # materialize 먼저 → 해시는 실제 MinIO 업로드 내용(submit.py가 exec하는
                # 문자열) 기준 (BON-255). raw.pipelines.code(winner source)와는 다른 문자열.
                materialized = materialize_best_pipeline(current_best, winner_source)
                pipeline_sha256 = hashlib.sha256(materialized.encode()).hexdigest()
                with conn.transaction():
                    insert_pipeline(
                        conn,
                        pipeline_id=str(_uuid.uuid4()),
                        attempt_id=winner_row[0],
                        competition_id=competition_id,
                        fingerprint_snapshot=fp_dict,
                        code=winner_source,
                        cv_score=winner_row[2],
                        gain_vs_best=winner_gain,
                        pipeline_sha256=pipeline_sha256,
                    )
                upload_best_pipeline(competition_id, materialized)
                print(f"[run_promote_task] best pipeline materialized for {competition_id}")

    for i, r in enumerate(rows):
        (attempt_id, gain_vs_best, cv_score, label, error_trace,
         hypothesis, action_type, reflection_ids, cv_fold_var, code_path) = r

        is_winner = (i == winner_idx)
        # winner: jump/regression/error만 reflect (neutral은 교훈 불명확)
        # loser: neutral 포함 전부 reflect ("이 시도는 효과 없었다"도 학습 신호)
        if is_winner and label not in ("jump", "regression") and error_trace is None:
            continue

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
        role = "winner" if is_winner else "loser"
        try:
            output = reflect(conn, attempt_id=attempt_id, competition_id=competition_id, context=ctx)
            print(f"[run_promote_task] reflect {role} {attempt_id[:8]} → reflection_id={output.reflection_id}")
        except EmbeddingUnavailableError as exc:
            print(f"[run_promote_task] reflect {role} {attempt_id[:8]} skipped — embedding unavailable: {exc}")
        except ValueError as exc:
            print(f"[run_promote_task] reflect {role} {attempt_id[:8]} skipped — LLM error: {exc}")

    conn.close()


if __name__ == "__main__":
    main()
