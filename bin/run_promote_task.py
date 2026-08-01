"""Super-cycle promote step — Airflow task 4 of 4.

Picks winner from 3 attempts, updates was_promoted, reflects winner.
cross-seed confirmation + audit holdout 측정 후 confirmed=True일 때만 승격.

context lookup/delete key is --run-id (Airflow dag_run_id), not --queue-id.
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
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
# 병합본(materialize_best_pipeline 산출물) cv_score가 winner 자신의 기록된
# cv_score와 크게 다르면 병합 손상 신호 — 같은 seed·fold라 결정적
# 재현이면 거의 bit-identical해야 한다. 부동소수 연산차만 허용하는 엄격한 허용오차.
_MERGE_VERIFY_TOLERANCE = 1e-6


def main() -> None:
    # 이 프로세스는 Airflow DockerOperator가 별도 실행하는 진입점이라 부모의
    # 로깅 설정을 상속받지 않는다 — basicConfig 없이는 cycle/promotion.py의
    # 게이트 실패 로그(_LOG.warning 등)가 lastResort 핸들러에 의존하게 되는데,
    # 그마저도 없던 기간엔 INFO 로그가 전부 조용히 사라졌다(#73).
    logging.basicConfig(level=logging.INFO)

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
    from runtime.isolate import eval_isolated
    from store.s3_code import download as _code_download
    from store.s3_code import download_best_pipeline, upload_best_pipeline
    from store.train_data import load_train
    from cycle.run import _CODE_HEADER_SEP, _prev_best_fold_scores

    conn = connect(apply_schema=False)

    ctx_row = conn.execute(
        "SELECT super_cycle_id, competition_id FROM raw.super_cycle_context WHERE run_id = %s",
        [args.run_id],
    ).fetchone()
    if not ctx_row:
        print(f"[run_promote_task] no context for run_id={args.run_id}", file=sys.stderr)
        sys.exit(1)

    super_cycle_id, competition_id = ctx_row

    # 컨텍스트를 다 읽은 시점이라 즉시 삭제 — 이후 모든 return 경로를
    # 일괄 커버(no attempts/no winner 포함). ON CONFLICT UPDATE라 재실행 시 재삽입 정상.
    # run_id로 삭제 — queue_id는 같은 super-cycle의 다른(동시 실행) cycle과
    # 공유되므로 그 키로 지우면 다른 cycle의 아직 안 읽은 context까지 지워버린다.
    conn.execute(
        "DELETE FROM raw.super_cycle_context WHERE run_id = %s",
        [args.run_id],
    )

    rows = conn.execute(
        """
        SELECT attempt_id, gain_vs_best, cv_score, label, error_trace,
               hypothesis, action_type, reflection_ids, cv_fold_var, code_path,
               fold_scores
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
        print(f"  [{winner_mark}{i}] {r[0][:8]} action={r[6]} cv={r[2]} gain={r[1]} label={r[3]}")

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
    winner_fold_scores = winner_row[10]

    # paired per-fold 검정용 metric_sign + baseline fold_scores.
    # 이 시점엔 comp 모듈을 아직 import 안 했으므로(뒤에서 필요할 때 import) DB에서 바로 조회.
    _sign_row = conn.execute(
        "select metric_sign from raw.competitions where competition_id = %s",
        [competition_id],
    ).fetchone()
    _metric_sign = _sign_row[0] if _sign_row and _sign_row[0] is not None else 1
    _baseline_fold_scores = _prev_best_fold_scores(
        conn, competition_id, exclude_attempt_id=winner_row[0]
    )

    _stage1_significant = is_significant_gain(
        winner_gain, winner_cv_fold_var,
        candidate_fold_scores=winner_fold_scores,
        baseline_fold_scores=_baseline_fold_scores,
        metric_sign=_metric_sign,
    )
    print(
        f"[run_promote_task] gate stage1: significant={_stage1_significant} "
        f"gain={winner_gain} cv_fold_var={winner_cv_fold_var} "
        f"candidate_folds={len(winner_fold_scores) if winner_fold_scores else 0} "
        f"baseline_folds={len(_baseline_fold_scores) if _baseline_fold_scores else 0} "
        f"has_error={bool(winner_error)} has_code={bool(winner_code_path)}"
    )

    if _stage1_significant and not winner_error and winner_code_path:
        winner_content = _code_download(winner_code_path) or ""
        sep = _CODE_HEADER_SEP + "\n"
        winner_source = winner_content.split(sep, 1)[1].strip() if sep in winner_content else winner_content
        if winner_source:
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
                full_train = load_train(comp)
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
                    reason = "holdout 악화" if confirm.holdout_regressed else "cross-seed 미확인"
                    print(f"[run_promote_task] {reason} — 승격 스킵 winner={winner_row[0][:8]}")
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
                # 문자열) 기준. raw.pipelines.code(winner source)와는 다른 문자열.
                materialized = materialize_best_pipeline(current_best, winner_source)
                pipeline_sha256 = hashlib.sha256(materialized.encode()).hexdigest()

                # merge-verify — 병합본을 실제로 1회 평가해 winner 자신의 cv_score와
                # 어긋나지 않는지 확인(정적 AST 검증만으로는 병합 손상을 못 잡음).
                # train90 없으면(train 로드 실패) 확인 불가 — 기존 confirm/holdout 스킵과
                # 같은 원칙으로 검증을 건너뛰고 진행(보수적으로 막지 않음, 기존 동작 유지).
                merge_ok = True
                merge_oof_preds = None
                if train90 is not None:
                    merge_eval = eval_isolated(
                        source=materialized,
                        train=train90,
                        target_col=comp.TARGET,
                        metric=comp.METRIC,
                        prev_best=None,
                        n_splits=n_splits,
                        seed=42,
                        is_classification=is_classification,
                        collect_oof=True,  # 이 1회 eval에 얹어 OOF 확보(추가 비용 없음)
                    )
                    if merge_eval.error_trace or merge_eval.cv_score is None:
                        merge_ok = False
                        # [:200] 절단이 Airflow 로그에서 실제 예외를 가려 원인을 못 잡은
                        # 전례가 있어 전체 출력.
                        print(
                            "[run_promote_task] merge-verify 실패(평가 에러) — 승격 스킵 "
                            f"competition={competition_id} winner={winner_row[0][:8]}\n"
                            f"{merge_eval.error_trace or '(cv_score is None)'}"
                        )
                    else:
                        merge_delta = abs(merge_eval.cv_score - winner_row[2])
                        if merge_delta > _MERGE_VERIFY_TOLERANCE:
                            merge_ok = False
                            print(
                                f"[run_promote_task] merge-verify 실패 — 승격 스킵: "
                                f"merged_cv={merge_eval.cv_score:.6f} winner_cv={winner_row[2]:.6f} "
                                f"delta={merge_delta:.6f} (tolerance={_MERGE_VERIFY_TOLERANCE})"
                            )
                        else:
                            merge_oof_preds = merge_eval.oof_preds

                if merge_ok:
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
                            oof_preds=merge_oof_preds,
                            materialized_code=materialized,
                        )
                    upload_best_pipeline(competition_id, materialized)
                    print(f"[run_promote_task] best pipeline materialized for {competition_id}")

    # auto-submit(매일 06:00)이 제출하는 건 이번 super-cycle의 확정 승격 winner가 아니라
    # 대회 전역 best attempt(bin/api.py:_best_attempt와 동일 기준)다 — 확정 승격 여부와
    # 무관하게 매 promote task 종료 시점마다 그 attempt의 제출 CSV를 미리 캐싱해두면
    # ops-vm daemon이 auto-submit 시점에 fit 없이 업로드만 하게 된다(fit은 이미 big
    # 큐에서 도는 이 promote task 쪽으로 옮겨짐). 이미 캐시돼 있으면 재fit하지 않는다.
    # best-effort — 실패해도 promote 자체는 성공 처리한다.
    best = None
    try:
        from bin.api import _best_attempt
        from bin.submit import generate_submission_csv
        from store.s3_code import download_submission_csv, upload_submission_csv
        best = _best_attempt(conn, competition_id)
        if best:
            best_attempt_id, best_cv = best
            if download_submission_csv(competition_id, best_attempt_id) is None:
                csv_path, _, _ = generate_submission_csv(args.competition, attempt_id=best_attempt_id)
                upload_submission_csv(competition_id, best_attempt_id, csv_path.read_bytes())
                print(f"[run_promote_task] submission csv cached for best={best_attempt_id[:8]} cv={best_cv}")
    except Exception as exc:
        # competition_id/attempt_id를 문구에 남겨 daemon 로그에서 대회 단위로
        # grep 가능하게 한다 — 이 블록이 조용히 죽어 auto-submit 실패가 다음날
        # 06:00까지 안 보이던 사고(#71)의 재발 방지.
        best_attempt_id = best[0] if best else None
        print(
            f"[run_promote_task] submission csv caching failed for {competition_id} "
            f"(best_attempt={best_attempt_id[:8] if best_attempt_id else 'unknown'}, non-fatal): {exc}"
        )

    for i, r in enumerate(rows):
        (attempt_id, gain_vs_best, cv_score, label, error_trace,
         hypothesis, action_type, reflection_ids, cv_fold_var, code_path,
         _fold_scores) = r

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
