"""Super-cycle promote step — Airflow task 4 of 4.

Picks winner from whichever attempts exist for this super-cycle at gate(#203) time
(not always exactly 3), updates was_promoted, reflects winner. cross-seed
confirmation + audit holdout 측정 후 confirmed=True일 때만 승격.

context lookup key is --run-id (Airflow dag_run_id), not --queue-id. queue_id is
shared by every cycle of the same super-cycle (max_active_runs=4 lets several run
concurrently) — keying by queue_id let a later cycle's retrieve overwrite an
earlier cycle's context row.

Usage (container):
    uv run python -m bin.run_promote_task --queue-id <id> --run-id <run_id>
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from cycle.promotion import ConfirmResult
    from store.db import PgConn

ROOT = Path(__file__).parent.parent


def _unmark_promoted(conn: PgConn, attempt_id: str) -> None:
    conn.execute(
        "UPDATE raw.attempts SET was_promoted = false WHERE attempt_id = %s",
        [attempt_id],
    )


def _pick_winner(conn: PgConn, rows: list[tuple], super_cycle_id: str) -> int | None:
    """gain 최대 attempt(없으면 cv가 있는 첫 attempt)를 winner로 표시하고 목록을 출력한다."""
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
    return winner_idx


def _load_train(
    competition_slug: str, competition_id: str,
) -> tuple[ModuleType | None, pl.DataFrame | None, pl.DataFrame | None]:
    """(대회 config 모듈, train90, holdout10). 로드에 실패하면 train90이 None이라 confirm과 승격을 건너뛴다."""
    import importlib

    from evaluator.harness import split_audit_holdout
    from store.train_data import load_train

    try:
        comp = importlib.import_module(f"config.competitions.{competition_slug}")
        if comp.COMPETITION_ID != competition_id:
            print(
                f"[run_promote_task] WARNING: comp.COMPETITION_ID={comp.COMPETITION_ID!r}"
                f" != DB competition_id={competition_id!r}",
                file=sys.stderr,
            )
        full_train = load_train(comp)
        train90, holdout10 = split_audit_holdout(full_train, comp.TARGET, comp.IS_CLASSIFICATION)
    except Exception as exc:
        print(f"[run_promote_task] train 로드 실패 — holdout/confirm 스킵: {exc}")
        return None, None, None
    return comp, train90, holdout10


def _confirm_winner(
    conn: PgConn, comp: ModuleType, competition_id: str, train90: pl.DataFrame, holdout10: pl.DataFrame,
    winner_row: tuple, winner_source: str, current_best: str | None, best_params: dict | None, tuned_params: dict | None,
) -> ConfirmResult:
    """cross-seed + holdout confirm. 거부되면 was_promoted를 되돌린다."""
    from config.competitions import comp_cpu_budget_secs, comp_n_splits
    from config.settings import PROMOTE_CONFIRM_SEEDS
    from cycle.promotion import PromotionCache, confirm_and_measure, record_confirm

    attempt_id = winner_row[0]
    confirm = confirm_and_measure(
        source=winner_source,
        best_source=current_best,
        train90=train90,
        holdout10=holdout10,
        target_col=comp.TARGET,
        metric=comp.METRIC,
        n_splits=comp_n_splits(comp),
        seed=42,
        is_classification=comp.IS_CLASSIFICATION,
        confirm_seeds=PROMOTE_CONFIRM_SEEDS,
        cache=PromotionCache(conn),
        competition_id=competition_id,
        candidate_cv=winner_row[2],
        candidate_fold_scores=winner_row[10],
        cpu_budget_sec=comp_cpu_budget_secs(comp),
        conn=conn,
        best_params=best_params,
        tuned_params=tuned_params,
    )
    record_confirm(conn, attempt_id, confirm)
    if not confirm.confirmed:
        reason = (
            "holdout 악화" if confirm.holdout_regressed
            else "holdout 측정 실패" if getattr(confirm, "holdout_measurement_failed", False)
            else "cross-seed 미확인"
        )
        print(f"[run_promote_task] {reason} — 승격 스킵 winner={attempt_id[:8]}")
        # was_promoted는 gate(#203) 이전에 gain 최고 attempt로 무조건 True가 심겨 있었다(_pick_winner) — confirm이 거부하면
        # 되돌린다(#395). 안 그러면 cycle/stagnation.py의 "최근 실제 승격" 판단이 버려진 attempt를 승격으로 오판한다.
        _unmark_promoted(conn, attempt_id)
    return confirm


def _promote_winner(
    conn: PgConn, comp: ModuleType, competition_id: str, train90: pl.DataFrame, winner_row: tuple, winner_source: str,
    current_best: str | None, best_params: dict | None, tuned_params: dict | None,
) -> None:
    """병합본을 merge-verify로 재평가한 뒤 raw.pipelines에 insert하고 MinIO에 올린다. 실패하면 was_promoted를 되돌린다."""
    from config.competitions import comp_cpu_budget_secs, comp_n_splits
    from cycle.materialize import materialize_best_pipeline, with_frozen_params
    from evaluator.metrics import float_noise_tolerance
    from runtime.isolate import eval_isolated
    from store.db import competition_fingerprint, insert_pipeline
    from store.s3_code import BestPipelineUploadError, upload_best_pipeline

    attempt_id, winner_gain, winner_cv, winner_params = winner_row[0], winner_row[1], winner_row[2], winner_row[11]
    fp_dict = competition_fingerprint(conn, competition_id)
    # materialize 먼저 → 해시는 실제 MinIO 업로드 내용(submit.py가 exec하는 문자열) 기준. raw.pipelines.code(winner source)와는
    # 다른 문자열.
    promoted_source = with_frozen_params(winner_source, winner_params)
    materialized = materialize_best_pipeline(current_best, promoted_source)
    pipeline_sha256 = hashlib.sha256(materialized.encode()).hexdigest()

    # merge-verify — 병합본을 실제로 1회 평가해 winner 자신의 cv_score와 어긋나지 않는지 확인(정적 AST 검증만으로는 병합 손상을 못 잡음).
    merge_eval = eval_isolated(
        source=materialized,
        train=train90,
        target_col=comp.TARGET,
        metric=comp.METRIC,
        prev_best=None,
        n_splits=comp_n_splits(comp),
        seed=42,
        is_classification=comp.IS_CLASSIFICATION,
        collect_oof=True,  # 이 1회 eval에 얹어 OOF 확보(추가 비용 없음)
        cpu_budget_sec=comp_cpu_budget_secs(comp),
        # winner의 attempt-time eval과 같은 값이어야 merge-verify 허용오차 비교가 의미 있다(#388) — confirm과 같은 값.
        best_params=best_params,
        tuned_params=tuned_params,
    )
    if merge_eval.error_trace or merge_eval.cv_score is None:
        # [:200] 절단이 Airflow 로그에서 실제 예외를 가려 원인을 못 잡은 전례가 있어 전체 출력.
        print(
            "[run_promote_task] merge-verify 실패(평가 에러) — 승격 스킵 "
            f"competition={competition_id} winner={attempt_id[:8]}\n"
            f"{merge_eval.error_trace or '(cv_score is None)'}"
        )
        _unmark_promoted(conn, attempt_id)
        return
    merge_delta = abs(merge_eval.cv_score - winner_cv)
    merge_tolerance = float_noise_tolerance(winner_cv)
    if merge_delta > merge_tolerance:
        print(
            f"[run_promote_task] merge-verify 실패 — 승격 스킵: "
            f"merged_cv={merge_eval.cv_score:.6f} winner_cv={winner_cv:.6f} "
            f"delta={merge_delta:.3e} (tolerance={merge_tolerance:.3e})"
        )
        _unmark_promoted(conn, attempt_id)
        return

    try:
        with conn.transaction():
            insert_pipeline(
                conn,
                pipeline_id=str(uuid.uuid4()),
                attempt_id=attempt_id,
                competition_id=competition_id,
                fingerprint_snapshot=fp_dict,
                code=promoted_source,
                cv_score=winner_cv,
                gain_vs_best=winner_gain,
                pipeline_sha256=pipeline_sha256,
                oof_preds=merge_eval.oof_preds,
                materialized_code=materialized,
            )
            # 트랜잭션 안에서 부른다 — 업로드 실패가 insert를 롤백해야 DB와 blob이 어긋나지 않는다.
            upload_best_pipeline(competition_id, materialized, strict=True)
    except BestPipelineUploadError as exc:
        print(f"[run_promote_task] best pipeline 업로드 실패 — 승격 롤백 winner={attempt_id[:8]}: {exc}")
        # pipeline이 없는데 was_promoted가 True로 남으면 cycle/stagnation.py가 기각된 jump를 승격으로 세어 정체 신호가 늦게 켜진다.
        _unmark_promoted(conn, attempt_id)
    else:
        print(f"[run_promote_task] best pipeline materialized for {competition_id}")


def _cache_submission_csv(conn: PgConn, competition_slug: str, competition_id: str) -> None:
    """대회 전역 best attempt의 제출 CSV를 미리 캐싱한다 — auto-submit(매일 06:00)이 fit 없이 업로드만 하게 하려는 것이다.

    확정 승격 여부와 무관하게 매 promote task 종료 시점에 돈다(auto-submit이 제출하는 건 이번 super-cycle의 winner가 아니라
    bin/api.py:_best_attempt 기준의 전역 best다). 이미 캐시돼 있으면 재fit하지 않는다. fit은 별도 프로세스에서 wall 상한을 걸어
    돌리고, 상한을 넘긴 attempt는 표식을 남겨 다음 promote가 같은 실패를 반복하지 않게 한다(#355). best-effort — 실패해도
    promote 자체는 성공 처리한다.
    """
    best = None
    try:
        import importlib
        import subprocess

        from bin.api import _best_attempt
        from bin.submit import generate_submission_csv_isolated
        from store.s3_code import (
            download_submission_csv,
            mark_submission_csv_timed_out,
            submission_csv_timed_out,
            upload_submission_csv,
        )
        best = _best_attempt(conn, competition_id)
        if best and download_submission_csv(competition_id, best[0]) is None:
            best_attempt_id, best_cv = best
            if submission_csv_timed_out(competition_id, best_attempt_id):
                print(f"[run_promote_task] submission csv skipped for best={best_attempt_id[:8]}: previous fit timed out")
            else:
                comp = importlib.import_module(f"config.competitions.{competition_slug}")
                try:
                    csv_path = generate_submission_csv_isolated(
                        competition_slug, best_attempt_id, full_data=getattr(comp, "SUBMIT_FULL_DATA", False),
                    )
                except subprocess.TimeoutExpired:
                    mark_submission_csv_timed_out(competition_id, best_attempt_id)
                    raise
                upload_submission_csv(competition_id, best_attempt_id, csv_path.read_bytes())
                print(f"[run_promote_task] submission csv cached for best={best_attempt_id[:8]} cv={best_cv}")
    except Exception as exc:
        # competition_id/attempt_id를 문구에 남겨 daemon 로그에서 대회 단위로 grep 가능하게 한다 — 이 블록이 조용히 죽어
        # auto-submit 실패가 다음날 06:00까지 안 보이던 사고의 재발 방지.
        best_attempt_id = best[0] if best else None
        print(
            f"[run_promote_task] submission csv caching failed for {competition_id} "
            f"(best_attempt={best_attempt_id[:8] if best_attempt_id else 'unknown'}, non-fatal): {exc}"
        )


def _reflect_attempts(
    conn: PgConn, competition_id: str, rows: list[tuple], winner_idx: int, bandit_label: str | None,
) -> None:
    """winner는 jump/regression/error만 reflect하고(neutral은 교훈 불명확), loser는 neutral 포함 전부 reflect한다
    ("이 시도는 효과 없었다"도 학습 신호)."""
    from agents.reflector import AttemptContext, reflect
    from memory.retriever import EmbeddingUnavailableError
    from store.s3_code import download as _code_download
    from store.s3_code import strip_code_header

    for i, r in enumerate(rows):
        (attempt_id, gain_vs_best, cv_score, label, error_trace,
         hypothesis, action_type, reflection_ids, cv_fold_var, code_path,
         _fold_scores, _params) = r

        is_winner = (i == winner_idx)
        if is_winner and label not in ("jump", "regression") and error_trace is None:
            continue

        source = ""
        if code_path:
            source = strip_code_header(_code_download(code_path) or "")

        # winner이고 confirm이 실제로 돌았으면 confirm-보정된 label로 lesson을 남긴다 — "CV에서는 좋아 보였지만
        # 실제 검증은 통과 못 했다"는 신호가 strategist 학습에 전달돼야 한다(#164).
        reflect_label = bandit_label if (is_winner and bandit_label is not None) else label

        ctx = AttemptContext(
            hypothesis=hypothesis or "",
            action_type=action_type or "",
            code=source,
            cv_score=cv_score or 0.0,
            cv_fold_var=cv_fold_var or 0.0,
            gain_vs_best=gain_vs_best,
            label=reflect_label or "regression",
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


def main() -> None:
    # 이 프로세스는 Airflow DockerOperator가 별도 실행하는 진입점이라 부모의
    # 로깅 설정을 상속받지 않는다 — basicConfig 없이는 cycle/promotion.py의
    # 게이트 실패 로그(_LOG.warning 등)가 lastResort 핸들러에 의존하게 되는데,
    # 그마저도 없던 기간엔 INFO 로그가 전부 조용히 사라졌다.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--competition", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))

    from cycle.action_optimizer import update_bandit
    from cycle.promotion import effective_label
    from cycle.run import _baseline_source_guard, _latest_tuned_params, _prev_best_fold_scores, _prev_best_params
    from evaluator.harness import is_significant_gain
    from store.db import connect
    from store.s3_code import download as _code_download
    from store.s3_code import download_best_pipeline, strip_code_header

    conn = connect(apply_schema=False)

    ctx_row = conn.execute(
        "SELECT super_cycle_id, competition_id FROM raw.super_cycle_context WHERE run_id = %s",
        [args.run_id],
    ).fetchone()
    if not ctx_row:
        print(f"[run_promote_task] no context for run_id={args.run_id}", file=sys.stderr)
        sys.exit(1)

    super_cycle_id, competition_id = ctx_row

    # 여기서 context를 DELETE하지 않는다(#206) — attempt_gate(#203) 도입 후 promote가 일찍 뜰 수 있는데, 아직 시작도 안 한
    # straggler가 그 사이 context를 못 찾으면 bin/run_attempt_task.py가 하드 실패한다. 위생은 retrieve의 7일 TTL 청소가 맡는다.

    rows = conn.execute(
        """
        SELECT attempt_id, gain_vs_best, cv_score, label, error_trace,
               hypothesis, action_type, reflection_ids, cv_fold_var, code_path,
               fold_scores, params
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

    winner_idx = _pick_winner(conn, rows, super_cycle_id)
    if winner_idx is None:
        print("  -> all errored, no winner")
        conn.close()
        return

    print(f"  -> promoted {rows[winner_idx][0][:8]} (gain={rows[winner_idx][1]})")

    winner_row = rows[winner_idx]
    winner_gain = winner_row[1]
    winner_cv_fold_var = winner_row[8] or 0.0
    winner_error = winner_row[4]
    winner_code_path = winner_row[9]
    winner_fold_scores = winner_row[10]

    # paired per-fold 검정용 metric_sign + baseline fold_scores. comp 모듈은 아직 import 안 했으므로 DB에서 바로 조회.
    sign_row = conn.execute(
        "select metric_sign from raw.competitions where competition_id = %s",
        [competition_id],
    ).fetchone()
    metric_sign = sign_row[0] if sign_row and sign_row[0] is not None else 1
    baseline_fold_scores = _prev_best_fold_scores(conn, competition_id)

    stage1_significant = is_significant_gain(
        winner_gain, winner_cv_fold_var,
        candidate_fold_scores=winner_fold_scores,
        baseline_fold_scores=baseline_fold_scores,
        metric_sign=metric_sign,
    )
    print(
        f"[run_promote_task] gate stage1: significant={stage1_significant} "
        f"gain={winner_gain} cv_fold_var={winner_cv_fold_var} "
        f"candidate_folds={len(winner_fold_scores) if winner_fold_scores else 0} "
        f"baseline_folds={len(baseline_fold_scores) if baseline_fold_scores else 0} "
        f"has_error={bool(winner_error)} has_code={bool(winner_code_path)}"
    )

    # confirm이 실제로 돈 경우에만 effective_label로 재계산된다 — stage1 자체가 미유의/에러/코드없음이면 confirm을 안 타므로
    # None으로 남고, reflect 루프에서 winner_row[3](원본 label) 그대로 쓴다.
    bandit_label: str | None = None

    if stage1_significant and not winner_error and winner_code_path:
        winner_source = strip_code_header(_code_download(winner_code_path) or "")
        if winner_source:
            comp, train90, holdout10 = _load_train(args.competition, competition_id)

            # confirm은 이 blob을 baseline으로 재평가한다 — 레지스트리 유효행과 어긋난 blob(격리 후 미재구성 등, #278)이면
            # 팬텀 baseline에 대고 confirm을 돌려 컴퓨트만 태우므로, 여기서 멈춘다.
            _baseline_source_guard(conn, competition_id)
            current_best = download_best_pipeline(competition_id)
            if train90 is None:
                # train 로드 실패는 confirm이 스킵돼 승격도 스킵한다(#389) — "검증 불가"를 "검증 통과"로 취급하지 않는다.
                # 재시도 가능한 작업 실패라 다음 promote task가 train 로드를 다시 시도한다.
                bandit_label = winner_row[3]
            else:
                # attempt 평가(cycle/run.py)와 같은 조회 함수를 같은 시점(super-cycle 종료 직후)에 다시 불러 같은 값을
                # 얻는다(#388) — confirm과 merge-verify가 이 값을 공유해야 winner의 attempt-time cv와 어긋나지 않는다.
                best_params = _prev_best_params(conn, competition_id)
                tuned_params = _latest_tuned_params(conn, competition_id)
                confirm = _confirm_winner(
                    conn, comp, competition_id, train90, holdout10, winner_row, winner_source, current_best,
                    best_params, tuned_params,
                )
                # bandit 보상을 confirm 결과와 연동 — cycle/run.py의 attempt-생성 시점 update_bandit(defer_promotion=True라
                # 원본 label로 이미 한 번 쐈다)은 confirm을 모른다. confirm이 jump를 거부하면 regression 방향으로 보정
                # 신호를 추가로 준다 — 안 그러면 같은 action_type이 다음 cycle에 계속 높은 확률로 재선택된다(#164).
                bandit_label = effective_label(winner_row[3], confirm)
                update_bandit(
                    conn, competition_id=competition_id, action_type=winner_row[6],
                    label=bandit_label, gain_vs_best=winner_gain, error_trace=winner_error,
                )
                if confirm.confirmed:
                    _promote_winner(
                        conn, comp, competition_id, train90, winner_row, winner_source, current_best,
                        best_params, tuned_params,
                    )

    _cache_submission_csv(conn, args.competition, competition_id)
    _reflect_attempts(conn, competition_id, rows, winner_idx, bandit_label)

    conn.close()


if __name__ == "__main__":
    main()
