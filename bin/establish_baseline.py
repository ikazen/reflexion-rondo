"""기존 대회 baseline 소급 확립 스크립트. 배경·운영 절차는 docs/decisions.md
ADR-025, docs/runbook.md §4-2.

확정 파이프라인(raw.pipelines)이 0건인 대회의 cv 상위 top-k개 attempt를 순회하며
BasePipeline 대비 cross-seed confirm + holdout 게이트를 통과하는 첫 번째를
raw.pipelines에 확정 baseline으로 승격한다. phantom(실제로는 재현 안 되는 비정상
고점)은 게이트를 통과 못 해 자동 탈락하고 다음 순위로 넘어간다 —
cycle/run.py:establish_bootstrap_baseline(단일 최고 attempt만 시도)과 달리 이미
대량 축적된 attempt 풀에서 phantom이 최고점을 차지하고 있을 가능성에 대응하기
위해 top-k 폴백을 갖는다.

Dry-run (승격 후보만 출력, DB/MinIO 반영 없음):
  uv run python -m bin.establish_baseline --dry-run
  uv run python -m bin.establish_baseline --competition playground-series-s5e7 --dry-run

실제 반영 (전체 대상 대회):
  uv run python -m bin.establish_baseline

특정 대회만:
  uv run python -m bin.establish_baseline --competition playground-series-s5e7
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl

from config.settings import PROMOTE_CONFIRM_SEEDS
from cycle.materialize import materialize_best_pipeline
from cycle.promotion import PromotionCache, confirm_and_measure
from cycle.run import _CODE_HEADER_SEP
from evaluator.harness import split_audit_holdout
from runtime.isolate import eval_isolated
from store.db import connect, insert_pipeline
from store.s3_code import download as _code_download
from store.s3_code import upload_best_pipeline
from store.train_data import load_train

_DEFAULT_TOP_K = 5


def _competition_id_to_slug() -> dict[str, str]:
    """config/competitions/*.py 스캔 → {competition_id: module_slug} 맵."""
    result: dict[str, str] = {}
    for path in (ROOT / "config" / "competitions").glob("*.py"):
        if path.stem.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"config.competitions.{path.stem}")
        except Exception:
            continue
        cid = getattr(mod, "COMPETITION_ID", None)
        if cid:
            result[cid] = path.stem
    return result


def competitions_without_baseline(conn) -> list[str]:
    rows = conn.execute(
        """
        SELECT c.competition_id
        FROM raw.competitions c
        WHERE NOT EXISTS (
            SELECT 1 FROM raw.pipelines p
            WHERE p.competition_id = c.competition_id AND p.invalid_reason IS NULL
              AND COALESCE(p.materialized_origin, '') NOT LIKE 'unverifiable:%'
        )
        ORDER BY c.competition_id
        """
    ).fetchall()
    return [r[0] for r in rows]


def _top_k_attempts(conn, competition_id: str, top_k: int) -> list[tuple[str, float, str, list]]:
    return conn.execute(
        """
        SELECT a.attempt_id, a.cv_score, a.code_path, a.fold_scores
        FROM raw.attempts a
        JOIN raw.competitions c USING (competition_id)
        WHERE a.competition_id = %s
          AND a.cv_score IS NOT NULL
          AND a.error_trace IS NULL
          AND a.code_path IS NOT NULL
        ORDER BY c.metric_sign * a.cv_score DESC
        LIMIT %s
        """,
        [competition_id, top_k],
    ).fetchall()


def _promote(conn, comp: object, attempt_id: str, cv_score: float, source: str, confirm, train90: pl.DataFrame) -> None:
    competition_id = comp.COMPETITION_ID
    if confirm.holdout_score is not None:
        conn.execute(
            "UPDATE raw.attempts SET holdout_score = %s WHERE attempt_id = %s",
            [confirm.holdout_score, attempt_id],
        )
    if confirm.seed_gains:
        conn.execute(
            "UPDATE raw.attempts SET confirm_seed_gains = %s WHERE attempt_id = %s",
            [json.dumps(confirm.seed_gains), attempt_id],
        )

    fp_row = conn.execute(
        "SELECT fingerprint FROM raw.competitions WHERE competition_id = %s",
        [competition_id],
    ).fetchone()
    fp_val = fp_row[0] if fp_row and fp_row[0] else {}
    fp_dict = fp_val if isinstance(fp_val, dict) else json.loads(fp_val)

    materialized = materialize_best_pipeline(None, source)
    pipeline_sha256 = hashlib.sha256(materialized.encode()).hexdigest()

    # OOF 확보 — bin/run_promote_task.py의 merge-verify와 동일 패턴(추가 eval 아님).
    # 소비처였던 bin/blend.py는 #231로 폐기됐지만 oof_preds 자체는 계속 채운다(향후
    # 분석 재료). best-effort — 실패해도 baseline 확립 자체는 막지 않는다.
    merge_oof_preds = None
    try:
        merge_eval = eval_isolated(
            source=materialized,
            train=train90,
            target_col=comp.TARGET,
            metric=comp.METRIC,
            prev_best=None,
            n_splits=getattr(comp, "N_SPLITS", 5),
            seed=42,
            is_classification=comp.IS_CLASSIFICATION,
            collect_oof=True,
            cpu_budget_sec=getattr(comp, "CPU_BUDGET_SECS", None),
        )
        if not merge_eval.error_trace and merge_eval.cv_score is not None:
            merge_oof_preds = merge_eval.oof_preds
    except Exception as exc:
        print(f"  merge-verify OOF 수집 실패(무시하고 계속): {exc}")

    with conn.transaction():
        insert_pipeline(
            conn,
            pipeline_id=str(uuid.uuid4()),
            attempt_id=attempt_id,
            competition_id=competition_id,
            fingerprint_snapshot=fp_dict,
            code=source,
            cv_score=cv_score,
            gain_vs_best=None,
            pipeline_sha256=pipeline_sha256,
            oof_preds=merge_oof_preds,
            materialized_code=materialized,
        )
    upload_best_pipeline(competition_id, materialized)


def _current_confirmed_pipeline(conn, competition_id: str) -> tuple[str, str, float] | None:
    """확정 pipeline 중 최고점(pipeline_id, materialized_code, cv_score). 없으면 None."""
    row = conn.execute(
        """
        SELECT p.pipeline_id, p.materialized_code, p.cv_score
        FROM raw.pipelines p
        JOIN raw.competitions c USING (competition_id)
        WHERE p.competition_id = %s
          AND p.cv_score IS NOT NULL
          AND p.invalid_reason IS NULL
          AND p.materialized_code IS NOT NULL
        ORDER BY c.metric_sign * p.cv_score DESC
        LIMIT 1
        """,
        [competition_id],
    ).fetchone()
    return (row[0], row[1], row[2]) if row else None


def remeasure_competition(conn, comp: object, dry_run: bool) -> bool:
    """MAX_TRAIN_ROWS 등으로 학습 데이터가 바뀌었을 때, 현재 확정 pipeline의
    cv_score를 새 데이터로 재측정해 raw.pipelines에 반영한다.

    데이터가 바뀌면 기존 cv_score(옛 데이터 기준)와 신규 attempt의 cv_score(새
    데이터 기준)가 비교 불가능해져 gain_vs_best/승격 게이트가 깨진다(#135) —
    재학습 없이 확정 pipeline의 materialized_code를 그대로 새 데이터에 재평가만
    한다. 반드시 새로 측정한 값만 쓴다 — establish_for_competition/_promote처럼
    과거 attempt 행의 cv_score를 그대로 심으면 옛 데이터 기준 값이 섞여 들어간다.
    """
    found = _current_confirmed_pipeline(conn, comp.COMPETITION_ID)
    if not found:
        print(f"  {comp.COMPETITION_ID}: 확정 pipeline 없음 — 재측정 불가")
        return False
    pipeline_id, materialized_code, old_cv = found

    train = load_train(comp)
    train90, _holdout10 = split_audit_holdout(train, comp.TARGET, comp.IS_CLASSIFICATION)

    result = eval_isolated(
        source=materialized_code,
        train=train90,
        target_col=comp.TARGET,
        metric=comp.METRIC,
        prev_best=None,
        n_splits=getattr(comp, "N_SPLITS", 5),
        seed=42,
        is_classification=comp.IS_CLASSIFICATION,
    )
    if result.error_trace or result.cv_score is None:
        print(f"  {comp.COMPETITION_ID}: 재측정 실패 — {result.error_trace}")
        return False

    new_cv = result.cv_score
    action = "dry-run, 미반영" if dry_run else "반영"
    print(f"  {comp.COMPETITION_ID} pipeline={pipeline_id[:8]}: cv {old_cv} -> {new_cv} ({action})")
    if not dry_run:
        conn.execute(
            "UPDATE raw.pipelines SET cv_score = %s WHERE pipeline_id = %s",
            [new_cv, pipeline_id],
        )
    return True


def establish_for_competition(conn, comp: object, top_k: int, dry_run: bool) -> str | None:
    """이 대회에 baseline 확립을 시도한다. 성공 시 승격된 attempt_id, 실패면 None."""
    candidates = _top_k_attempts(conn, comp.COMPETITION_ID, top_k)
    if not candidates:
        print(f"  {comp.COMPETITION_ID}: 후보 attempt 없음 — 스킵")
        return None

    train = load_train(comp)
    train90, holdout10 = split_audit_holdout(train, comp.TARGET, comp.IS_CLASSIFICATION)
    sep = _CODE_HEADER_SEP + "\n"

    cache = PromotionCache(conn)
    for rank, (attempt_id, cv_score, code_path, fold_scores) in enumerate(candidates, 1):
        content = _code_download(code_path) or ""
        source = content.split(sep, 1)[1].strip() if sep in content else content.strip()
        if not source:
            print(f"  {comp.COMPETITION_ID} rank={rank} attempt={attempt_id[:8]}: 코드 없음 — 스킵")
            continue

        confirm = confirm_and_measure(
            source=source,
            best_source=None,
            train90=train90,
            holdout10=holdout10,
            target_col=comp.TARGET,
            metric=comp.METRIC,
            n_splits=getattr(comp, "N_SPLITS", 5),
            seed=42,
            is_classification=comp.IS_CLASSIFICATION,
            confirm_seeds=PROMOTE_CONFIRM_SEEDS,
            cache=cache,
            competition_id=comp.COMPETITION_ID,
            candidate_cv=cv_score,
            candidate_fold_scores=fold_scores,
            cpu_budget_sec=getattr(comp, "CPU_BUDGET_SECS", None),
        )
        reason = (
            "confirmed" if confirm.confirmed
            else "holdout 악화" if confirm.holdout_regressed
            else "cross-seed 미재현"
        )
        print(f"  {comp.COMPETITION_ID} rank={rank} attempt={attempt_id[:8]} cv={cv_score}: {reason}")

        if not confirm.confirmed:
            continue

        if dry_run:
            print(f"  [dry-run] would establish baseline: {comp.COMPETITION_ID} attempt={attempt_id[:8]}")
        else:
            _promote(conn, comp, attempt_id, cv_score, source, confirm, train90)
            print(f"  established baseline: {comp.COMPETITION_ID} attempt={attempt_id[:8]} cv={cv_score}")
        return attempt_id

    print(f"  {comp.COMPETITION_ID}: 후보 {len(candidates)}개 전부 미확립")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default=None, help="특정 competition_id만 (기본: baseline 없는 전체)")
    parser.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K, help="시도할 상위 attempt 수")
    parser.add_argument("--dry-run", action="store_true", help="승격/반영 없이 결과만 출력")
    parser.add_argument(
        "--remeasure", action="store_true",
        help="baseline 신규 확립이 아니라, 확정 pipeline의 cv_score를 현재 "
             "load_train(예: MAX_TRAIN_ROWS 변경 후)으로 재측정 — --competition 필수",
    )
    args = parser.parse_args()

    slug_map = _competition_id_to_slug()
    conn = connect(apply_schema=False)

    if args.remeasure:
        if not args.competition:
            parser.error("--remeasure requires --competition")
        slug = slug_map.get(args.competition)
        if not slug:
            parser.error(f"unknown competition: {args.competition}")
        comp = importlib.import_module(f"config.competitions.{slug}")
        try:
            remeasure_competition(conn, comp, args.dry_run)
        finally:
            conn.close()
        return

    try:
        targets = [args.competition] if args.competition else competitions_without_baseline(conn)
        print(f"대상 대회 {len(targets)}개: {targets}\n")

        established: list[str] = []
        failed: list[str] = []
        for cid in targets:
            slug = slug_map.get(cid)
            if not slug:
                print(f"  {cid}: config 모듈 없음 — 스킵")
                failed.append(cid)
                continue
            comp = importlib.import_module(f"config.competitions.{slug}")
            attempt_id = establish_for_competition(conn, comp, args.top_k, args.dry_run)
            (established if attempt_id else failed).append(cid)
    finally:
        conn.close()

    verb = "would establish" if args.dry_run else "established"
    print(f"\n{verb}={len(established)} failed={len(failed)}")
    if failed:
        print("미확립 대회 (수동 판단 필요 — 재큐잉 후 재시도 등):", failed)


if __name__ == "__main__":
    main()
