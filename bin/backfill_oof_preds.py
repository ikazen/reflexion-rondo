"""oof_preds 결측 확정 pipeline 백필 (#145).

oof_preds 컬럼(v1.4.13)이 생기기 전에 승격된 pipeline은 영영 blend(#75) 후보가
될 수 없다 — fetch_oof_candidates(bin/blend.py)가 oof_preds IS NOT NULL만 본다.
materialized_code(실제 MinIO에 올라간, submit.py가 exec하는 문자열)를 train90으로
재평가(collect_oof=True)해 oof_preds를 채운다. cv_score가 원본과 크게 어긋나면
(materialize 손상 등, #137류 버그) 신뢰할 수 없으므로 저장하지 않고 스킵한다.

materialized_code가 없는 pipeline(v1.4.10 이전, #89 백필 대상 밖)은 원본 code로
재구성할 수 없으므로 스킵 — 대부분 대회당 1개(최고점)만 materialized_code를 갖고
있어(#89 백필 범위), 그게 이미 fetch_oof_candidates의 top-N 후보이므로 실질적
손실은 적다.

Dry-run (재평가만, DB 반영 없음):
  uv run python -m bin.backfill_oof_preds --dry-run

특정 대회만:
  uv run python -m bin.backfill_oof_preds --competition playground-series-s4e10
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.harness import split_audit_holdout
from runtime.isolate import eval_isolated
from store.db import connect
from store.train_data import load_train

_CV_MATCH_TOLERANCE = 1e-4


def _competition_id_to_slug() -> dict[str, str]:
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


def _missing_oof_pipelines(conn, competition_id: str | None) -> list[tuple[str, str, float]]:
    where = "p.invalid_reason IS NULL AND p.oof_preds IS NULL AND p.materialized_code IS NOT NULL"
    params: list = []
    if competition_id:
        where += " AND p.competition_id = %s"
        params.append(competition_id)
    return conn.execute(
        f"""
        SELECT p.pipeline_id, p.competition_id, p.materialized_code, p.cv_score
        FROM raw.pipelines p
        WHERE {where}
        ORDER BY p.competition_id, p.cv_score DESC
        """,
        params,
    ).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default=None, help="특정 competition_id만 (기본: 전체)")
    parser.add_argument("--dry-run", action="store_true", help="재평가 결과만 출력, DB 반영 없음")
    args = parser.parse_args()

    slug_map = _competition_id_to_slug()
    conn = connect(apply_schema=False)
    try:
        rows = _missing_oof_pipelines(conn, args.competition)
        print(f"대상 pipeline {len(rows)}개\n")

        filled = 0
        skipped = 0
        train_cache: dict[str, tuple] = {}

        for pipeline_id, competition_id, materialized_code, cv_score in rows:
            slug = slug_map.get(competition_id)
            if not slug:
                print(f"  {pipeline_id[:8]} ({competition_id}): config 모듈 없음 — 스킵")
                skipped += 1
                continue
            comp = importlib.import_module(f"config.competitions.{slug}")

            if competition_id not in train_cache:
                try:
                    full_train = load_train(comp)
                    train90, _holdout10 = split_audit_holdout(
                        full_train, comp.TARGET, comp.IS_CLASSIFICATION
                    )
                    train_cache[competition_id] = (train90,)
                except Exception as exc:
                    print(f"  {competition_id}: train 로드 실패({exc}) — 대회 전체 스킵")
                    train_cache[competition_id] = None
            cached = train_cache[competition_id]
            if cached is None:
                skipped += 1
                continue
            (train90,) = cached

            eval_result = eval_isolated(
                source=materialized_code,
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

            if eval_result.error_trace or eval_result.cv_score is None:
                print(
                    f"  {pipeline_id[:8]} ({competition_id}): 재평가 에러 — 스킵\n"
                    f"    {eval_result.error_trace or '(cv_score is None)'}"
                )
                skipped += 1
                continue

            delta = abs(eval_result.cv_score - cv_score)
            if delta > _CV_MATCH_TOLERANCE:
                print(
                    f"  {pipeline_id[:8]} ({competition_id}): cv 불일치 "
                    f"(원본={cv_score:.6f} 재평가={eval_result.cv_score:.6f} delta={delta:.6f}) — 스킵"
                )
                skipped += 1
                continue

            if eval_result.oof_preds is None:
                print(f"  {pipeline_id[:8]} ({competition_id}): metric_class가 OOF 미수집 대상 — 스킵")
                skipped += 1
                continue

            if args.dry_run:
                print(f"  [dry-run] {pipeline_id[:8]} ({competition_id}): oof_preds {len(eval_result.oof_preds)}개 확보")
            else:
                conn.execute(
                    "UPDATE raw.pipelines SET oof_preds = %s::jsonb WHERE pipeline_id = %s",
                    [json.dumps(eval_result.oof_preds), pipeline_id],
                )
                print(f"  filled: {pipeline_id[:8]} ({competition_id})")
            filled += 1
    finally:
        conn.close()

    verb = "would fill" if args.dry_run else "filled"
    print(f"\n{verb}={filled} skipped={skipped}")


if __name__ == "__main__":
    main()
