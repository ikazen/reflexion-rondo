"""
raw.pipelines.materialized_code 백필 (#89) — 기존 승격 행에 병합본 스냅샷 소급.

과거 승격 행은 materialized_code가 없어 submit이 replay 폴백을 타는데, replay는
materialize 로직 변경(#83 등) 이후 당시 병합본을 재현하지 못한다(strict_sha로
중단됨). 유일하게 남아있는 당시 병합본은 각 대회의 현 MinIO best_pipeline.py —
이것이 마지막 승격 행의 pipeline_sha256과 일치하면 그 행에 저장한다.
중간 이력 행은 복원 불가하므로 건드리지 않는다(제출 base는 최신 행만 쓰는 게
대부분이라 마지막 행 백필로 충분).

Dry-run (대상만 출력, DB 쓰기 안 함):
  uv run python -m bin.backfill_materialized_code --dry-run

실제 반영:
  uv run python -m bin.backfill_materialized_code
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def backfill(dry_run: bool) -> None:
    sys.path.insert(0, str(ROOT))
    from store.db import connect
    from store.s3_code import download_best_pipeline

    conn = connect(apply_schema=False)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT ON (p.competition_id)
                   p.competition_id, p.pipeline_id, p.pipeline_sha256, p.materialized_code
            FROM raw.pipelines p
            JOIN raw.attempts a USING (attempt_id)
            ORDER BY p.competition_id, a.run_ts DESC
            """
        ).fetchall()

        filled, skipped, mismatched = 0, 0, []
        for competition_id, pipeline_id, trusted_sha, existing in rows:
            if existing:
                skipped += 1
                continue
            blob = download_best_pipeline(competition_id)
            if not blob:
                mismatched.append((competition_id, "MinIO best_pipeline.py 없음"))
                continue
            actual = hashlib.sha256(blob.encode()).hexdigest()
            if trusted_sha and actual != trusted_sha:
                mismatched.append((competition_id, f"sha 불일치 (expected {trusted_sha[:12]}…, got {actual[:12]}…)"))
                continue
            if dry_run:
                print(f"[dry-run] would backfill {competition_id} (pipeline_id={pipeline_id[:8]})")
            else:
                conn.execute(
                    "UPDATE raw.pipelines SET materialized_code = %s WHERE pipeline_id = %s",
                    [blob, pipeline_id],
                )
                print(f"backfilled {competition_id} (pipeline_id={pipeline_id[:8]})")
            filled += 1
    finally:
        conn.close()

    print(f"\n{'would backfill' if dry_run else 'backfilled'}={filled} already-filled={skipped} unresolved={len(mismatched)}")
    if mismatched:
        print("unresolved (수동 판단 필요 — bin/rebuild_best_pipeline.py 재생 후 재승격 등):")
        for competition_id, reason in mismatched:
            print(f"  {competition_id}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print targets only, do not write")
    args = parser.parse_args()
    backfill(args.dry_run)


if __name__ == "__main__":
    main()
