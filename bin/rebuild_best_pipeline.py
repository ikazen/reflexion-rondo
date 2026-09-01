"""
best_pipeline.py 재구성 — raw.pipelines 히스토리를 시간순으로 재생(replay)한다.

materialize_best_pipeline()의 병합 결과가 손상돼(참조는 있는데 정의가
없는 이름) 운영 중 best_pipeline.py가 깨지는 사고가 있었다. 정확한 손상 트리거는
미확정이지만, raw.pipelines에 저장된 개별 attempt 코드는 전부 자기완결적이므로
base=None부터 승격 이력을 순서대로 다시 병합하면 항상 self-consistent한 결과가
나온다 — 이 스크립트는 그 재생을 수행하는 범용 복구 도구다(이번 사고 전용이
아니라 향후 유사 손상에도 재사용 가능).

Dry-run (결과만 stdout 출력, 업로드 안 함):
  uv run python -m bin.rebuild_best_pipeline --competition playground-series-s4e10 --dry-run

실제 반영 (MinIO best_pipeline.py 덮어쓰기):
  uv run python -m bin.rebuild_best_pipeline --competition playground-series-s4e10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def rebuild(competition_id: str, dry_run: bool) -> str:
    sys.path.insert(0, str(ROOT))
    import hashlib

    from cycle.materialize import replay_best_pipeline
    from store.db import connect
    from store.s3_code import upload_best_pipeline

    conn = connect(apply_schema=False)
    try:
        best, _, count = replay_best_pipeline(conn, competition_id)

        if not best:
            print(f"no raw.pipelines rows for competition_id={competition_id!r} — nothing to rebuild")
            sys.exit(1)

        print(f"replayed {count} promoted pipeline(s) for {competition_id}")

        if dry_run:
            print("\n--dry-run: not uploading. Result below:\n")
            print(best)
            return best

        upload_best_pipeline(competition_id, best)
        # 재생 결과를 최신 유효행의 신뢰 스냅샷으로 기록한다(#254/ADR-039와 동일 방식):
        # materialize 로직이 바뀌었으면 재생본이 승격 당시 병합본과 텍스트로 다를 수
        # 있는데, 그게 이제 MinIO에 올라간 정본이다. _baseline_source_guard와 submit.py가
        # coalesce(materialized_sha256, pipeline_sha256)로 이 blob을 검증한다.
        new_sha = hashlib.sha256(best.encode()).hexdigest()
        latest_valid = conn.execute(
            "select p.pipeline_id from raw.pipelines p join raw.attempts a using (attempt_id)"
            " where p.competition_id = %s and p.invalid_reason is null"
            " order by a.run_ts desc limit 1",
            [competition_id],
        ).fetchone()
        if latest_valid:
            conn.execute(
                "update raw.pipelines set materialized_code = %s, materialized_sha256 = %s"
                " where pipeline_id = %s",
                [best, new_sha, latest_valid[0]],
            )
            print(f"recorded trusted snapshot on {latest_valid[0][:8]} (sha {new_sha[:12]})")
    finally:
        conn.close()

    print(f"\nuploaded rebuilt best_pipeline.py for {competition_id}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", required=True, help="competition_id (e.g. playground-series-s4e10)")
    parser.add_argument("--dry-run", action="store_true", help="print result only, do not upload")
    args = parser.parse_args()
    rebuild(args.competition, args.dry_run)


if __name__ == "__main__":
    main()
