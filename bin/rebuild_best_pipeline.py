"""
best_pipeline.py 재구성 — raw.pipelines 히스토리를 시간순으로 재생(replay)한다.

BON-233: materialize_best_pipeline()의 병합 결과가 손상돼(참조는 있는데 정의가
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
    from cycle.materialize import materialize_best_pipeline
    from store.db import connect
    from store.s3_code import upload_best_pipeline

    conn = connect(apply_schema=False)
    rows = conn.execute(
        """
        SELECT p.pipeline_id, a.run_ts, a.action_type, p.code
        FROM raw.pipelines p
        JOIN raw.attempts a USING (attempt_id)
        WHERE p.competition_id = %s
        ORDER BY a.run_ts ASC
        """,
        [competition_id],
    ).fetchall()
    conn.close()

    if not rows:
        print(f"no raw.pipelines rows for competition_id={competition_id!r} — nothing to rebuild")
        sys.exit(1)

    print(f"replaying {len(rows)} promoted pipeline(s) for {competition_id} ...")
    best: str | None = None
    for pipeline_id, run_ts, action_type, code in rows:
        try:
            best = materialize_best_pipeline(best, code)
        except Exception as exc:
            print(
                f"REPLAY FAILED at pipeline_id={pipeline_id} run_ts={run_ts} "
                f"action_type={action_type}: {exc}",
                file=sys.stderr,
            )
            print("stopping — manual investigation needed for this pipeline_id.", file=sys.stderr)
            sys.exit(1)
        print(f"  ok  {run_ts} {action_type:16s} {pipeline_id[:8]}")

    assert best is not None  # rows is non-empty, loop always assigns

    if dry_run:
        print("\n--dry-run: not uploading. Result below:\n")
        print(best)
    else:
        upload_best_pipeline(competition_id, best)
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
