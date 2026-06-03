"""저효율 교훈 자동 archive.

기준: times_applied >= MIN_APPLIED AND avg_gain <= MAX_GAIN
archive된 교훈은 retriever.search()에서 자동 제외 (where archived=false).

Usage:
    uv run python -m bin.archive_lessons              # 기본값으로 실행
    uv run python -m bin.archive_lessons --dry-run    # 변경 없이 후보만 출력
    uv run python -m bin.archive_lessons --min-applied 5 --max-gain -0.001
"""
from __future__ import annotations

import argparse

from store.db import connect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-applied", type=int, default=3,
                        help="최소 적용 횟수 (기본 3)")
    parser.add_argument("--max-gain",    type=float, default=0.0,
                        help="archive 임계 avg_gain (기본 0.0, 이하 대상)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 변경 없이 후보만 출력")
    args = parser.parse_args()

    conn = connect()

    candidates = conn.execute(
        """
        select r.reflection_id, i.times_applied, round(i.avg_gain, 5) as avg_gain,
               r.generality, r.competition_id
        from reflection_impact i
        join raw.reflections r using (reflection_id)
        where i.avg_gain <= ?
          and i.times_applied >= ?
          and r.archived = false
        order by i.avg_gain asc
        """,
        [args.max_gain, args.min_applied],
    ).fetchall()

    print(f"archive 후보: {len(candidates)}개 "
          f"(times_applied >= {args.min_applied}, avg_gain <= {args.max_gain})")
    for c in candidates:
        print(f"  {c[0][:8]}  applied={c[1]}  avg_gain={c[2]}  [{c[3]}]  {c[4]}")

    if not candidates:
        print("없음.")
        conn.close()
        return

    if args.dry_run:
        print("\n--dry-run: 변경하지 않음.")
        conn.close()
        return

    ids = [c[0] for c in candidates]
    conn.execute(
        f"update raw.reflections set archived = true where reflection_id in ({','.join('?' * len(ids))})",
        ids,
    )
    print(f"\n{len(ids)}개 교훈 archived.")
    conn.close()


if __name__ == "__main__":
    main()
