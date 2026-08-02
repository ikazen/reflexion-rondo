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

DEFAULT_MIN_APPLIED = 3
DEFAULT_MAX_GAIN = 0.0


def find_archive_candidates(
    conn, min_applied: int = DEFAULT_MIN_APPLIED, max_gain: float = DEFAULT_MAX_GAIN,
) -> list[tuple]:
    """(reflection_id, times_applied, avg_gain, generality, competition_id) 목록.

    반복 인용됐는데(times_applied >= min_applied) 평균 gain이 임계 이하인 교훈 —
    표본 부족으로 인한 노이즈를 min_applied 문턱으로 걸러낸 뒤에만 판단한다.
    """
    return conn.execute(
        """
        select r.reflection_id, i.times_applied, round(i.avg_gain, 5) as avg_gain,
               r.generality, r.competition_id
        from reflection_impact i
        join raw.reflections r using (reflection_id)
        where i.avg_gain <= %s
          and i.times_applied >= %s
          and r.archived = false
        order by i.avg_gain asc
        """,
        [max_gain, min_applied],
    ).fetchall()


def archive_low_gain_lessons(
    conn, min_applied: int = DEFAULT_MIN_APPLIED, max_gain: float = DEFAULT_MAX_GAIN,
) -> list[str]:
    """저효율 교훈을 archived=true로 표기하고 archive된 reflection_id 목록을 반환한다.

    #76 — 이전엔 이 정리가 수동 CLI로만 실행돼 검색 후보 풀이 계속 커지기만 했다
    (인용률 75~97%인데 양의 gain 사실상 0%, near-duplicate 686쌍 — retriever의
    MMR 재랭킹·impact z-score 감쇠가 이미 일부 완화하지만, 검증된 저효율 교훈을
    풀에서 아예 빼는 건 그와 별개로 유효한 위생 관리다). bin/run_daemon.py가
    주기적으로 호출(#76 자동 배선).
    """
    candidates = find_archive_candidates(conn, min_applied, max_gain)
    if not candidates:
        return []
    ids = [c[0] for c in candidates]
    conn.execute(
        "update raw.reflections set archived = true where reflection_id = any(%s::text[])",
        [ids],
    )
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-applied", type=int, default=DEFAULT_MIN_APPLIED,
                        help="최소 적용 횟수 (기본 3)")
    parser.add_argument("--max-gain",    type=float, default=DEFAULT_MAX_GAIN,
                        help="archive 임계 avg_gain (기본 0.0, 이하 대상)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 변경 없이 후보만 출력")
    args = parser.parse_args()

    conn = connect()

    candidates = find_archive_candidates(conn, args.min_applied, args.max_gain)
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

    ids = archive_low_gain_lessons(conn, args.min_applied, args.max_gain)
    print(f"\n{len(ids)}개 교훈 archived.")
    conn.close()


if __name__ == "__main__":
    main()
