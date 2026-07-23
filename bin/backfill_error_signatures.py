"""raw.attempts.error_signature 소급 채움 (#66 P2).

error_trace는 있지만 error_signature가 아직 없는 행에 error_pitfalls.normalize_error()를
적용해 채운다. idempotent — 이미 채워진 행은 건드리지 않으므로 재실행 안전(정규화 로직이
바뀌면 다시 돌려 갱신하고 싶을 땐 --force로 전체 재계산).

Usage:
    uv run python -m bin.backfill_error_signatures              # 실제 UPDATE
    uv run python -m bin.backfill_error_signatures --dry-run    # 변경 없이 대상만 집계
    uv run python -m bin.backfill_error_signatures --force      # error_signature 유무 무관 전체 재계산
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from cycle.error_pitfalls import normalize_error
from store.db import connect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 변경 없이 대상 건수만 출력")
    parser.add_argument("--force", action="store_true",
                        help="error_signature가 이미 있는 행도 재계산")
    args = parser.parse_args()

    conn = connect()

    where = "error_trace is not null" if args.force else \
            "error_trace is not null and error_signature is null"
    rows = conn.execute(
        f"select attempt_id, error_trace from raw.attempts where {where}"
    ).fetchall()

    print(f"대상: {len(rows)}개 (force={args.force})")
    if not rows:
        conn.close()
        return

    by_signature: dict[str | None, list[str]] = defaultdict(list)
    for attempt_id, error_trace in rows:
        sig = normalize_error(error_trace)
        by_signature[sig].append(attempt_id)

    n_syntax = len(by_signature.get(None, []))
    n_signed = len(rows) - n_syntax
    print(f"정규화: signature 부여 {n_signed}개, None(SyntaxError류) {n_syntax}개")

    if args.dry_run:
        for sig, ids in sorted(by_signature.items(), key=lambda kv: -len(kv[1]))[:10]:
            print(f"  {len(ids):4d}  {sig!r}")
        print("\n--dry-run: 변경하지 않음.")
        conn.close()
        return

    for sig, ids in by_signature.items():
        conn.execute(
            "update raw.attempts set error_signature = %s where attempt_id = any(%s::text[])",
            [sig, ids],
        )
    print(f"\n{len(rows)}개 행 error_signature 갱신 완료.")
    conn.close()


if __name__ == "__main__":
    main()
