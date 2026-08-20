"""Super-cycle attempt gate — Airflow task, retrieve와 병렬로 뜬다 (#203).

promote가 3개 attempt 중 가장 느린 것(대개 실패, 2026-08 실측: rank-1 승격률 25.9%·에러율
41.6%)까지 기다리며 낭비하는 시간을 줄인다. Airflow task 상태가 아니라 raw.attempts를 직접
폴링해 "충분히 모였다" 판단이 서면 exit 0으로 즉시 반환 — attempt_i 자신은 취소하지 않고
기존 45분 execution_timeout까지 그대로 돈다(#176/#195의 "예산은 넉넉히, 검열하지 않는다"
원칙과 동일 — 여기서 줄이는 건 orchestration의 대기 시간이지 attempt 자신의 실행 시간이 아님).

Usage (container):
    uv run python -m bin.run_attempt_gate_task --run-id <run_id> --expected 3 --min-done 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

_POLL_SEC = 15  # bin/airflow_client.py의 폴링 주기와 맞춤
_DEFAULT_GRACE_SEC = 900  # 실측 span p90(487s)의 ~1.8배 여유
_DEFAULT_MAX_WAIT_SEC = 3000  # attempt execution_timeout(45분=2700s) + 5분 슬랙
_CONTEXT_LOOKUP_RETRY_SEC = 60


def _should_fire(
    n_success: int,
    n_done: int,
    n_expected: int,
    span_sec: float | None,
    waited_sec: float,
    min_done: int,
    grace_sec: float,
    max_wait_sec: float,
) -> str | None:
    """발화 사유(quorum/all_done/grace/maxwait) 또는 아직 대기(None). 순수 함수 — 테스트는
    이것만 검증.

    quorum은 n_success(성공만) 기준 — n_done(성공+실패)으로 세면 실패 2개만으로도
    발화해 promote가 winner 없이 바로 종료되고(#214), 아직 도는 세 번째(성공할
    수도 있는) attempt는 이 사이클에서 영영 빠진다. grace/maxwait는 "뭐라도 결과가
    나오는 대로 무기한 대기하지 않는다"는 안전장치라 n_done/경과시간 그대로 둔다 —
    전부 에러여도 grace/maxwait는 발화해야 사이클이 안 막힌다.

    n_done>=n_expected(전부 끝남)면 성공 개수와 무관하게 즉시 발화한다(#219) — 3개가
    전부 몇 분 만에 에러로 끝나도 이 조건이 없으면 성공이 안 나온 채로 grace(900s)가
    찰 때까지 남은 시간을 그냥 흘려보낸다. 더 기다려도 나올 게 없는 상태다."""
    if n_success >= min_done:
        return "quorum"
    if n_done >= n_expected:
        return "all_done"
    if n_done >= 1 and span_sec is not None and span_sec >= grace_sec:
        return "grace"
    if waited_sec >= max_wait_sec:
        return "maxwait"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected", type=int, default=3)
    parser.add_argument("--min-done", type=int, default=2)
    args = parser.parse_args()

    grace_sec = float(os.environ.get("RONDO_GATE_GRACE_SEC", str(_DEFAULT_GRACE_SEC)))
    max_wait_sec = float(os.environ.get("RONDO_GATE_MAX_WAIT_SEC", str(_DEFAULT_MAX_WAIT_SEC)))

    sys.path.insert(0, str(ROOT))
    from store.db import connect

    conn = connect(apply_schema=False)

    super_cycle_id: str | None = None
    lookup_deadline = time.monotonic() + _CONTEXT_LOOKUP_RETRY_SEC
    while time.monotonic() < lookup_deadline:
        row = conn.execute(
            "SELECT super_cycle_id FROM raw.super_cycle_context WHERE run_id = %s",
            [args.run_id],
        ).fetchone()
        if row:
            super_cycle_id = row[0]
            break
        time.sleep(5)

    if super_cycle_id is None:
        # gate가 사이클을 죽이면 안 된다 — retrieve가 아직 안 끝났거나 일시적 지연일 수
        # 있으므로 조용히 넘긴다. promote는 자기 컨텍스트를 스스로 다시 조회한다.
        print(f"[run_attempt_gate_task] no context for run_id={args.run_id} — passing through")
        conn.close()
        sys.exit(0)

    start = time.monotonic()
    while True:
        waited_sec = time.monotonic() - start
        try:
            # run_ts는 naive timestamp 컬럼(schema.sql) — now()를 그대로 빼면 세션 타임존만큼
            # 조용히 틀어진다. AT TIME ZONE 'utc' 캐스트를 지우지 말 것.
            row = conn.execute(
                """
                SELECT count(*) FILTER (WHERE cv_score IS NOT NULL),
                       count(*) FILTER (WHERE cv_score IS NOT NULL OR error_trace IS NOT NULL),
                       EXTRACT(EPOCH FROM ((now() AT TIME ZONE 'utc') - min(run_ts)))
                FROM raw.attempts
                WHERE super_cycle_id = %s
                """,
                [super_cycle_id],
            ).fetchone()
            n_success, n_done, span_sec = (row[0] or 0, row[1] or 0, row[2]) if row else (0, 0, None)
        except Exception as exc:
            print(f"[run_attempt_gate_task] poll failed(무시하고 계속): {exc}", file=sys.stderr)
            time.sleep(_POLL_SEC)
            continue

        reason = _should_fire(
            n_success, n_done, args.expected, span_sec, waited_sec, args.min_done, grace_sec, max_wait_sec,
        )
        if reason:
            print(
                f"[run_attempt_gate_task] fired reason={reason} n_success={n_success} n_done={n_done} "
                f"span_sec={span_sec} waited_sec={waited_sec:.0f} super_cycle={super_cycle_id[:8]}"
            )
            break
        time.sleep(_POLL_SEC)

    conn.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
