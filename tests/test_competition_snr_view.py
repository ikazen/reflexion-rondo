"""competition_snr 뷰(#291, ADR-047) — 실제 Postgres에 스키마를 적용해 문법·계산
검증. 순수 SQL 뷰라 psycopg 목킹으로는 계산 자체(percentile_cont, jsonb 언네스트)를
검증할 수 없어 다른 스키마 레벨 회귀 테스트(test_auto_submit_gate.py의 ambiguous
column 테스트 등)와 같은 skipif 패턴을 쓴다.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RONDO_DB_URL"),
    reason="RONDO_DB_URL not set — DB-backed competition_snr view test skipped",
)


def test_competition_snr_view_returns_positive_finite_snr_when_data_present():
    from store.db import connect

    conn = connect(apply_schema=True)
    try:
        rows = conn.execute(
            "select competition_id, p90_p50_spread, avg_fold_var, avg_n_splits, snr"
            " from competition_snr"
        ).fetchall()
    finally:
        conn.close()

    # 스냅샷+fold_var 실측 데이터가 있는 대회가 하나도 없으면(빈 DB 등) 검증 대상이
    # 없어 스킵 — 뷰 존재·문법 자체는 이미 connect(apply_schema=True)가 확인했다.
    if not rows:
        pytest.skip("no competition has both a leaderboard snapshot and cv_fold_var yet")

    for competition_id, spread, avg_fold_var, avg_n_splits, snr in rows:
        assert spread >= 0, f"{competition_id}: p90_p50_spread must be non-negative"
        assert avg_fold_var > 0, f"{competition_id}: avg_fold_var must be positive to reach this row"
        assert avg_n_splits > 0
        assert snr is not None and snr > 0, f"{competition_id}: snr must be a positive ratio"
