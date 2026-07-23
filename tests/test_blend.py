"""bin/blend.py 단위 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from bin.blend import build_oof_matrix, fetch_oof_candidates, fit_blend


def _conn_with(rows) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows
    return conn


def test_fetch_oof_candidates_parses_json_string():
    conn = _conn_with([("p1", "[0.1, 0.2, 0.3]", 0.9)])
    result = fetch_oof_candidates(conn, "s4e1", 5)
    assert result == [("p1", [0.1, 0.2, 0.3], 0.9)]


def test_fetch_oof_candidates_accepts_native_list():
    conn = _conn_with([("p1", [0.1, 0.2], 0.9)])
    result = fetch_oof_candidates(conn, "s4e1", 5)
    assert result == [("p1", [0.1, 0.2], 0.9)]


def test_fetch_oof_candidates_query_filters_null_oof_and_orders_by_cv():
    conn = _conn_with([])
    fetch_oof_candidates(conn, "s4e1", 5)
    sql = conn.execute.call_args.args[0]
    assert "oof_preds IS NOT NULL" in sql
    assert "metric_sign * p.cv_score DESC" in sql


def test_build_oof_matrix_skips_length_mismatch():
    candidates = [
        ("p1", [0.1, 0.2, 0.3], 0.9),
        ("p2", [0.1, 0.2], 0.85),  # 길이 불일치 — 스킵돼야 함
        ("p3", [0.4, 0.5, 0.6], 0.8),
    ]
    matrix, used_ids = build_oof_matrix(candidates, n_rows=3)
    assert used_ids == ["p1", "p3"]
    assert matrix.shape == (3, 2)


def test_build_oof_matrix_empty_when_all_mismatch():
    candidates = [("p1", [0.1, 0.2], 0.9)]
    matrix, used_ids = build_oof_matrix(candidates, n_rows=3)
    assert used_ids == []
    assert matrix.shape[1] == 0


def test_fit_blend_returns_nonnegative_weights():
    rng = np.random.default_rng(0)
    n = 200
    true_signal = rng.standard_normal(n)
    # 두 컬럼 다 signal과 양의 상관 — non-negative Ridge가 합리적 가중치를 낼 것으로 기대
    oof_matrix = np.column_stack([
        true_signal + rng.standard_normal(n) * 0.1,
        true_signal + rng.standard_normal(n) * 0.1,
    ])
    target = true_signal

    weights, intercept = fit_blend(oof_matrix, target)
    assert len(weights) == 2
    assert all(w >= 0 for w in weights)
    assert np.isfinite(intercept)
