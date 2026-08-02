"""bin/blend.py 단위 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from bin.blend import build_oof_matrix, compute_and_store_blend, fetch_oof_candidates, fit_blend


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


def test_fetch_oof_candidates_query_excludes_invalid_reason():
    """격리된(GH #96 타깃 누수 등, #99) pipeline은 blend 후보에서 제외돼야 한다 —
    부풀려진 cv_score가 blend 가중치를 오염시키면 안 된다."""
    conn = _conn_with([])
    fetch_oof_candidates(conn, "s4e1", 5)
    sql = conn.execute.call_args.args[0]
    assert "invalid_reason IS NULL" in sql


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


# --- compute_and_store_blend (#75 — 승격 시점 자동 배선) ---

def _train_df(n: int = 50) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame({"y": rng.standard_normal(n)})


def test_compute_and_store_blend_skips_when_fewer_than_two_candidates():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [("p1", [0.1] * 50, 0.9)]
    result = compute_and_store_blend(conn, "s4e1", _train_df(50), "y", "rmse")
    assert result is None
    upsert_calls = [c for c in conn.execute.call_args_list if "raw.blend_weights" in c.args[0]]
    assert upsert_calls == []


def test_compute_and_store_blend_skips_when_length_mismatch_leaves_fewer_than_two():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("p1", [0.1] * 50, 0.9),
        ("p2", [0.1] * 10, 0.85),  # 길이 불일치 — build_oof_matrix가 스킵
    ]
    result = compute_and_store_blend(conn, "s4e1", _train_df(50), "y", "rmse")
    assert result is None


def test_compute_and_store_blend_succeeds_and_upserts():
    rng = np.random.default_rng(1)
    n = 60
    train = pl.DataFrame({"y": rng.standard_normal(n)})
    target = train["y"].to_numpy()
    oof1 = (target + rng.standard_normal(n) * 0.1).tolist()
    oof2 = (target + rng.standard_normal(n) * 0.1).tolist()

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("p1", oof1, 0.9),
        ("p2", oof2, 0.88),
    ]
    result = compute_and_store_blend(conn, "s4e1", train, "y", "rmse")

    assert result is not None
    assert set(result["pipeline_ids"]) == {"p1", "p2"}
    assert len(result["weights"]) == 2
    assert np.isfinite(result["blend_cv_score"])

    upsert_calls = [c for c in conn.execute.call_args_list if "raw.blend_weights" in c.args[0]]
    assert len(upsert_calls) == 1
    assert "ON CONFLICT (competition_id) DO UPDATE" in upsert_calls[0].args[0]
    assert upsert_calls[0].args[1][0] == "s4e1"
