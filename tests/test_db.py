"""store.db.PgConn.transaction() 및 insert_pipeline oof_preds 직렬화 단위 테스트."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from store.db import PgConn, insert_pipeline


def _make_pgconn() -> tuple[PgConn, MagicMock]:
    raw = MagicMock()
    raw.autocommit = True
    conn = PgConn(raw)
    return conn, raw


def test_transaction_commits_on_success():
    conn, raw = _make_pgconn()
    with conn.transaction():
        pass
    raw.commit.assert_called_once()
    raw.rollback.assert_not_called()


def test_transaction_restores_autocommit_on_success():
    conn, raw = _make_pgconn()
    raw.autocommit = True
    with conn.transaction():
        pass
    assert raw.autocommit is True


def test_transaction_rolls_back_on_exception():
    conn, raw = _make_pgconn()
    try:
        with conn.transaction():
            raise ValueError("boom")
    except ValueError:
        pass
    raw.rollback.assert_called_once()
    raw.commit.assert_not_called()


def test_transaction_restores_autocommit_on_exception():
    conn, raw = _make_pgconn()
    raw.autocommit = True
    try:
        with conn.transaction():
            raise RuntimeError("fail")
    except RuntimeError:
        pass
    assert raw.autocommit is True


def test_transaction_reraises_exception():
    conn, raw = _make_pgconn()
    import pytest
    with pytest.raises(ValueError, match="boom"):
        with conn.transaction():
            raise ValueError("boom")


def _insert_pipeline_oof_param(oof_preds):
    conn = MagicMock()
    insert_pipeline(
        conn,
        pipeline_id="p1",
        attempt_id="a1",
        competition_id="playground-series-s6e8",
        fingerprint_snapshot={},
        code="x",
        cv_score=0.9,
        gain_vs_best=0.01,
        oof_preds=oof_preds,
    )
    return conn.execute.call_args.args[1][8]


def test_insert_pipeline_nan_oof_becomes_json_null():
    # #228 is_original 행 위치는 NaN으로 남는다 — jsonb가 거부하지 않도록 null로 눕혀야 한다.
    oof_json = _insert_pipeline_oof_param([0.1, float("nan"), 0.3, float("inf"), -float("inf")])
    assert json.loads(oof_json) == [0.1, None, 0.3, None, None]


def test_insert_pipeline_none_oof_stays_none():
    assert _insert_pipeline_oof_param(None) is None
