"""store.db.PgConn.transaction()의 커밋/롤백/autocommit 복원 단위 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock

from store.db import PgConn


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
