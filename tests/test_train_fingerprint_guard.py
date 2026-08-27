"""cycle.run._train_fingerprint_guard + promotion.train_data_fingerprint 단위 테스트 (#258, ADR-040).

load_train 설정(EXTRA_TRAIN_PATHS/MAX_TRAIN_ROWS/DROP_COLS)이 바뀌면 확정 baseline의
cv_score가 옛 데이터 기준이라 새 attempt와 비교 불가능해진다 — 지문이 어긋나면
게이트가 멈춰야 한다.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import polars as pl
import pytest

from cycle.promotion import train_data_fingerprint
from cycle.run import TrainFingerprintMismatchError, _train_fingerprint_guard

_BASE = pl.DataFrame({"a": [1.0, 2.0, 3.0], "y": [0.0, 1.0, 0.0]})
_EXTRA_COL = _BASE.with_columns(pl.lit(False).alias("is_original"))
_MORE_ROWS = pl.concat([_BASE, _BASE])


def test_fingerprint_changes_on_extra_column():
    assert train_data_fingerprint(_BASE) != train_data_fingerprint(_EXTRA_COL)


def test_fingerprint_changes_on_row_count():
    assert train_data_fingerprint(_BASE) != train_data_fingerprint(_MORE_ROWS)


def test_fingerprint_stable_for_same_frame():
    assert train_data_fingerprint(_BASE) == train_data_fingerprint(_BASE.clone())


def _conn(stored):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (stored,)
    return conn


def test_guard_stamps_when_unset():
    conn = _conn(None)
    _train_fingerprint_guard(conn, "s4e11", _BASE)
    writes = [c for c in conn.execute.call_args_list if "set train_fingerprint" in c.args[0]]
    assert len(writes) == 1
    assert writes[0].args[1] == [train_data_fingerprint(_BASE), "s4e11"]


def test_guard_passes_when_match():
    conn = _conn(train_data_fingerprint(_BASE))
    _train_fingerprint_guard(conn, "s4e11", _BASE)
    writes = [c for c in conn.execute.call_args_list if c.args[0].strip().lower().startswith("update")]
    assert writes == []


def test_guard_raises_and_pauses_on_mismatch():
    conn = _conn(train_data_fingerprint(_MORE_ROWS))  # 옛 지문(다른 행수)
    with pytest.raises(TrainFingerprintMismatchError):
        _train_fingerprint_guard(conn, "s4e11", _BASE)
    pauses = [c for c in conn.execute.call_args_list if "auto_submit_paused_reason" in c.args[0]]
    assert len(pauses) == 1
    assert "coalesce(auto_submit_paused_reason" in pauses[0].args[0]
    assert "establish_baseline --remeasure --competition s4e11" in pauses[0].args[1][0]
