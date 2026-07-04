"""store/train_data.py: load_train 단위 테스트.

6곳(bin/run_attempt_task.py 등)에 중복돼 있던 S3-or-local train.csv 로딩 로직을
하나로 통합한 것 — 기존 인라인 동작과 100% 동일해야 한다(순수 추출).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import polars as pl

from store.train_data import load_train


def _fake_comp(**overrides) -> SimpleNamespace:
    base = dict(
        S3_DATA_PATH=None,
        DATA_DIR=Path("/data/s4e1"),
        DROP_COLS=["id"],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_df() -> MagicMock:
    df = MagicMock(spec=pl.DataFrame)
    df.drop.return_value = "dropped"
    return df


def test_local_path_used_when_no_s3_data_path():
    comp = _fake_comp(S3_DATA_PATH=None)
    with patch("store.train_data.pl.read_csv", return_value=_fake_df()) as mock_read:
        result = load_train(comp)
    mock_read.assert_called_once_with(comp.DATA_DIR / "train.csv")
    assert result == "dropped"


def test_local_path_used_when_s3_path_set_but_minio_endpoint_missing():
    comp = _fake_comp(S3_DATA_PATH="s4e1/data/")
    with patch("store.train_data._MINIO_ENDPOINT", ""), \
         patch("store.train_data.pl.read_csv", return_value=_fake_df()) as mock_read:
        load_train(comp)
    mock_read.assert_called_once_with(comp.DATA_DIR / "train.csv")


def test_s3_path_used_when_s3_data_path_and_minio_endpoint_set():
    comp = _fake_comp(S3_DATA_PATH="s4e1/data/")
    with patch("store.train_data._MINIO_ENDPOINT", "http://minio.internal"), \
         patch("store.train_data.pl.read_csv", return_value=_fake_df()) as mock_read:
        load_train(comp)
    mock_read.assert_called_once_with("http://minio.internal/kaggle/s4e1/data/train.csv")


def test_drop_cols_applied():
    comp = _fake_comp(DROP_COLS=["id", "Surname"])
    df = _fake_df()
    with patch("store.train_data.pl.read_csv", return_value=df):
        result = load_train(comp)
    df.drop.assert_called_once_with(["id", "Surname"])
    assert result == "dropped"
