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



def test_no_extra_paths_omits_is_original_column():
    """EXTRA_TRAIN_PATHS 미설정(기본값 [])이면 is_original 컬럼조차 추가하지 않는다 —
    기존 대회의 스키마·동작이 완전히 불변이어야 한다."""
    comp = _fake_comp(DROP_COLS=[])
    base_df = pl.DataFrame({"x": [1, 2], "y": [0, 1]})
    with patch("store.train_data.pl.read_csv", return_value=base_df):
        result = load_train(comp)
    assert "is_original" not in result.columns
    assert result.to_dicts() == base_df.to_dicts()


def test_extra_paths_merge_flags_is_original():
    """EXTRA_TRAIN_PATHS 지정 시 base=False, 병합분=True로 플래그되고 concat된다."""
    comp = _fake_comp(DROP_COLS=[], EXTRA_TRAIN_PATHS=["original.csv"])
    base_df = pl.DataFrame({"x": [1, 2], "y": [0, 1]})
    extra_df = pl.DataFrame({"x": [3], "y": [1]})

    def fake_read_csv(path):
        return extra_df if "original.csv" in str(path) else base_df

    with patch("store.train_data.pl.read_csv", side_effect=fake_read_csv):
        result = load_train(comp)

    assert len(result) == 3
    assert result["is_original"].to_list() == [False, False, True]


def test_extra_paths_selects_only_common_columns():
    """원본 데이터셋이 base에 없는 컬럼을 가지면 버리고, base에만 있는 컬럼은 null로 채운다."""
    comp = _fake_comp(DROP_COLS=[], EXTRA_TRAIN_PATHS=["original.csv"])
    base_df = pl.DataFrame({"x": [1, 2], "y": [0, 1], "engineered": [9, 9]})
    extra_df = pl.DataFrame({"x": [3], "y": [1], "unrelated_extra_col": ["z"]})

    def fake_read_csv(path):
        return extra_df if "original.csv" in str(path) else base_df

    with patch("store.train_data.pl.read_csv", side_effect=fake_read_csv):
        result = load_train(comp)

    assert "unrelated_extra_col" not in result.columns
    assert result["engineered"].to_list() == [9, 9, None]


# MAX_TRAIN_ROWS (#84 — s4e7 11.5M행 OOM 대응)

def _big_binary_df(n: int = 1000, minority_frac: float = 0.2) -> pl.DataFrame:
    n_minority = int(n * minority_frac)
    y = [1] * n_minority + [0] * (n - n_minority)
    return pl.DataFrame({"x": list(range(n)), "y": y})


def test_no_max_train_rows_leaves_data_unchanged():
    """MAX_TRAIN_ROWS 미설정(기본)이면 행 수를 전혀 건드리지 않는다 — opt-in 원칙."""
    comp = _fake_comp(DROP_COLS=[])
    df = _big_binary_df(1000)
    with patch("store.train_data.pl.read_csv", return_value=df):
        result = load_train(comp)
    assert result.height == 1000


def test_max_train_rows_below_actual_size_is_noop():
    """MAX_TRAIN_ROWS가 실제 행 수보다 크면 그대로 둔다(축소 아님, 확장 아님)."""
    comp = _fake_comp(DROP_COLS=[], MAX_TRAIN_ROWS=5000)
    df = _big_binary_df(1000)
    with patch("store.train_data.pl.read_csv", return_value=df):
        result = load_train(comp)
    assert result.height == 1000


def test_max_train_rows_classification_preserves_class_ratio():
    """분류(IS_CLASSIFICATION=True)는 층화 샘플링으로 클래스 비율을 보존한다."""
    comp = _fake_comp(DROP_COLS=[], MAX_TRAIN_ROWS=200, IS_CLASSIFICATION=True, TARGET="y")
    df = _big_binary_df(1000, minority_frac=0.2)
    with patch("store.train_data.pl.read_csv", return_value=df):
        result = load_train(comp)
    assert result.height <= 220  # fraction 샘플이라 정확히 200은 아닐 수 있음(오차 허용)
    ratio = (result["y"] == 1).sum() / result.height
    assert 0.15 < ratio < 0.25  # 원본 20% 비율 근방 유지


def test_max_train_rows_regression_uses_plain_sample():
    """회귀(IS_CLASSIFICATION 미설정/False)는 층화 없이 단순 랜덤 샘플로 축소한다."""
    comp = _fake_comp(DROP_COLS=[], MAX_TRAIN_ROWS=200, IS_CLASSIFICATION=False, TARGET="y")
    df = pl.DataFrame({"x": list(range(1000)), "y": [float(i) for i in range(1000)]})
    with patch("store.train_data.pl.read_csv", return_value=df):
        result = load_train(comp)
    assert result.height == 200


def test_max_train_rows_is_deterministic_across_calls():
    """cross-seed confirm/merge-verify가 같은 cv_score를 재현하려면 같은 표본이어야 한다
    — 고정 seed로 매 호출 동일 결과."""
    comp = _fake_comp(DROP_COLS=[], MAX_TRAIN_ROWS=200, IS_CLASSIFICATION=True, TARGET="y")
    df = _big_binary_df(1000, minority_frac=0.2)
    with patch("store.train_data.pl.read_csv", return_value=df):
        result1 = load_train(comp)
    with patch("store.train_data.pl.read_csv", return_value=df):
        result2 = load_train(comp)
    assert result1.to_dicts() == result2.to_dicts()


def test_max_train_rows_applies_after_extra_paths_merge():
    """EXTRA_TRAIN_PATHS 병합 이후의 합산 행 수 기준으로 축소한다."""
    comp = _fake_comp(
        DROP_COLS=[], EXTRA_TRAIN_PATHS=["original.csv"],
        MAX_TRAIN_ROWS=200, IS_CLASSIFICATION=True, TARGET="y",
    )
    base_df = _big_binary_df(600, minority_frac=0.2)
    extra_df = _big_binary_df(600, minority_frac=0.2)

    def fake_read_csv(path):
        return extra_df if "original.csv" in str(path) else base_df

    with patch("store.train_data.pl.read_csv", side_effect=fake_read_csv):
        result = load_train(comp)
    assert result.height <= 220
