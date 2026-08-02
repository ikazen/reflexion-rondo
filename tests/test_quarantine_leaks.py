"""bin/quarantine_leaks.py 단위 테스트.

_scan_pipeline은 evaluator.harness._check_preprocess_target_leak을 그대로
재사용하므로 그 로직 자체의 회귀는 tests/test_harness.py가 커버한다 — 여기서는
스캐너의 배선(코드 exec → PatchedPipeline 구성 → 실제 대회 데이터로 검사 →
invalid_reason 문자열 반환)이 실제 대회 설정(s5e10, 로컬 CSV)으로 끝까지 도는지만
확인한다.
"""
from __future__ import annotations

import importlib

from bin.quarantine_leaks import _competition_id_to_slug, _scan_pipeline

_S5E10 = importlib.import_module("config.competitions.s5e10")

_LEAKY_CODE = """
import numpy as np
import polars as pl


class Patch:
    action_type = "preprocessing"
    changed_stages = ["preprocess"]
    rationale = "quantile bin from valid target (s5e10 실제 사고 재현)"

    def preprocess(self, train, valid, target, ctx):
        y_train = train[target].to_numpy()
        edges = np.unique(np.quantile(y_train, np.linspace(0, 1, 11)))

        def to_bins(arr):
            return pl.Series(np.digitize(arr, edges, right=False) - 1).cast(pl.Int32)

        train = train.with_columns(to_bins(y_train).alias("target_bin"))
        valid = valid.with_columns(to_bins(valid[target].to_numpy()).alias("target_bin"))
        return train, valid
"""

_CLEAN_CODE = """
import polars as pl


class Patch:
    action_type = "feature_engineering"
    changed_stages = ["feature_transform"]
    rationale = "simple interaction feature, no target access"

    def feature_transform(self, train, valid, target, ctx):
        cols = [c for c in train.columns if c != target]
        return train.select(cols), valid.select(cols)
"""


def test_competition_id_to_slug_includes_known_competitions():
    mapping = _competition_id_to_slug()
    assert mapping.get("playground-series-s5e10") == "s5e10"
    assert mapping.get("playground-series-s4e1") == "s4e1"


def test_scan_pipeline_flags_valid_target_read_in_preprocess():
    reason = _scan_pipeline(_S5E10, _LEAKY_CODE)
    assert reason is not None
    assert reason.startswith("target_leak_preprocess")


def test_scan_pipeline_returns_none_for_clean_pipeline():
    assert _scan_pipeline(_S5E10, _CLEAN_CODE) is None


def test_scan_pipeline_handles_broken_code_gracefully():
    """exec 자체가 실패하는 코드(SyntaxError 등)는 판정 불가이지 누수 확정이
    아니다. None(스킵)을 반환해야 한다 — 예외를 던지지도, 격리 대상으로 잘못
    집계되지도 않아야 한다."""
    reason = _scan_pipeline(_S5E10, "class Patch:\n    def preprocess(:\n        pass\n")
    assert reason is None


def test_scan_pipeline_data_load_failure_returns_none_not_quarantine_reason():
    """load_train이 실패하면(로컬에 train.csv 없음 등 순수 환경 문제) None을
    반환해야 한다 — non-None 문자열을 반환하면 호출부(scan())가 격리 대상으로
    집계한다."""

    class _CompWithMissingData:
        COMPETITION_ID = "playground-series-does-not-exist"
        TARGET = "y"
        METRIC = "auc"
        IS_CLASSIFICATION = True
        DATA_DIR = __import__("pathlib").Path("/nonexistent/path")
        S3_DATA_PATH = None
        DROP_COLS = []

    reason = _scan_pipeline(_CompWithMissingData(), _CLEAN_CODE)
    assert reason is None


def test_scan_pipeline_missing_patch_class_returns_none():
    """Patch 클래스가 없는 코드는 판정 대상이 아니므로 None(스킵)을 반환한다."""
    assert _scan_pipeline(_S5E10, "x = 1\n") is None
