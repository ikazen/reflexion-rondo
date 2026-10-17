"""cycle.strategist_context의 "이미 시도한 것" 블록과 base 요약 단위 테스트."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from cycle.strategist_context import _TRIED_LIMIT, base_summary_lines, tried_on_base_lines

_BASE = """
import polars as pl
from xgboost import XGBClassifier

class _Wrap:
    def _build_model(self):
        return XGBClassifier(n_estimators=10)

class Patch:
    def build_model(self, params, ctx):
        return _Wrap()

    def param_candidates(self, ctx):
        return [{'max_depth': 6, 'learning_rate': 0.05}]

    def feature_transform(self, train, valid, target, ctx):
        e = [(pl.col('u') - pl.col('g')).alias('u_g'), (pl.col('g') - pl.col('r')).alias("g_r"), pl.col('u').alias('u_g')]
        return train.with_columns(*e), valid.with_columns(*e)
"""


def _conn(rows: list[tuple]) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows
    return conn


def test_tried_lines_group_rows_with_count_gain_and_hypothesis():
    rows = [
        ("preprocessing", Decimal("-0.0003187"), 28, "Replace ordinal\nencoding with one-hot"),
        ("feature_engineering", Decimal("0.0000770"), 1, None),
    ]
    lines = tried_on_base_lines(_conn(rows), "s6e6")

    assert lines[0].startswith("\n## Already tried on this base")
    assert lines[1] == "- preprocessing x28 gain=-0.000319: Replace ordinal encoding with one-hot"
    assert lines[2] == "- feature_engineering x1 gain=+0.000077: "


def test_tried_lines_are_empty_without_rows():
    assert tried_on_base_lines(_conn([]), "s6e6") == []


def test_tried_query_is_scoped_to_the_competition_and_capped():
    conn = _conn([])
    tried_on_base_lines(conn, "s6e6")

    sql, params = conn.execute.call_args[0]
    assert params == ["s6e6", "s6e6", _TRIED_LIMIT]
    assert "error_trace is null" in sql
    assert "invalid_reason is null" in sql


def test_tried_hypothesis_is_truncated():
    lines = tried_on_base_lines(_conn([("preprocessing", Decimal("0"), 2, "x" * 500)]), "s6e6")
    assert len(lines[1]) < 200


def test_base_summary_lists_model_columns_and_params_once():
    lines = base_summary_lines(_BASE)

    assert lines[0].startswith("\n## Current base pipeline")
    assert "- model classes: XGBClassifier" in lines
    assert "- engineered columns already created: u_g, g_r" in lines
    assert "- param_candidates: [{'max_depth': 6, 'learning_rate': 0.05}]" in lines


def test_base_summary_is_empty_without_code_or_recognizable_content():
    assert base_summary_lines(None) == []
    assert base_summary_lines("x = 1\n") == []


def test_base_summary_survives_unparseable_code():
    lines = base_summary_lines("def broken(:\n  return XGBClassifier()")
    assert "- model classes: XGBClassifier" in lines
    assert not any(line.startswith("- param_candidates") for line in lines)
