from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import polars as pl
import pytest

from agents.strategist import StrategyDecision
from agents.reflector import ReflectionOutput
from cycle.run import CycleConfig, CycleResult, run_cycle

_SCHEMA = (Path(__file__).parent.parent / "store" / "schema.sql").read_text()

_VALID_CODE = """
import polars as pl
from lightgbm import LGBMClassifier

def feature_fn(train, valid, target):
    return train.drop([target]), valid.drop([target])

def model_fn(params):
    return LGBMClassifier(n_estimators=10, verbose=-1)
""".strip()


@pytest.fixture()
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute(_SCHEMA)
    c.execute(
        """
        insert into raw.competitions (competition_id, name, task_type, metric, metric_sign)
        values ('test-comp', 'test', 'binary', 'auc', 1)
        """
    )
    return c


@pytest.fixture()
def train_df() -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200
    return pl.DataFrame({
        "f1": rng.standard_normal(n).tolist(),
        "f2": rng.standard_normal(n).tolist(),
        "target": rng.integers(0, 2, n).tolist(),
    })


def _strategy() -> StrategyDecision:
    return StrategyDecision(
        hypothesis="LightGBM baseline",
        action_type="model_swap",
        reflection_ids=[],
    )


def _reflection() -> ReflectionOutput:
    return ReflectionOutput(
        reflection_id="r-test",
        embedded_text="baseline worked",
        full_lesson="LightGBM baseline is a solid start.",
        generality="L3_general",
        reflector_label="neutral",
    )


def test_successful_cycle(conn: duckdb.DuckDBPyConnection, train_df: pl.DataFrame) -> None:
    config = CycleConfig(
        competition_id="test-comp",
        train=train_df,
        target_col="target",
        metric="auc",
        stage="bootstrap",
        eda_card="n_rows=200, task=binary",
    )

    with (
        patch("cycle.run.search", return_value=[]),
        patch("cycle.run.strategize", return_value=_strategy()),
        patch("cycle.run.generate_code", return_value=_VALID_CODE),
        patch("cycle.run.reflect", return_value=_reflection()),
        patch("memory.retriever.embed", return_value=[0.0] * 1024),
    ):
        result = run_cycle(conn, config)

    assert isinstance(result, CycleResult)
    assert result.cv_score is not None
    assert result.cv_score > 0.0
    assert result.error_trace is None
    assert result.reflection_id == "r-test"

    row = conn.execute(
        "select attempt_id, stage, label from raw.attempts where competition_id='test-comp'"
    ).fetchone()
    assert row is not None
    assert row[1] == "bootstrap"


def test_cycle_with_code_error(conn: duckdb.DuckDBPyConnection, train_df: pl.DataFrame) -> None:
    bad_code = "def feature_fn(): pass"  # wrong arity, missing model_fn

    config = CycleConfig(
        competition_id="test-comp",
        train=train_df,
        target_col="target",
        metric="auc",
        stage="reflexion",
        eda_card="x",
    )

    with (
        patch("cycle.run.search", return_value=[]),
        patch("cycle.run.strategize", return_value=_strategy()),
        patch("cycle.run.generate_code", return_value=bad_code),
        patch("cycle.run.reflect", return_value=_reflection()),
        patch("memory.retriever.embed", return_value=[0.0] * 1024),
    ):
        result = run_cycle(conn, config)

    assert result.cv_score is None
    assert result.error_trace is not None
    assert result.reflection_id == "r-test"  # Reflect still runs


def test_attempt_persisted(conn: duckdb.DuckDBPyConnection, train_df: pl.DataFrame) -> None:
    config = CycleConfig(
        competition_id="test-comp",
        train=train_df,
        target_col="target",
        metric="auc",
        stage="reflexion",
        eda_card="x",
    )

    with (
        patch("cycle.run.search", return_value=[]),
        patch("cycle.run.strategize", return_value=_strategy()),
        patch("cycle.run.generate_code", return_value=_VALID_CODE),
        patch("cycle.run.reflect", return_value=_reflection()),
        patch("memory.retriever.embed", return_value=[0.0] * 1024),
    ):
        result = run_cycle(conn, config)

    count = conn.execute(
        "select count(*) from raw.attempts where attempt_id = ?", [result.attempt_id]
    ).fetchone()[0]
    assert count == 1
