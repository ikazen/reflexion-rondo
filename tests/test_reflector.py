from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from agents.reflector import AttemptContext, ReflectionOutput, reflect

_SCHEMA = (Path(__file__).parent.parent / "store" / "schema.sql").read_text()


@pytest.fixture()
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute(_SCHEMA)
    return c


def _mock_response(
    embedded_text: str,
    full_lesson: str,
    generality: str,
    reflector_label: str,
) -> MagicMock:
    msg = MagicMock()
    msg.message.content = json.dumps({
        "embedded_text": embedded_text,
        "full_lesson": full_lesson,
        "generality": generality,
        "reflector_label": reflector_label,
    })
    return msg


_CTX = AttemptContext(
    hypothesis="Try target encoding on Geography",
    action_type="feature_engineering",
    code="def build_features(train, valid, target): ...",
    cv_score=0.8721,
    cv_fold_var=0.0003,
    gain_vs_best=0.005,
    label="jump",
    retrieved_ids=["r1"],
)


def test_reflect_returns_output_and_stores_in_db(conn: duckdb.DuckDBPyConnection) -> None:
    mock_resp = _mock_response(
        "Target encoding on Geography improved AUC by 0.005.",
        "Target encoding on high-cardinality categorical columns reduces noise vs label encoding.",
        "L2_class",
        "jump",
    )

    with (
        patch("agents.reflector._client") as mock_client,
        patch("agents.reflector.insert_reflection") as mock_insert,
    ):
        mock_client.return_value.chat.return_value = mock_resp
        result = reflect(conn, attempt_id="a1", competition_id="comp1", context=_CTX)

    assert isinstance(result, ReflectionOutput)
    assert result.generality == "L2_class"
    assert result.reflector_label == "jump"
    assert result.reflection_id  # non-empty UUID

    mock_insert.assert_called_once()
    call_kwargs = mock_insert.call_args.kwargs
    assert call_kwargs["attempt_id"] == "a1"
    assert call_kwargs["competition_id"] == "comp1"
    assert call_kwargs["label"] == "jump"
    assert call_kwargs["gain_vs_best"] == 0.005


def test_reflect_persists_to_db(conn: duckdb.DuckDBPyConnection) -> None:
    mock_resp = _mock_response(
        "summary", "detailed lesson", "L3_general", "neutral"
    )

    fake_vec = [0.0] * 1024
    fake_vec[0] = 1.0

    with (
        patch("agents.reflector._client") as mock_client,
        patch("memory.retriever.embed", return_value=fake_vec),
    ):
        mock_client.return_value.chat.return_value = mock_resp
        result = reflect(conn, attempt_id="a1", competition_id="comp1", context=_CTX)

    row = conn.execute(
        "select reflection_id, generality, full_lesson from raw.reflections where reflection_id = ?",
        [result.reflection_id],
    ).fetchone()

    assert row is not None
    assert row[1] == "L3_general"
    assert row[2] == "detailed lesson"


def test_error_trace_included_in_prompt(conn: duckdb.DuckDBPyConnection) -> None:
    ctx_with_error = AttemptContext(
        hypothesis="Try XGBoost",
        action_type="model_swap",
        code="def build_features(...): ...",
        cv_score=0.0,
        cv_fold_var=0.0,
        gain_vs_best=None,
        label="regression",
        error_trace="ValueError: feature mismatch",
    )
    mock_resp = _mock_response("error lesson", "XGBoost failed due to feature mismatch.", "L1_local", "regression")

    with (
        patch("agents.reflector._client") as mock_client,
        patch("agents.reflector.insert_reflection"),
    ):
        mock_client.return_value.chat.return_value = mock_resp
        captured_prompt = []

        def capture_chat(**kwargs):
            captured_prompt.append(kwargs["messages"][0]["content"])
            return mock_resp

        mock_client.return_value.chat.side_effect = capture_chat
        reflect(conn, attempt_id="a1", competition_id="comp1", context=ctx_with_error)

    assert "ValueError: feature mismatch" in captured_prompt[0]


def test_none_gain_stored_as_zero(conn: duckdb.DuckDBPyConnection) -> None:
    ctx_no_gain = AttemptContext(
        hypothesis="first attempt",
        action_type="model_swap",
        code="...",
        cv_score=0.80,
        cv_fold_var=0.001,
        gain_vs_best=None,
        label="neutral",
    )
    mock_resp = _mock_response("first attempt lesson", "baseline lesson", "L3_general", "neutral")

    with (
        patch("agents.reflector._client") as mock_client,
        patch("agents.reflector.insert_reflection") as mock_insert,
    ):
        mock_client.return_value.chat.return_value = mock_resp
        reflect(conn, attempt_id="a1", competition_id="comp1", context=ctx_no_gain)

    assert mock_insert.call_args.kwargs["gain_vs_best"] == 0.0
