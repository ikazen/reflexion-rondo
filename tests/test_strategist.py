from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.strategist import ACTION_TYPES, StrategyDecision, strategize


def _mock_response(hypothesis: str, action_type: str, reflection_ids: list[str]) -> MagicMock:
    msg = MagicMock()
    msg.message.content = json.dumps({
        "hypothesis": hypothesis,
        "action_type": action_type,
        "reflection_ids": reflection_ids,
    })
    return msg


LESSONS = [
    {"reflection_id": "r1", "generality": "L3_general", "full_lesson": "Use target encoding for high-cardinality columns."},
    {"reflection_id": "r2", "generality": "L2_class", "full_lesson": "Stratified k-fold helps with imbalanced targets."},
]


def test_returns_strategy_decision() -> None:
    mock_resp = _mock_response("Try target encoding on Geography", "feature_engineering", ["r1"])

    with patch("agents.strategist._client") as mock_client:
        mock_client.return_value.chat.return_value = mock_resp
        result = strategize(eda_card="n_rows=100k", lessons=LESSONS, stage="reflexion")

    assert isinstance(result, StrategyDecision)
    assert result.action_type == "feature_engineering"
    assert result.reflection_ids == ["r1"]
    assert "encoding" in result.hypothesis.lower()


def test_reflection_ids_subset_of_provided() -> None:
    # model hallucinated "r99" which was not in the provided lessons
    mock_resp = _mock_response("some hypothesis", "model_swap", ["r1", "r99"])

    with patch("agents.strategist._client") as mock_client:
        mock_client.return_value.chat.return_value = mock_resp
        result = strategize(eda_card="x", lessons=LESSONS, stage="reflexion")

    assert "r99" not in result.reflection_ids
    assert result.reflection_ids == ["r1"]


def test_empty_reflection_ids_allowed() -> None:
    mock_resp = _mock_response("Try LGBM with lower learning rate", "hyperparam_search", [])

    with patch("agents.strategist._client") as mock_client:
        mock_client.return_value.chat.return_value = mock_resp
        result = strategize(eda_card="x", lessons=LESSONS, stage="reflexion")

    assert result.reflection_ids == []


def test_no_lessons_still_works() -> None:
    mock_resp = _mock_response("Baseline LightGBM", "model_swap", [])

    with patch("agents.strategist._client") as mock_client:
        mock_client.return_value.chat.return_value = mock_resp
        result = strategize(eda_card="x", lessons=[], stage="bootstrap")

    assert result.action_type in ACTION_TYPES
    assert result.reflection_ids == []


def test_action_type_passthrough() -> None:
    for at in ACTION_TYPES:
        mock_resp = _mock_response("hypothesis", at, [])
        with patch("agents.strategist._client") as mock_client:
            mock_client.return_value.chat.return_value = mock_resp
            result = strategize(eda_card="x", lessons=[], stage="reflexion")
        assert result.action_type == at
