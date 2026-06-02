from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.coder import generate_code, _extract_code


def _mock_resp(content: str) -> MagicMock:
    m = MagicMock()
    m.message.content = content
    return m


_VALID_CODE = """
import polars as pl
from lightgbm import LGBMClassifier

def feature_fn(train, valid, target):
    drop = [target]
    return train.drop(drop), valid.drop(drop)

def model_fn(params):
    return LGBMClassifier(**params)
""".strip()


def test_extract_code_from_markdown() -> None:
    text = f"Here is the code:\n```python\n{_VALID_CODE}\n```\nDone."
    assert _extract_code(text) == _VALID_CODE


def test_extract_code_plain() -> None:
    assert _extract_code(_VALID_CODE) == _VALID_CODE


def test_generate_code_returns_string() -> None:
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_CODE)
        result = generate_code(
            hypothesis="Use LightGBM baseline",
            action_type="model_swap",
            eda_card="n_rows=165034, task=binary",
        )
    assert "feature_fn" in result
    assert "model_fn" in result


def test_generate_code_with_error_feedback() -> None:
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_CODE)
        generate_code(
            hypothesis="Fix missing function",
            action_type="feature_engineering",
            eda_card="x",
            error_feedback="missing function definition: model_fn",
        )
        prompt = mock_client.return_value.chat.call_args.kwargs["messages"][0]["content"]
    assert "missing function definition" in prompt
