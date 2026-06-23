from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.coder import generate_code, _extract_code


def _mock_resp(content: str) -> MagicMock:
    m = MagicMock()
    m.message.content = content
    return m


_VALID_PATCH = """
import polars as pl

class Patch:
    action_type = "feature_engineering"
    changed_stages = ["feature"]
    rationale = "Drop low-variance columns."

    def feature_transform(self, train, valid, target, ctx):
        cols = [c for c in train.columns if c != target]
        return train.select(cols), valid.select(cols)
""".strip()


def test_extract_code_from_markdown() -> None:
    text = f"Here is the code:\n```python\n{_VALID_PATCH}\n```\nDone."
    assert _extract_code(text) == _VALID_PATCH


def test_extract_code_plain() -> None:
    assert _extract_code(_VALID_PATCH) == _VALID_PATCH


def test_generate_code_returns_string() -> None:
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_PATCH)
        result = generate_code(
            hypothesis="Drop low-variance columns",
            action_type="feature_engineering",
            eda_card="n_rows=165034, task=binary",
        )
    assert "Patch" in result
    assert "feature_transform" in result


def test_generate_code_with_error_feedback() -> None:
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_PATCH)
        generate_code(
            hypothesis="Fix missing class",
            action_type="feature_engineering",
            eda_card="x",
            error_feedback="missing class definition: Patch",
        )
        messages = mock_client.return_value.chat.call_args.kwargs["messages"]
    # system message = static contract; user message = dynamic context + error_feedback
    system_msg = next(m for m in messages if m["role"] == "system")
    user_msg   = next(m for m in messages if m["role"] == "user")
    assert "Patch" in system_msg["content"]
    assert "missing class definition" in user_msg["content"]


def test_generate_code_system_message_is_contract() -> None:
    """정적 contract가 system 메시지, 동적 컨텍스트는 user 메시지여야 한다."""
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_PATCH)
        generate_code(
            hypothesis="Drop low-variance columns",
            action_type="feature_engineering",
            eda_card="n_rows=100",
        )
        messages = mock_client.return_value.chat.call_args.kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    # EDA Card는 user 메시지에 있어야 함
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "n_rows=100" in user_content
