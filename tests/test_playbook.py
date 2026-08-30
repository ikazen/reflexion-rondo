"""정적 playbook(#235)이 Strategist/Coder 프롬프트에 실제로 주입되는지 배선 테스트."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agents.playbook import CODER_PLAYBOOK, STRATEGIST_PLAYBOOK


def _coder_resp() -> MagicMock:
    m = MagicMock()
    m.message.content = "```python\nclass Patch:\n    action_type = 'feature_engineering'\n```"
    return m


def _strategist_resp() -> MagicMock:
    m = MagicMock()
    m.message.content = json.dumps(
        {"hypothesis": "add group aggregates", "action_type": "feature_engineering", "reflection_ids": []}
    )
    return m


def test_playbooks_do_not_advertise_uninstalled_libs() -> None:
    for text in (STRATEGIST_PLAYBOOK, CODER_PLAYBOOK):
        lower = text.lower()
        assert "tabpfn" not in lower
        assert "pandas" not in lower
        assert "import " not in lower


def test_strategist_injects_playbook_into_prompt() -> None:
    with patch("agents.strategist._client") as mock_client:
        mock_client.return_value.chat.return_value = _strategist_resp()
        from agents.strategist import strategize

        strategize(eda_card="n_rows=100k", lessons=[], stage="reflexion")
        messages = mock_client.return_value.chat.call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert STRATEGIST_PLAYBOOK in user_content


def test_coder_injects_playbook_into_system_message() -> None:
    from agents.coder import generate_code

    for action_type in ("feature_engineering", "bootstrap"):
        with patch("agents.coder._client") as mock_client:
            mock_client.return_value.chat.return_value = _coder_resp()
            generate_code(hypothesis="h", action_type=action_type, eda_card="n_rows=100")
            messages = mock_client.return_value.chat.call_args.kwargs["messages"]
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert CODER_PLAYBOOK in system_content
