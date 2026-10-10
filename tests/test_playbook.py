"""정적 playbook(#235)이 Strategist/Coder 프롬프트에 실제로 주입되는지 배선 테스트."""
from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

from agents.playbook import CODER_PLAYBOOK, STRATEGIST_PLAYBOOK
from evaluator.contract import _PANDAS_ONLY_ATTRS


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
        assert "import " not in text
    # STRATEGIST 출력은 가설 문장이라 pandas 언급 자체가 불필요 — Coder playbook은
    # polars 규약을 재확인하느라 부정 맥락으로 "pandas"를 쓴다(아래 별도 테스트가 API를 검사).
    assert "pandas" not in STRATEGIST_PLAYBOOK.lower()


def test_coder_playbook_uses_no_pandas_only_api() -> None:
    """#136: playbook의 예제 코드가 정적 가드에서 거부되는 pandas 관용구를 유도하면 안 된다."""
    for attr in _PANDAS_ONLY_ATTRS:
        assert not re.search(rf"\.{re.escape(attr)}\s*\(", CODER_PLAYBOOK), attr


def test_playbooks_warn_that_every_param_candidate_costs_a_full_refit() -> None:
    """ADR-068: "6-12 wide candidates" 지시가 n_estimators 5000~7000 후보 6개를 유도해 preselect가 예산을 태웠다."""
    coder, strategist = " ".join(CODER_PLAYBOOK.split()), " ".join(STRATEGIST_PLAYBOOK.split())
    assert "6-12" not in coder
    assert "3-6 genuinely different" in coder
    assert "CPU budget" in coder
    assert "CPU budget" in strategist


def test_strategist_injects_playbook_into_prompt() -> None:
    with patch("agents.llm_retry._client") as mock_client:
        mock_client.return_value.chat.return_value = _strategist_resp()
        from agents.strategist import strategize

        strategize(eda_card="n_rows=100k", lessons=[], stage="reflexion")
        messages = mock_client.return_value.chat.call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert STRATEGIST_PLAYBOOK in user_content


def test_coder_injects_playbook_into_system_message() -> None:
    from agents.coder import generate_code

    for action_type in ("feature_engineering", "bootstrap"):
        with patch("agents.llm_retry._client") as mock_client:
            mock_client.return_value.chat.return_value = _coder_resp()
            generate_code(hypothesis="h", action_type=action_type, eda_card="n_rows=100")
            messages = mock_client.return_value.chat.call_args.kwargs["messages"]
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert CODER_PLAYBOOK in system_content


def test_playbooks_warn_that_feature_count_drives_cv_cost() -> None:
    """FE/preprocessing 233건 중 162건이 CPU 예산 kill — 컬럼 수가 비용을 키운다는 안내와 개수 상한."""
    coder, strategist = " ".join(CODER_PLAYBOOK.split()), " ".join(STRATEGIST_PLAYBOOK.split())
    assert "at most about 8 new columns" in coder
    assert "killed by the CPU budget" in coder
    assert "CV time grows with the column count" in strategist
