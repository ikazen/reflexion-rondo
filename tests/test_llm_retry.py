"""agents/llm_retry.py 단위 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.llm_retry import chat_with_retry


def test_chat_with_retry_succeeds_first_try_no_sleep(monkeypatch):
    monkeypatch.setattr("agents.llm_retry.time.sleep", MagicMock(side_effect=AssertionError("should not sleep")))
    client = MagicMock()
    client.chat.return_value = "ok"
    resp = chat_with_retry(lambda: client, model="m", messages=[])
    assert resp == "ok"
    assert client.chat.call_count == 1


def test_chat_with_retry_recovers_after_transient_failures(monkeypatch):
    """_CHAT_RETRY_DELAYS는 건드리지 않는다 — 그건 재시도 횟수 자체를 결정하므로
    줄이면 테스트 의도(횟수는 유지, sleep만 생략)와 어긋난다. time.sleep만 no-op."""
    monkeypatch.setattr("agents.llm_retry.time.sleep", MagicMock())
    client = MagicMock()
    client.chat.side_effect = [RuntimeError("503 overloaded"), RuntimeError("503 overloaded"), "ok"]
    resp = chat_with_retry(lambda: client, model="m", messages=[])
    assert resp == "ok"
    assert client.chat.call_count == 3


def test_chat_with_retry_raises_last_exception_after_exhausting(monkeypatch):
    monkeypatch.setattr("agents.llm_retry.time.sleep", MagicMock())
    client = MagicMock()
    client.chat.side_effect = [RuntimeError("first"), RuntimeError("second"),
                                RuntimeError("third"), RuntimeError("last")]
    with pytest.raises(RuntimeError, match="last"):
        chat_with_retry(lambda: client, model="m", messages=[])
    assert client.chat.call_count == 4  # 1 + len(_CHAT_RETRY_DELAYS)


def test_chat_with_retry_calls_client_factory_each_attempt(monkeypatch):
    """client_factory()를 매 시도 새로 호출한다 — 각 파일의 기존 _client() seam 유지 확인."""
    monkeypatch.setattr("agents.llm_retry.time.sleep", MagicMock())
    client = MagicMock()
    client.chat.side_effect = [RuntimeError("boom"), "ok"]
    factory = MagicMock(return_value=client)
    resp = chat_with_retry(factory, model="m", messages=[])
    assert resp == "ok"
    assert factory.call_count == 2
