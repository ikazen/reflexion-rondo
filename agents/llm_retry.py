"""Ollama Cloud 클라이언트와 .chat() 호출 공용 재시도 헬퍼(일시적 5xx 대응)."""
from __future__ import annotations

import logging
import time

from ollama import ChatResponse, Client

from config import settings

_LOG = logging.getLogger(__name__)

_CHAT_RETRY_DELAYS = (1.0, 4.0, 16.0)


def _client() -> Client:
    kwargs: dict = {"host": settings.OLLAMA_CLOUD_BASE_URL}
    if settings.OLLAMA_API_KEY:
        kwargs["headers"] = {"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"}
    return Client(**kwargs)


def chat_with_retry(**chat_kwargs: object) -> ChatResponse:
    """Ollama Cloud .chat() 호출 — 지수 백오프 재시도(memory/retriever.embed와 동일 패턴).

    'model is temporarily overloaded' 같은 일시 5xx로 attempt task 전체가 크래시하는 것을 막는다.
    """
    last_exc: Exception | None = None
    for delay in (0.0, *_CHAT_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            return _client().chat(**chat_kwargs)
        except Exception as exc:
            last_exc = exc
            _LOG.warning("ollama chat 실패(재시도 예정): %s", exc)
    raise last_exc
