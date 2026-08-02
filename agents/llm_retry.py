"""Ollama Cloud .chat() 호출 공용 재시도 헬퍼.

agents/coder.py, agents/strategist.py, agents/reflector.py가 전부 동일한
Ollama Cloud 엔드포인트를 같은 방식(_client().chat(...))으로 호출하고 같은
실패 클래스(일시적 5xx)에 노출돼 있어 공용 헬퍼로 뺀다.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

_LOG = logging.getLogger(__name__)

_CHAT_RETRY_DELAYS = (1.0, 4.0, 16.0)


def chat_with_retry(client_factory: Callable[[], object], **chat_kwargs):
    """Ollama Cloud .chat() 호출 — 지수 백오프 재시도(memory/retriever.embed와 동일 패턴).

    'model is temporarily overloaded' 같은 일시 5xx로 attempt task 전체가
    크래시하는 것을 막는다. client_factory를 매 시도 새로 호출해 각 파일의
    기존 _client() 시드 형태를 그대로 유지한다(테스트가 patch하는 지점 불변).
    """
    last_exc: Exception | None = None
    for delay in (0.0, *_CHAT_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            return client_factory().chat(**chat_kwargs)
        except Exception as exc:
            last_exc = exc
            _LOG.warning("ollama chat 실패(재시도 예정): %s", exc)
    raise last_exc
