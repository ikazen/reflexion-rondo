"""config/settings.py 회귀 테스트."""
from __future__ import annotations

import importlib


def test_promote_confirm_seeds_default_excludes_42(monkeypatch):
    """BON-247: 메인 CV seed(42)가 confirm seed 목록에 있으면 confirm이 자명하게 통과해
    실질 독립 확인 개수가 줄어든다 — 기본값에서 42를 제외해야 한다."""
    monkeypatch.delenv("PROMOTE_CONFIRM_SEEDS", raising=False)
    from config import settings
    importlib.reload(settings)
    assert 42 not in settings.PROMOTE_CONFIRM_SEEDS


def test_promote_confirm_seeds_env_override_respected(monkeypatch):
    monkeypatch.setenv("PROMOTE_CONFIRM_SEEDS", "1,2,3")
    from config import settings
    importlib.reload(settings)
    assert settings.PROMOTE_CONFIRM_SEEDS == [1, 2, 3]
    monkeypatch.delenv("PROMOTE_CONFIRM_SEEDS", raising=False)
    importlib.reload(settings)
