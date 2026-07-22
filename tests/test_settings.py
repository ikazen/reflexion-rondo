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


def test_model_reflector_think_defaults_to_false(monkeypatch):
    """#51: kimi-k2.6 hidden thinking이 num_predict 예산을 잠식하는 문제 — 기본은 비활성화."""
    monkeypatch.delenv("MODEL_REFLECTOR_THINK", raising=False)
    from config import settings
    importlib.reload(settings)
    assert settings.MODEL_REFLECTOR_THINK is False


def test_model_reflector_think_parses_bool_strings(monkeypatch):
    from config import settings

    monkeypatch.setenv("MODEL_REFLECTOR_THINK", "true")
    importlib.reload(settings)
    assert settings.MODEL_REFLECTOR_THINK is True

    monkeypatch.setenv("MODEL_REFLECTOR_THINK", "0")
    importlib.reload(settings)
    assert settings.MODEL_REFLECTOR_THINK is False

    monkeypatch.delenv("MODEL_REFLECTOR_THINK", raising=False)
    importlib.reload(settings)


def test_model_reflector_think_passes_through_effort_level(monkeypatch):
    """bool 문자열이 아니면(low/medium/high) 그대로 통과."""
    from config import settings

    monkeypatch.setenv("MODEL_REFLECTOR_THINK", "low")
    importlib.reload(settings)
    assert settings.MODEL_REFLECTOR_THINK == "low"

    monkeypatch.delenv("MODEL_REFLECTOR_THINK", raising=False)
    importlib.reload(settings)
