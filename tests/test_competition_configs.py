"""competition config 정합성 검사."""
from __future__ import annotations

import importlib
import pkgutil

import config.competitions as _comp_pkg
from config.settings import is_classification


def _load_all_comps():
    mods = []
    for info in pkgutil.iter_modules(_comp_pkg.__path__):
        if info.name.startswith("_"):
            continue
        mods.append(importlib.import_module(f"config.competitions.{info.name}"))
    return mods


def test_is_classification_consistent_with_task_type():
    for mod in _load_all_comps():
        derived = is_classification(mod.TASK_TYPE)
        assert derived == mod.IS_CLASSIFICATION, (
            f"{mod.__name__}: IS_CLASSIFICATION={mod.IS_CLASSIFICATION} "
            f"but TASK_TYPE={mod.TASK_TYPE!r} implies {derived}"
        )


def test_active_flag_is_bool_on_every_competition():
    """#227(Milestone v1.6.0 fleet 동결): ACTIVE 오탈자(문자열 "False" 등)는
    Python에서 항상 truthy라 daemon이 동결을 무시하고 계속 큐잉한다 — 타입까지 검사."""
    for mod in _load_all_comps():
        assert isinstance(mod.ACTIVE, bool), f"{mod.__name__}: ACTIVE must be bool, got {mod.ACTIVE!r}"
