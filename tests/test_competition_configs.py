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
