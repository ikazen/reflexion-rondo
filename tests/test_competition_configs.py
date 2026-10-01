"""competition config 정합성 검사."""
from __future__ import annotations

import importlib
import pkgutil

import config.competitions as _comp_pkg
from config.competitions import attempt_cpu_budget_secs
from config.settings import is_classification


def _load_all_comps():
    mods = []
    for info in pkgutil.iter_modules(_comp_pkg.__path__):
        if info.name.startswith("_"):
            continue
        mods.append(importlib.import_module(f"config.competitions.{info.name}"))
    return mods


_REQUIRED_CONSTANTS = (
    "COMPETITION_ID", "NAME", "TARGET", "METRIC", "TASK_TYPE", "METRIC_SIGN", "IS_CLASSIFICATION",
    "DROP_COLS", "DATA_DIR", "ACTIVE", "EDA_CARD",
)


def test_required_constants_are_declared_on_every_competition():
    """코드가 getattr 기본값 없이 직접 읽는 상수들 — 하나라도 빠지면 daemon이 런타임에 AttributeError로 죽는다."""
    for mod in _load_all_comps():
        missing = [c for c in _REQUIRED_CONSTANTS if not hasattr(mod, c)]
        assert not missing, f"{mod.__name__}: missing {missing}"


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


def test_attempt_cpu_budget_prefers_the_attempt_specific_value():
    class _Comp:
        ATTEMPT_CPU_BUDGET_SECS = 2400
        CPU_BUDGET_SECS = 3600

    assert attempt_cpu_budget_secs(_Comp) == 2400


def test_attempt_cpu_budget_falls_back_to_the_shared_budget_then_none():
    class _Shared:
        CPU_BUDGET_SECS = 5400

    assert attempt_cpu_budget_secs(_Shared) == 5400
    assert attempt_cpu_budget_secs(object()) is None


def test_s5e4_lowers_only_the_attempt_budget():
    """#421/ADR-063: confirm/holdout/merge-verify가 쓰는 CPU_BUDGET_SECS는 건드리지 않는다."""
    from config.competitions import s5e4

    assert s5e4.ATTEMPT_CPU_BUDGET_SECS == 2400
    assert not hasattr(s5e4, "CPU_BUDGET_SECS")
    assert attempt_cpu_budget_secs(s5e4) == 2400
