from __future__ import annotations

import pytest

from evaluator.contract import validate_patch

_VALID_FEATURE_ENG = """
import polars as pl

class Patch:
    action_type = "feature_engineering"
    changed_stages = ["feature_transform"]
    rationale = "Add interaction feature."

    def feature_transform(self, train, valid, target, ctx):
        cols = [c for c in train.columns if c != target]
        return train.select(cols), valid.select(cols)
""".strip()


def test_valid_patch_returns_no_errors():
    assert validate_patch(_VALID_FEATURE_ENG, "feature_engineering") == []


def test_parse_level_syntax_error_caught():
    source = "def f(:\n    pass"
    errs = validate_patch(source, "feature_engineering")
    assert len(errs) == 1
    assert errs[0].startswith("SyntaxError")


def test_return_outside_function_caught():
    """ast.parse passes this; compile() must catch it."""
    source = "x = 1\nreturn x"
    errs = validate_patch(source, "feature_engineering")
    assert len(errs) == 1
    assert errs[0].startswith("SyntaxError")


def test_break_outside_loop_caught():
    """ast.parse passes this; compile() must catch it."""
    source = _VALID_FEATURE_ENG + "\nbreak\n"
    errs = validate_patch(source, "feature_engineering")
    assert len(errs) == 1
    assert errs[0].startswith("SyntaxError")


def test_duplicate_arg_caught():
    """ast.parse passes duplicate args; compile() catches them."""
    source = "class Patch:\n    action_type = 'feature_engineering'\n    def feature_transform(self, a, a, b, c):\n        pass\n"
    errs = validate_patch(source, "feature_engineering")
    assert any(e.startswith("SyntaxError") for e in errs)


def test_forbidden_import_still_caught():
    source = "import os\n" + _VALID_FEATURE_ENG
    errs = validate_patch(source, "feature_engineering")
    assert any("forbidden import" in e for e in errs)


def test_action_type_mismatch_still_caught():
    errs = validate_patch(_VALID_FEATURE_ENG, "model_swap")
    assert any("action_type" in e for e in errs)


def test_pandas_import_forbidden():
    source = "import pandas as pd\n" + _VALID_FEATURE_ENG
    errs = validate_patch(source, "feature_engineering")
    assert any("forbidden import" in e and "pandas" in e for e in errs)


def test_importlib_import_now_forbidden():
    """importlib.import_module('os')로 _FORBIDDEN_IMPORTS 우회하던 경로를 막는다."""
    source = "import importlib\n" + _VALID_FEATURE_ENG
    errs = validate_patch(source, "feature_engineering")
    assert any("forbidden import" in e and "importlib" in e for e in errs)


def test_getattr_string_concat_bypass_not_caught():
    """soft guard 한계 문서화: _collect_calls는 ast.Name/Attribute만 보므로
    getattr(...)(...) 호출 형태는 forbidden call로 잡히지 않는다. 진짜 경계는 실행
    샌드박스이며, 이 테스트는 lint의 한계를 회귀 기준으로 고정한다."""
    source = (
        'class Patch:\n'
        '    action_type = "feature_engineering"\n'
        '    changed_stages = ["feature_transform"]\n'
        '    rationale = "bypass demo"\n'
        '    def feature_transform(self, train, valid, target, ctx):\n'
        '        getattr(__builtins__, "ope" + "n")("/etc/passwd")\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "feature_engineering")
    assert not any("forbidden call" in e for e in errs)


def test_dunder_subclass_chain_bypass_not_caught():
    """soft guard 한계 문서화: dunder 체인으로 임의 클래스 접근은
    이름 기반 검사로 탐지되지 않는다. eval/exec/open 등 직접 호출만 잡힘."""
    source = (
        'class Patch:\n'
        '    action_type = "feature_engineering"\n'
        '    changed_stages = ["feature_transform"]\n'
        '    rationale = "bypass demo"\n'
        '    def feature_transform(self, train, valid, target, ctx):\n'
        '        subclasses = ().__class__.__bases__[0].__subclasses__()\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "feature_engineering")
    assert not any("forbidden call" in e or "forbidden import" in e for e in errs)


def test_allowed_hooks_covers_all_action_types():
    from config.settings import ACTION_TYPES
    from evaluator.contract import _ALLOWED_HOOKS
    # Every emittable action_type must have a contract entry.
    # bootstrap is valid but not bandit-emitted (special stage).
    non_bandit = {"bootstrap"}
    bandit_keys = set(_ALLOWED_HOOKS) - non_bandit
    assert bandit_keys == set(ACTION_TYPES), (
        f"_ALLOWED_HOOKS bandit keys {bandit_keys} != ACTION_TYPES {set(ACTION_TYPES)}"
    )


# --- pandas-only API 정적 금지 ---

@pytest.mark.parametrize("attr", [
    "groupby", "map_dict", "take", "apply", "iterrows", "applymap", "get_dummies",
])
def test_pandas_only_attr_forbidden(attr: str):
    """polars 1.41.2에 hasattr로 직접 확인한 순수 pandas 관용구 — 실제 최근 3일 AttributeError
    상위 원인. DataFrame/Series 어디에도 없어 오탐 없이 금지 가능하다."""
    source = (
        'class Patch:\n'
        '    action_type = "feature_engineering"\n'
        '    changed_stages = ["feature_transform"]\n'
        '    rationale = "pandas confusion demo"\n'
        '    def feature_transform(self, train, valid, target, ctx):\n'
        f'        train.{attr}("x")\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "feature_engineering")
    assert any("pandas-only API" in e and attr in e for e in errs)


def test_value_counts_not_forbidden():
    """value_counts는 polars Series에 실존한다(DataFrame에는 없음) — 금지 목록에서
    의도적으로 제외했으므로 오탐이 없어야 한다(회귀 고정)."""
    source = (
        'class Patch:\n'
        '    action_type = "feature_engineering"\n'
        '    changed_stages = ["feature_transform"]\n'
        '    rationale = "value_counts is legit on Series"\n'
        '    def feature_transform(self, train, valid, target, ctx):\n'
        '        counts = train["target"].value_counts()\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "feature_engineering")
    assert not any("pandas-only API" in e for e in errs)


# --- candidate patch 자체의 undefined-name 검사 ---

def test_undefined_name_in_hook_caught():
    """hook 안에서 자기 소스 어디에도 정의되지 않은 이름을 참조하면 에러.

    실제 사고(WeightedEnsemble)와 같은 클래스의 버그를 candidate patch 자신이
    저지른 경우(예: ensemble action이 helper 클래스를 참조만 하고 정의를 빼먹음)를
    재현한다."""
    source = (
        'class Patch:\n'
        '    action_type = "model_swap"\n'
        '    changed_stages = ["build_model"]\n'
        '    rationale = "forgot to define helper"\n'
        '    def build_model(self, params, ctx):\n'
        '        return WeightedEnsemble(params)\n'
    )
    errs = validate_patch(source, "model_swap")
    assert any("undefined name" in e and "WeightedEnsemble" in e for e in errs)


def test_undefined_name_resolved_via_import_or_toplevel_helper_ok():
    """import된 이름이나 같은 소스의 top-level helper로 해석되면 오탐이 없어야 한다."""
    source = (
        'from sklearn.ensemble import RandomForestClassifier\n'
        '\n'
        'def _make_params():\n'
        '    return {"n_estimators": 100}\n'
        '\n'
        'class Patch:\n'
        '    action_type = "model_swap"\n'
        '    changed_stages = ["build_model"]\n'
        '    rationale = "uses import + top-level helper"\n'
        '    def build_model(self, params, ctx):\n'
        '        return RandomForestClassifier(**_make_params())\n'
    )
    errs = validate_patch(source, "model_swap")
    assert not any("undefined name" in e for e in errs)


def test_star_import_skips_undefined_name_check():
    """star import가 있으면 무엇이 바인딩되는지 알 수 없어 미탐지를 택한다
    (cycle/materialize.py의 동일 원칙과 일치, 오탐 방지 우선)."""
    source = (
        'from sklearn.ensemble import *\n'
        '\n'
        'class Patch:\n'
        '    action_type = "model_swap"\n'
        '    changed_stages = ["build_model"]\n'
        '    rationale = "star import hides bindings"\n'
        '    def build_model(self, params, ctx):\n'
        '        return SomeUnknownEstimator(params)\n'
    )
    errs = validate_patch(source, "model_swap")
    assert not any("undefined name" in e for e in errs)
