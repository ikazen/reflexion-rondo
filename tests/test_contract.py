from __future__ import annotations

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
    """BON-192: importlib.import_module('os')로 _FORBIDDEN_IMPORTS 우회하던 경로를 막는다."""
    source = "import importlib\n" + _VALID_FEATURE_ENG
    errs = validate_patch(source, "feature_engineering")
    assert any("forbidden import" in e and "importlib" in e for e in errs)


def test_getattr_string_concat_bypass_not_caught():
    """soft guard 한계 문서화 (BON-192): _collect_calls는 ast.Name/Attribute만 보므로
    getattr(...)(...) 호출 형태는 forbidden call로 잡히지 않는다. 진짜 경계는 실행
    샌드박스(BON-191)이며, 이 테스트는 lint의 한계를 회귀 기준으로 고정한다."""
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
    """soft guard 한계 문서화 (BON-192): dunder 체인으로 임의 클래스 접근은
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
