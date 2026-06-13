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
