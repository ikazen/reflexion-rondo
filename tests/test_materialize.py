from __future__ import annotations

import ast
import textwrap

from cycle.materialize import materialize_best_pipeline


_BASE = textwrap.dedent("""
    import polars as pl

    THRESHOLD = 0.5

    def _base_helper(x):
        return x

    class Patch:
        action_type = "preprocessing"
        changed_stages = ["preprocess"]
        rationale = "base"
        base_attr = "base"

        def preprocess(self, train, valid, target, ctx):
            return train, valid

        def feature_transform(self, train, valid, target, ctx):
            cols = [c for c in train.columns if c != target]
            return train.select(cols), valid.select(cols)
""").strip()

_PATCH = textwrap.dedent("""
    import polars as pl
    import numpy as np

    SCALE = 2.0

    def _patch_helper(df):
        return df

    class Patch:
        action_type = "feature_engineering"
        changed_stages = ["feature"]
        rationale = "add scale"
        patch_attr = "patch"

        def feature_transform(self, train, valid, target, ctx):
            return _patch_helper(train), _patch_helper(valid)

        def param_candidates(self, ctx):
            return [{"n_estimators": 100}]
""").strip()


def _method_names(src: str) -> set[str]:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Patch":
            return {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    return set()


def _class_assign_names(src: str) -> set[str]:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Patch":
            names: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for t in item.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
            return names
    return set()


def _toplevel_names(src: str) -> set[str]:
    tree = ast.parse(src)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def test_output_is_valid_python():
    result = materialize_best_pipeline(_BASE, _PATCH)
    ast.parse(result)


def test_patch_overrides_base_hook():
    result = materialize_best_pipeline(_BASE, _PATCH)
    assert "_patch_helper" in result
    assert "cols = [c for c in" not in result


def test_base_only_hook_preserved():
    result = materialize_best_pipeline(_BASE, _PATCH)
    assert "preprocess" in _method_names(result)


def test_patch_adds_new_hook():
    result = materialize_best_pipeline(_BASE, _PATCH)
    methods = _method_names(result)
    assert "preprocess" in methods
    assert "feature_transform" in methods
    assert "param_candidates" in methods


def test_toplevel_helpers_merged():
    result = materialize_best_pipeline(_BASE, _PATCH)
    names = _toplevel_names(result)
    assert "_base_helper" in names
    assert "_patch_helper" in names


def test_toplevel_constants_preserved():
    result = materialize_best_pipeline(_BASE, _PATCH)
    names = _toplevel_names(result)
    assert "THRESHOLD" in names
    assert "SCALE" in names


def test_class_attrs_preserved():
    result = materialize_best_pipeline(_BASE, _PATCH)
    attrs = _class_assign_names(result)
    assert "base_attr" in attrs
    assert "patch_attr" in attrs


def test_meta_attrs_use_materialized_values():
    result = materialize_best_pipeline(_BASE, _PATCH)
    attrs = _class_assign_names(result)
    assert "action_type" in attrs
    assert "rationale" in attrs
    assert 'action_type = "materialized"' in result
    assert 'rationale = "Accumulated materialized pipeline"' in result


def test_imports_deduplicated():
    result = materialize_best_pipeline(_BASE, _PATCH)
    assert result.count("import polars as pl") == 1


def test_none_base_uses_patch_only():
    result = materialize_best_pipeline(None, _PATCH)
    ast.parse(result)
    assert "feature_transform" in _method_names(result)
    assert "_patch_helper" in result


def test_toplevel_helpers_appear_before_class():
    result = materialize_best_pipeline(_BASE, _PATCH)
    helper_pos = result.index("def _patch_helper")
    class_pos = result.index("class Patch:")
    assert helper_pos < class_pos


def test_multi_target_assign_emitted_once():
    patch = textwrap.dedent("""
        import polars as pl

        x = y = 42

        class Patch:
            action_type = "preprocessing"
            changed_stages = []
            rationale = "test"

            def preprocess(self, train, valid, target, ctx):
                return train, valid
    """).strip()
    result = materialize_best_pipeline(None, patch)
    ast.parse(result)
    assert result.count("x = y = 42") == 1
