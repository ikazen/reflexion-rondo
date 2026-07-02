from __future__ import annotations

import ast
import textwrap

import pytest
from cycle.materialize import materialize_best_pipeline, _validate_materialized


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


# --- _validate_materialized ---

def test_validate_materialized_valid_source():
    _validate_materialized(_PATCH)  # should not raise


def test_validate_materialized_syntax_error():
    bad = "class Patch:\n    def broken(self\n"
    with pytest.raises(ValueError, match="SyntaxError"):
        _validate_materialized(bad)


def test_validate_materialized_missing_patch_class():
    no_patch = "x = 1\ndef helper(): pass\n"
    with pytest.raises(ValueError, match="missing class Patch"):
        _validate_materialized(no_patch)


def test_materialize_invalid_merge_raises():
    """syntactically broken base causes materialize to raise via _validate_materialized."""
    broken_base = "class Patch:\n    def preprocess(self\n"
    with pytest.raises(Exception):
        materialize_best_pipeline(broken_base, _PATCH)


# --- undefined-name guard (BON-233) ---
# 재현: s4e10 best_pipeline.py에서 build_model이 WeightedEnsemble(...)을 호출하는데
# 그 클래스 정의가 파일 어디에도 없어 NameError가 535회 반복 발생했다.

def test_undefined_name_dangling_reference_raises():
    broken = textwrap.dedent("""
        class Patch:
            action_type = "materialized"
            changed_stages = []
            rationale = "test"

            def build_model(self, params, ctx):
                return WeightedEnsemble([1, 2], [0.5, 0.5])
    """).strip()
    with pytest.raises(ValueError, match="WeightedEnsemble"):
        _validate_materialized(broken)


def test_undefined_name_builtins_and_params_ok():
    source = textwrap.dedent("""
        class Patch:
            action_type = "materialized"
            changed_stages = []
            rationale = "test"

            def build_model(self, params, ctx):
                return sorted(range(len(params)))
    """).strip()
    _validate_materialized(source)  # should not raise


def test_undefined_name_imported_name_ok():
    source = textwrap.dedent("""
        import numpy as np

        class Patch:
            action_type = "materialized"
            changed_stages = []
            rationale = "test"

            def build_model(self, params, ctx):
                return np.array([1, 2, 3])
    """).strip()
    _validate_materialized(source)  # should not raise


def test_undefined_name_toplevel_helper_ok():
    source = textwrap.dedent("""
        def _helper(x):
            return x

        class Patch:
            action_type = "materialized"
            changed_stages = []
            rationale = "test"

            def build_model(self, params, ctx):
                return _helper(params)
    """).strip()
    _validate_materialized(source)  # should not raise


def test_undefined_name_comprehension_lambda_nested_def_ok():
    """컴프리헨션/람다/중첩 함수의 자기 타겟·파라미터, 외부 지역변수 참조는 오탐이 아니다."""
    source = textwrap.dedent("""
        class Patch:
            action_type = "materialized"
            changed_stages = []
            rationale = "test"

            def build_model(self, params, ctx):
                base = params.get("n", 1)
                squares = [x * base for x in range(base)]
                adder = lambda y: y + base
                def nested(z):
                    return z + base
                return sum(squares) + adder(1) + nested(2)
    """).strip()
    _validate_materialized(source)  # should not raise


def test_undefined_name_in_annotation_raises():
    """annotation도 def 시점에 즉시 평가되므로(future annotations 미사용) 검사 대상이다."""
    source = textwrap.dedent("""
        from typing import List

        class Patch:
            action_type = "materialized"
            changed_stages = []
            rationale = "test"

            def build_model(self, params, ctx) -> List[SomeUndefinedType]:
                return []
    """).strip()
    with pytest.raises(ValueError, match="SomeUndefinedType"):
        _validate_materialized(source)


def test_undefined_name_star_import_skips_check():
    source = textwrap.dedent("""
        from sklearn.ensemble import *

        class Patch:
            action_type = "materialized"
            changed_stages = []
            rationale = "test"

            def build_model(self, params, ctx):
                return TotallyUndefinedThing()
    """).strip()
    _validate_materialized(source)  # star import → 검사 스킵, raise 안 함


def test_undefined_name_for_with_except_walrus_binding_ok():
    source = textwrap.dedent("""
        import contextlib

        class Patch:
            action_type = "materialized"
            changed_stages = []
            rationale = "test"

            def build_model(self, params, ctx):
                total = 0
                for i in range(3):
                    total += i
                with contextlib.suppress(ZeroDivisionError) as suppressed:
                    total += 1
                try:
                    risky = 1 / 0
                except ZeroDivisionError as err:
                    total += 1
                if (n := params.get("n", 0)) > 0:
                    total += n
                return total
    """).strip()
    _validate_materialized(source)  # should not raise


# --- helper/member name collision (BON-197) ---

_BASE_WITH_ENCODE = textwrap.dedent("""
    def _encode(x):
        return x + 1

    class Patch:
        action_type = "preprocessing"
        changed_stages = ["preprocess"]
        rationale = "base"

        def feature_transform(self, train, valid, target, ctx):
            return _encode(train), _encode(valid)
""").strip()

_PATCH_WITH_ENCODE = textwrap.dedent("""
    def _encode(x):
        return x * 2

    class Patch:
        action_type = "feature_engineering"
        changed_stages = ["feature"]
        rationale = "different _encode semantics"

        def param_candidates(self, ctx):
            return [{"n_estimators": 100}]
""").strip()


def test_helper_collision_logs_warning(caplog):
    with caplog.at_level("WARNING", logger="cycle.materialize"):
        result = materialize_best_pipeline(_BASE_WITH_ENCODE, _PATCH_WITH_ENCODE)
    assert any("_encode" in rec.message for rec in caplog.records)
    ast.parse(result)


def test_helper_collision_patch_wins_silently_in_output():
    """merge 결과 자체는 여전히 patch 정의로 override된다 — 경고는 가시화용, 동작은 불변."""
    result = materialize_best_pipeline(_BASE_WITH_ENCODE, _PATCH_WITH_ENCODE)
    assert "x * 2" in result
    assert "x + 1" not in result


def test_no_collision_no_warning(caplog):
    no_collision_base = textwrap.dedent("""
        def _only_in_base(x):
            return x

        class Patch:
            action_type = "preprocessing"
            changed_stages = ["preprocess"]
            rationale = "base"

            def preprocess(self, train, valid, target, ctx):
                return train, valid
    """).strip()
    no_collision_patch = textwrap.dedent("""
        def _only_in_patch(x):
            return x

        class Patch:
            action_type = "feature_engineering"
            changed_stages = ["feature"]
            rationale = "patch"

            def feature_transform(self, train, valid, target, ctx):
                return train, valid
    """).strip()
    with caplog.at_level("WARNING", logger="cycle.materialize"):
        materialize_best_pipeline(no_collision_base, no_collision_patch)
    assert not any("collision" in rec.message for rec in caplog.records)


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
