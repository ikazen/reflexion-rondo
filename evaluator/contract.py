"""Coder contract: type aliases and pre-execution validation gate.

Two functions the Coder must produce:
  feature_fn(train, valid, target) -> (train_feats, valid_feats)
  model_fn(params) -> sklearn-compatible estimator

validate_code() does AST-level static checks before any execution.
smoke_test() does a tiny runtime check (10 rows) before the full CV loop.
"""
import ast
import inspect
import traceback
from typing import Callable

import polars as pl

FeatureFn = Callable[[pl.DataFrame, pl.DataFrame, str], tuple[pl.DataFrame, pl.DataFrame]]
ModelFn = Callable[[dict], object]

_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "socket", "urllib", "urllib2", "urllib3",
    "requests", "httpx", "aiohttp", "http", "ftplib", "smtplib",
    "paramiko", "pickle", "marshal", "ctypes", "cffi",
})

_FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "open", "compile", "__import__",
})


def _collect_imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
    return names


def _collect_calls(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return names


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _check_signature(fn_node: ast.FunctionDef, expected_arity: int) -> str | None:
    args = fn_node.args
    n = len(args.args)
    if n != expected_arity:
        return f"{fn_node.name}: expected {expected_arity} positional args, got {n}"
    return None


def validate_code(source: str) -> list[str]:
    """AST-level static validation. Returns list of error strings (empty = OK)."""
    errors: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    # required function definitions
    feature_node = _find_function(tree, "feature_fn")
    model_node = _find_function(tree, "model_fn")
    if feature_node is None:
        errors.append("missing function definition: feature_fn")
    if model_node is None:
        errors.append("missing function definition: model_fn")

    # signature arity
    if feature_node is not None:
        err = _check_signature(feature_node, 3)
        if err:
            errors.append(err)
    if model_node is not None:
        err = _check_signature(model_node, 1)
        if err:
            errors.append(err)

    # forbidden imports
    for imp in _collect_imports(tree):
        if imp in _FORBIDDEN_IMPORTS:
            errors.append(f"forbidden import: {imp}")

    # forbidden calls
    for call in _collect_calls(tree):
        if call in _FORBIDDEN_CALLS:
            errors.append(f"forbidden call: {call}()")

    return errors


def smoke_test(
    feature_fn: FeatureFn,
    model_fn: ModelFn,
    sample: pl.DataFrame,
    target: str,
) -> str | None:
    """Runtime check on a tiny sample. Returns error string or None."""
    probe = sample.head(10)
    try:
        result = feature_fn(probe, probe, target)
    except Exception:
        return f"feature_fn raised on smoke sample:\n{traceback.format_exc()}"

    if not (isinstance(result, tuple) and len(result) == 2):
        return f"feature_fn must return tuple[DataFrame, DataFrame], got {type(result)}"
    for i, df in enumerate(result):
        if not isinstance(df, pl.DataFrame):
            return f"feature_fn return[{i}] is {type(df)}, expected polars.DataFrame"

    try:
        model = model_fn({})
    except Exception:
        return f"model_fn raised on smoke sample:\n{traceback.format_exc()}"

    if not (hasattr(model, "fit") and hasattr(model, "predict")):
        return "model_fn return value must have fit() and predict() methods"

    return None
