"""Coder contract: AST-level validation for Patch class."""
from __future__ import annotations

import ast

_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "socket", "urllib", "urllib2", "urllib3",
    "requests", "httpx", "aiohttp", "http", "ftplib", "smtplib",
    "paramiko", "pickle", "marshal", "ctypes", "cffi",
})

_FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "open", "compile", "__import__",
})

_ALL_HOOKS = frozenset({
    "preprocess", "feature_transform", "param_candidates",
    "build_model", "postprocess_predictions",
})

_ALLOWED_HOOKS: dict[str, frozenset[str]] = {
    "feature_engineering": frozenset({"feature_transform"}),
    "model_swap":          frozenset({"build_model"}),
    "preprocessing":       frozenset({"preprocess"}),
    "hyperparam_search":   frozenset({"param_candidates"}),
    "compound":            _ALL_HOOKS,
    "bootstrap":           _ALL_HOOKS,
}

_HOOK_ARITY = {
    "preprocess":               5,
    "feature_transform":        5,
    "param_candidates":         2,
    "build_model":              3,
    "postprocess_predictions":  3,
}


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


def _find_patch_class(tree: ast.AST) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Patch":
            return node
    return None


def validate_patch(source: str, action_type: str) -> list[str]:
    """AST-level validation for Patch class. Returns list of error strings (empty = OK)."""
    errors: list[str] = []

    try:
        tree = ast.parse(source)
        # ast.parse passes return/break/yield-outside-context and duplicate args;
        # compile() catches those at the bytecode level.
        compile(source, "<patch>", "exec")
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    for imp in _collect_imports(tree):
        if imp in _FORBIDDEN_IMPORTS:
            errors.append(f"forbidden import: {imp}")
    for call in _collect_calls(tree):
        if call in _FORBIDDEN_CALLS:
            errors.append(f"forbidden call: {call}()")

    patch_cls = _find_patch_class(tree)
    if patch_cls is None:
        errors.append("missing class definition: Patch")
        return errors

    # Collect hook method names and check arity
    hook_methods: set[str] = set()
    for item in patch_cls.body:
        if not isinstance(item, ast.FunctionDef) or item.name not in _ALL_HOOKS:
            continue
        hook_methods.add(item.name)
        expected = _HOOK_ARITY[item.name]
        actual = len(item.args.args)
        if actual != expected:
            errors.append(
                f"Patch.{item.name}: expected {expected} args (incl. self), got {actual}"
            )

    # action_type class attribute must match
    actual_at: str | None = None
    for item in patch_cls.body:
        if isinstance(item, ast.Assign):
            for tgt in item.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "action_type":
                    if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                        actual_at = item.value.value
    if actual_at != action_type:
        errors.append(f"Patch.action_type={actual_at!r}, expected {action_type!r}")

    # No disallowed hooks
    allowed = _ALLOWED_HOOKS.get(action_type, _ALL_HOOKS)
    disallowed = hook_methods - allowed
    if disallowed:
        errors.append(
            f"action_type={action_type!r} may not implement hooks: {sorted(disallowed)}"
        )

    if action_type == "compound" and len(hook_methods) > 2:
        errors.append(
            f"compound may implement at most 2 hooks, got {len(hook_methods)}: {sorted(hook_methods)}"
        )

    return errors
