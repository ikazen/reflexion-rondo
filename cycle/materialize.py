from __future__ import annotations

import ast
import textwrap

_HOOKS = frozenset({
    "preprocess", "feature_transform", "param_candidates",
    "build_model", "postprocess_predictions",
})


def _extract_hooks(source: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Patch":
            return {
                item.name: item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name in _HOOKS
            }
    return {}


def _extract_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    seen: set[str] = set()
    lines: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            line = ast.unparse(node)
            if line not in seen:
                seen.add(line)
                lines.append(line)
    return lines


def materialize_best_pipeline(base_source: str | None, patch_source: str) -> str:
    """Merge patch into base, returning a new materialized Patch source.

    Hooks from base are preserved; patch hooks override base hooks.
    """
    base_hooks = _extract_hooks(base_source) if base_source else {}
    patch_hooks = _extract_hooks(patch_source)
    merged: dict[str, ast.FunctionDef] = {**base_hooks, **patch_hooks}

    seen: set[str] = set()
    imports: list[str] = []
    for line in (_extract_imports(base_source) if base_source else []) + _extract_imports(patch_source):
        if line not in seen:
            seen.add(line)
            imports.append(line)

    parts: list[str] = imports + [
        "",
        "",
        "class Patch:",
        '    action_type = "materialized"',
        '    changed_stages = []',
        '    rationale = "Accumulated materialized pipeline"',
        "",
    ]
    for fn_node in merged.values():
        fn_src = ast.unparse(fn_node)
        parts.append(textwrap.indent(fn_src, "    "))
        parts.append("")

    return "\n".join(parts)
