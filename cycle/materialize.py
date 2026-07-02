from __future__ import annotations

import ast
import logging
import textwrap

logger = logging.getLogger(__name__)

_META_ATTRS = frozenset({"action_type", "changed_stages", "rationale"})


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


def _extract_toplevel_helpers(source: str) -> dict[str, ast.stmt]:
    tree = ast.parse(source)
    result: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ClassDef) and node.name == "Patch":
            continue
        if isinstance(node, ast.ClassDef):
            result[node.name] = node
        elif isinstance(node, ast.FunctionDef):
            result[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                result[node.target.id] = node
    return result


def _extract_class_members(source: str) -> dict[str, ast.stmt]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Patch":
            result: dict[str, ast.stmt] = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    result[item.name] = item
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id not in _META_ATTRS:
                            result[target.id] = item
                elif isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name) and item.target.id not in _META_ATTRS:
                        result[item.target.id] = item
            return result
    return {}


def materialize_best_pipeline(base_source: str | None, patch_source: str) -> str:
    base_helpers = _extract_toplevel_helpers(base_source) if base_source else {}
    patch_helpers = _extract_toplevel_helpers(patch_source)
    # base와 patch가 동명 helper/member를 정의하면 patch가 base를 조용히 덮어쓴다.
    # base의 다른 helper가 그 이름에 의존하면 merge 후 의미가 바뀌어 CV 퇴행으로만
    # 드러나고 엉뚱한 교훈으로 귀속될 수 있다 (BON-197). 에러로 막지 않고 경고만 남긴다
    # — override 자체는 의도적일 수 있어 가시화가 목적.
    helper_collisions = base_helpers.keys() & patch_helpers.keys()
    if helper_collisions:
        logger.warning("helper name collision (patch overrides base): %s", sorted(helper_collisions))
    merged_helpers: dict[str, ast.stmt] = {**base_helpers, **patch_helpers}

    base_members = _extract_class_members(base_source) if base_source else {}
    patch_members = _extract_class_members(patch_source)
    member_collisions = base_members.keys() & patch_members.keys()
    if member_collisions:
        logger.warning("Patch member collision (patch overrides base): %s", sorted(member_collisions))
    merged_members: dict[str, ast.stmt] = {**base_members, **patch_members}

    seen_imports: set[str] = set()
    imports: list[str] = []
    for line in (_extract_imports(base_source) if base_source else []) + _extract_imports(patch_source):
        if line not in seen_imports:
            seen_imports.add(line)
            imports.append(line)

    parts: list[str] = imports + ["", ""]

    seen_nodes: set[int] = set()
    for node in merged_helpers.values():
        nid = id(node)
        if nid in seen_nodes:
            continue
        seen_nodes.add(nid)
        parts.append(ast.unparse(node))
        parts.append("")

    parts += [
        "",
        "class Patch:",
        '    action_type = "materialized"',
        '    changed_stages = []',
        '    rationale = "Accumulated materialized pipeline"',
        "",
    ]

    seen_nodes = set()
    for member in merged_members.values():
        nid = id(member)
        if nid in seen_nodes:
            continue
        seen_nodes.add(nid)
        parts.append(textwrap.indent(ast.unparse(member), "    "))
        parts.append("")

    result = "\n".join(parts)
    _validate_materialized(result)
    return result


def _validate_materialized(source: str) -> None:
    """Compile-check + Patch presence guard. Raises ValueError on invalid merged output."""
    try:
        compile(source, "<materialized>", "exec")
    except SyntaxError as exc:
        raise ValueError(f"materialized pipeline has SyntaxError: {exc}") from exc
    tree = ast.parse(source)
    has_patch = any(
        isinstance(node, ast.ClassDef) and node.name == "Patch"
        for node in ast.walk(tree)
    )
    if not has_patch:
        raise ValueError("materialized pipeline is missing class Patch")
