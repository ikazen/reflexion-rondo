from __future__ import annotations

import ast
import builtins
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


def _extract_other_toplevel_statements(source: str) -> list[str]:
    """Import/named-helper(class/def/assign) 외의 top-level 문을 unparse해서 수집한다.

    `try: import catboost; CATBOOST_AVAILABLE = True except ImportError: ...` 같은
    optional-dependency 가드 패턴은 named helper로 분류되지 않아 (`ast.Try`는
    `_extract_toplevel_helpers`의 어떤 isinstance 분기에도 안 걸림) 조용히 드롭되던
    문제 — union으로 보존한다. 이름 기반 override 개념이 없으므로 텍스트
    dedup만 하고(단순 재현), base와 patch 양쪽 모두 유지한다.
    """
    tree = ast.parse(source)
    seen: set[str] = set()
    stmts: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ClassDef) and node.name == "Patch":
            continue
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign)):
            continue  # _extract_toplevel_helpers가 처리
        line = ast.unparse(node)
        if line not in seen:
            seen.add(line)
            stmts.append(line)
    return stmts


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
    # 드러나고 엉뚱한 교훈으로 귀속될 수 있다. 에러로 막지 않고 경고만 남긴다
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

    seen_other: set[str] = set()
    other_stmts: list[str] = []
    for line in (
        (_extract_other_toplevel_statements(base_source) if base_source else [])
        + _extract_other_toplevel_statements(patch_source)
    ):
        if line not in seen_other:
            seen_other.add(line)
            other_stmts.append(line)

    parts: list[str] = imports + ["", ""] + other_stmts + (["", ""] if other_stmts else [])

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


_SAFE_NAMES = frozenset(dir(builtins)) | {"self", "cls", "__class__"}


def _module_level_names(tree: ast.Module) -> set[str] | None:
    """모듈 최상단에서 해석 가능한 이름 전부 (helper 정의 + import 바인딩).

    star import(`from x import *`)가 있으면 무엇을 바인딩하는지 알 수 없으므로
    None을 반환해 호출자가 undefined-name 검사 자체를 건너뛰게 한다 — 미탐지가
    오탐(유효한 파이프라인을 잘못 거부)보다 안전하다.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif not isinstance(node, (ast.Import, ast.ImportFrom)):
            # try/except, if, with 등 top-level 복합문(예: optional-dependency 가드
            # `try: import x; FLAG=True except ImportError: FLAG=False`) 내부에서
            # 조건부로 바인딩되는 이름도 모듈 스코프로 인정한다 — _validate_materialized가
            # 이제 이런 문을 verbatim 보존하므로 오탐하면 안 된다.
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    names.add(n.id)
                elif isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(n.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    return None
                names.add(alias.asname or alias.name)
    return names


def _collect_bound_names(node: ast.AST) -> set[str]:
    """서브트리 내에서 지역적으로 바인딩되는 모든 이름 (과대추정 — 오탐 방지 우선).

    컴프리헨션 스코프 격리는 무시하고 Store 컨텍스트 Name을 전부 지역 바인딩으로
    친다. 매개변수·중첩 함수/클래스 이름·for/with/except 타겟·global/nonlocal
    선언명도 포함.
    """
    bound: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = n.args
            for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                bound.add(a.arg)
            if args.vararg:
                bound.add(args.vararg.arg)
            if args.kwarg:
                bound.add(args.kwarg.arg)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(n.name)
        elif isinstance(n, ast.ClassDef):
            bound.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound.update(n.names)
    return bound


def _collect_loaded_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _check_undefined_names(tree: ast.Module) -> None:
    """병합된 class Patch 메서드가 어디에도 정의되지 않은 이름을 참조하면 raise.

    위 helper collision 경고와 달리 이건 실제로 깨진 파이프라인(예: 삭제된
    top-level helper 클래스를 build_model이 여전히 호출)이라 경고가 아니라 에러로
    막는다 — 손상된 best_pipeline이 조용히 업로드되면 안 되기 때문.

    top-level helper 본문은 검사하지 않는다(멤버 스코프만으로 이 버그 클래스를
    잡기에 충분하고, helper 내부까지 보면 오탐 표면만 늘어난다 — under-detection은
    의도적으로 허용).
    """
    module_names = _module_level_names(tree)
    if module_names is None:
        return  # star import — 무엇이 바인딩되는지 알 수 없어 검사 스킵
    resolvable = module_names | _SAFE_NAMES

    patch_cls = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Patch"),
        None,
    )
    if patch_cls is None:
        return  # missing-Patch는 별도 가드가 처리

    broken: list[tuple[str, list[str]]] = []
    for item in patch_cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        loaded = _collect_loaded_names(item)
        bound = _collect_bound_names(item)
        unresolved = sorted(loaded - bound - resolvable)
        if unresolved:
            broken.append((item.name, unresolved))

    if broken:
        detail = "; ".join(f"{name}: {unresolved}" for name, unresolved in broken)
        raise ValueError(f"materialized pipeline references undefined name(s) — {detail}")


def _validate_materialized(source: str) -> None:
    """Compile-check + Patch presence guard + undefined-name guard.

    Raises ValueError on invalid merged output.
    """
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
    _check_undefined_names(tree)
