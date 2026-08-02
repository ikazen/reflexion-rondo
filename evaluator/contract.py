"""Coder contract: AST-level validation for Patch class.

이 검사는 이름 기반 정적 lint다 — 정직한 LLM이 실수로 금지 API를 쓰는 것을 막는
soft guard이며, 보안 경계가 아니다. `getattr(__builtins__, "ope"+"n")(...)`,
dunder 체인(`().__class__.__bases__[0].__subclasses__()`) 등은 우회 가능하다
(회귀 문서화: tests/test_contract.py). 실제 격리 경계는 실행 샌드박스
(runtime/isolate.py)가 담당한다.
"""
from __future__ import annotations

import ast
import builtins

_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "socket", "urllib", "urllib2", "urllib3",
    "requests", "httpx", "aiohttp", "http", "ftplib", "smtplib",
    "paramiko", "pickle", "marshal", "ctypes", "cffi",
    "pandas", "importlib",
})

_FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "open", "compile", "__import__",
})

# pandas 관용구 혼동으로 실제 발생한 AttributeError를
# polars 1.41.2 실물 DataFrame/Series에 hasattr로 직접 대조해 확정한 목록 — 전부
# DataFrame/Series 어디에도 존재하지 않아 오탐 없음. `value_counts`는 polars Series에
# 실존하므로(DataFrame에는 없음) 제외했다 — 넣으면 정당한 호출을 오탐으로 거부한다.
_PANDAS_ONLY_ATTRS = frozenset({
    "groupby", "map_dict", "take", "apply", "iterrows", "applymap", "get_dummies",
})

_ALL_HOOKS = frozenset({
    "preprocess", "feature_transform", "param_candidates",
    "build_model", "postprocess_predictions",
    # ensemble_spec — 선언형 앙상블. build_model 대신 "무엇을 조합할지"만
    # 반환하면 harness가 생성·적합·결합을 전담한다(evaluator/harness.py 참고).
    # ensemble/bootstrap에 자동 포함(둘 다 _ALL_HOOKS를 그대로 씀); 단일 변경
    # 원칙(ADR-006)이 적용되는 4개 제한 action_type에는 포함 안 됨.
    "ensemble_spec",
})

_ALLOWED_HOOKS: dict[str, frozenset[str]] = {
    "feature_engineering": frozenset({"feature_transform"}),
    "model_swap":          frozenset({"build_model"}),
    "preprocessing":       frozenset({"preprocess"}),
    "hyperparam_search":   frozenset({"param_candidates"}),
    "ensemble":            _ALL_HOOKS,
    "bootstrap":           _ALL_HOOKS,
}

_HOOK_ARITY = {
    "preprocess":               5,
    "feature_transform":        5,
    "param_candidates":         2,
    "build_model":              3,
    "postprocess_predictions":  3,
    "ensemble_spec":            2,
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


# candidate patch 자체의 undefined-name 검사. cycle/materialize.py의 동명 로직과
# 별도 구현 — runtime/runner.py가 candidate patch를 base와 완전히 분리된 빈
# namespace에 exec하는 것과 정확히 같은 이름 해석 범위(자기 자신의 import/top-level
# helper + builtins + self/ctx)로 검사해, patch 혼자 실행됐을 때 NameError가 날지
# 정적으로 미리 잡는다.
_SAFE_NAMES = frozenset(dir(builtins)) | {"self", "cls", "__class__"}


def _module_level_names(tree: ast.Module) -> set[str] | None:
    """모듈 최상단에서 해석 가능한 이름 전부 (helper 정의 + import 바인딩).

    star import(`from x import *`)가 있으면 무엇을 바인딩하는지 알 수 없으므로
    None을 반환해 호출자가 undefined-name 검사 자체를 건너뛰게 한다 — 미탐지가
    오탐(유효한 패치를 잘못 거부)보다 안전하다.
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
            # try/except, if, with 등 top-level 복합문(optional-dependency 가드 등) 내부에서
            # 조건부로 바인딩되는 이름도 모듈 스코프로 인정한다.
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
    """서브트리 내에서 지역적으로 바인딩되는 모든 이름 (과대추정 — 오탐 방지 우선)."""
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


def _undefined_names_in_patch(tree: ast.Module, patch_cls: ast.ClassDef) -> list[tuple[str, list[str]]]:
    """Patch 메서드가 참조하지만 자신의 소스 어디에도 정의되지 않은 이름 목록.

    (method_name, [unresolved_name, ...]) 쌍의 리스트. star import면 빈 리스트(스킵).
    """
    module_names = _module_level_names(tree)
    if module_names is None:
        return []
    resolvable = module_names | _SAFE_NAMES

    broken: list[tuple[str, list[str]]] = []
    for item in patch_cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        loaded = _collect_loaded_names(item)
        bound = _collect_bound_names(item)
        unresolved = sorted(loaded - bound - resolvable)
        if unresolved:
            broken.append((item.name, unresolved))
    return broken


_VALID_TARGET_READ_ATTRS = frozenset({"get_column", "select"})


def _preprocess_reads_valid_target(item: ast.FunctionDef) -> bool:
    """preprocess(self, train, valid, target, ctx) 안에서 valid[target] 스타일로
    검증 split의 타깃 컬럼을 직접 읽는지 정적으로 확인.

    이름 기반 lint다(파일 상단 docstring 참고) — 실제 판정은
    evaluator.harness._check_preprocess_target_leak의 런타임 동등성 검사가 담당하고,
    이건 흔한 패턴을 값싸게 미리 걸러 재생성 왕복을 줄이는 보조 장치. valid/target
    파라미터명은 시그니처 위치(arity 검사로 이미 5개 고정)로 가져온다.
    """
    args = item.args.args
    if len(args) < 4:
        return False
    valid_param, target_param = args[2].arg, args[3].arg

    for node in ast.walk(item):
        if isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Name) and node.value.id == valid_param
                and isinstance(node.slice, ast.Name) and node.slice.id == target_param
            ):
                return True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name) and node.func.value.id == valid_param
                and node.func.attr in _VALID_TARGET_READ_ATTRS
                and any(isinstance(a, ast.Name) and a.id == target_param for a in node.args)
            ):
                return True
    return False


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
        elif call in _PANDAS_ONLY_ATTRS:
            errors.append(
                f"pandas-only API (not on polars — use group_by/replace_strict/"
                f"map_elements/gather/etc): {call}()"
            )

    patch_cls = _find_patch_class(tree)
    if patch_cls is None:
        errors.append("missing class definition: Patch")
        return errors

    for method_name, unresolved in _undefined_names_in_patch(tree, patch_cls):
        errors.append(f"Patch.{method_name}: references undefined name(s): {unresolved}")

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
        if item.name == "preprocess" and _preprocess_reads_valid_target(item):
            errors.append(
                "Patch.preprocess: reads the validation split's target column directly "
                "(valid[target] or similar) — this is target leakage even if used to "
                "derive a feature (e.g. quantile bins), not just raw passthrough"
            )

    actual_at: str | None = None
    for item in patch_cls.body:
        if isinstance(item, ast.Assign):
            for tgt in item.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "action_type":
                    if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                        actual_at = item.value.value
    if actual_at != action_type:
        errors.append(f"Patch.action_type={actual_at!r}, expected {action_type!r}")

    allowed = _ALLOWED_HOOKS.get(action_type, _ALL_HOOKS)
    disallowed = hook_methods - allowed
    if disallowed:
        errors.append(
            f"action_type={action_type!r} may not implement hooks: {sorted(disallowed)}"
        )

    return errors
