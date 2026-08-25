"""승격 winner의 Patch를 현재 best pipeline 소스에 병합(materialize_best_pipeline).

patch가 새로 정의한 hook은 보존하고, base에만 있는 hook도 보존한다. 양쪽이 같은 합성
가능 훅(_COMPOSABLE_HOOKS)을 다르게 정의하면 완전 교체 대신 base 실행 후 patch를
적용하는 wrapper를 합성한다(ADR-037, #232, patch가 override로 명시하면 완전 교체).
그 외 훅(build_model 등)은 여전히 patch가 이긴다. 병합 결과는 undefined-name/
optional-dependency 가드로 검증한다.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import logging
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from store.db import PgConn

logger = logging.getLogger(__name__)

_META_ATTRS = frozenset({"action_type", "changed_stages", "rationale", "override"})

# base 실행 후 patch를 적용해 합성 가능한 훅(ADR-037, #232) — build_model/ensemble_spec/
# model_spec은 "이 모델을 어떻게 만들지"가 단일 값이라 두 구현을 조합할 방법이 없어
# 제외(evaluator/harness.py:PatchedPipeline과 동일 목록, 반드시 동기화 유지).
_COMPOSABLE_HOOKS = frozenset({"preprocess", "feature_transform", "postprocess_predictions", "param_candidates"})

# 합성된 wrapper가 호출하는 harness 헬퍼 — evaluator/harness.py:_union_feature_columns/
# _union_param_candidates와 정확히 같은 조합 규칙을 써야 attempt 평가 시점(PatchedPipeline)과
# 승격 후 materialize 결과가 같은 동작을 재현한다(측정=배포 불일치 방지).
_COMPOSE_HELPER_IMPORTS: dict[str, str] = {
    "feature_transform": "from evaluator.harness import _union_feature_columns",
    "param_candidates": "from evaluator.harness import _union_param_candidates",
}

_COMPOSE_TEMPLATES: dict[str, str] = {
    "preprocess": (
        "def preprocess(self, train, valid, target, ctx):\n"
        "    train, valid = self.{base}(train, valid, target, ctx)\n"
        "    return self.{patch}(train, valid, target, ctx)\n"
    ),
    "feature_transform": (
        "def feature_transform(self, train, valid, target, ctx):\n"
        "    Xtr_base, Xva_base = self.{base}(train, valid, target, ctx)\n"
        "    Xtr_patch, Xva_patch = self.{patch}(train, valid, target, ctx)\n"
        "    return _union_feature_columns(Xtr_base, Xtr_patch), _union_feature_columns(Xva_base, Xva_patch)\n"
    ),
    "postprocess_predictions": (
        "def postprocess_predictions(self, preds, ctx):\n"
        "    preds = self.{base}(preds, ctx)\n"
        "    return self.{patch}(preds, ctx)\n"
    ),
    "param_candidates": (
        "def param_candidates(self, ctx):\n"
        "    return _union_param_candidates(self.{base}(ctx), self.{patch}(ctx))\n"
    ),
}


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
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # ClassDef 포함: ensemble action_type 등이 build_model에서 참조하는
                    # wrapper 클래스를 Patch 안에 중첩 정의하는 패턴이 흔해, 누락하면
                    # 병합본에서만 그 클래스가 사라진다.
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


def _extract_override_hooks(source: str) -> frozenset[str]:
    """Patch.override(선택적 클래스 속성, 문자열 리스트)를 읽는다 — 여기 나열된 훅은
    합성하지 않고 기존처럼 완전 교체한다(ADR-037, #232). 미선언 시 빈 집합(= 이 patch가
    정의하는 모든 합성 가능 훅을 합성 대상으로 삼는다는 뜻)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Patch":
            for item in node.body:
                if not isinstance(item, ast.Assign):
                    continue
                if not any(isinstance(t, ast.Name) and t.id == "override" for t in item.targets):
                    continue
                if isinstance(item.value, (ast.List, ast.Tuple, ast.Set)):
                    return frozenset(
                        elt.value for elt in item.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    )
            return frozenset()
    return frozenset()


def _real_collisions(base_map: dict[str, ast.stmt], patch_map: dict[str, ast.stmt]) -> set[str]:
    """이름은 같지만 실제 정의(unparse 텍스트)가 다른 이름만 진짜 충돌로 본다 — 완전히
    동일한 우연한 재정의는 어느 쪽을 써도 결과가 같아 모호함이 없다."""
    shared = base_map.keys() & patch_map.keys()
    return {name for name in shared if ast.unparse(base_map[name]) != ast.unparse(patch_map[name])}


def _unique_name(base_name: str, existing: set[str]) -> str:
    if base_name not in existing:
        return base_name
    i = 2
    while f"{base_name}_{i}" in existing:
        i += 1
    return f"{base_name}_{i}"


def _synthesize_composed_member(
    hook_name: str, base_node: ast.stmt, patch_node: ast.stmt, existing_names: set[str],
) -> tuple[ast.stmt, ast.stmt, ast.stmt]:
    """같은 이름의 합성 가능 훅(_COMPOSABLE_HOOKS)이 base/patch 양쪽에 있을 때, base
    쪽을 통째로 버리는 대신 base를 실행한 뒤 patch를 적용하는 새 wrapper를 만든다
    (ADR-037, #232) — base/patch 원본은 유일한 이름으로 rename해 helper로 보존한다.
    base_node가 이미 이전 라운드의 합성 wrapper여도(반복 합성) _unique_name이 매번
    새 이름을 골라주므로 그 축적분이 사라지지 않는다."""
    base_name = _unique_name(f"_{hook_name}_prev", existing_names)
    existing_names.add(base_name)
    patch_name = _unique_name(f"_{hook_name}_new", existing_names)
    existing_names.add(patch_name)

    base_node.name = base_name
    patch_node.name = patch_name

    wrapper_src = _COMPOSE_TEMPLATES[hook_name].format(base=base_name, patch=patch_name)
    wrapper_node = ast.parse(wrapper_src).body[0]
    return base_node, patch_node, wrapper_node


def materialize_best_pipeline(base_source: str | None, patch_source: str) -> str:
    base_helpers = _extract_toplevel_helpers(base_source) if base_source else {}
    patch_helpers = _extract_toplevel_helpers(patch_source)
    # base와 patch가 동명 helper를 서로 다른 정의로 선언하면(우연히 완전히 동일한
    # 재정의는 모호함이 없어 제외, _real_collisions) 어느 쪽이 실제로 쓰이는지가
    # 조용히 결정돼 CV 퇴행이 엉뚱한 교훈으로 귀속될 수 있다 — 경고가 아니라 에러로
    # 막는다(ADR-037, #232, 이전엔 warning뿐이라 아무도 안 봄).
    helper_collisions = _real_collisions(base_helpers, patch_helpers)
    if helper_collisions:
        raise ValueError(
            f"materialize: top-level helper name collision with different definitions — "
            f"{sorted(helper_collisions)}"
        )
    merged_helpers: dict[str, ast.stmt] = {**base_helpers, **patch_helpers}

    base_members = _extract_class_members(base_source) if base_source else {}
    patch_members = _extract_class_members(patch_source)
    override_hooks = _extract_override_hooks(patch_source)
    member_collisions = _real_collisions(base_members, patch_members)
    composed_names = {n for n in member_collisions if n in _COMPOSABLE_HOOKS and n not in override_hooks}
    replaced_names = member_collisions - composed_names
    if replaced_names:
        logger.warning("Patch member collision (patch overrides base): %s", sorted(replaced_names))

    existing_names = set(base_helpers) | set(patch_helpers) | set(base_members) | set(patch_members)
    extra_compose_imports: set[str] = set()
    merged_members: dict[str, ast.stmt] = {**base_members, **patch_members}
    for name in sorted(composed_names):
        base_renamed, patch_renamed, wrapper = _synthesize_composed_member(
            name, base_members[name], patch_members[name], existing_names,
        )
        merged_members[base_renamed.name] = base_renamed
        merged_members[patch_renamed.name] = patch_renamed
        merged_members[name] = wrapper
        if name in _COMPOSE_HELPER_IMPORTS:
            extra_compose_imports.add(_COMPOSE_HELPER_IMPORTS[name])

    seen_imports: set[str] = set()
    imports: list[str] = []
    for line in (
        (_extract_imports(base_source) if base_source else [])
        + _extract_imports(patch_source)
        + sorted(extra_compose_imports)
    ):
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


def load_base_snapshot(
    conn: "PgConn", competition_id: str, before_run_ts=None
) -> tuple[str | None, str]:
    """attempt 평가 시점의 base pipeline 소스를 Postgres 신뢰 사본에서 가져온다.

    1순위는 해당 시점 이전 마지막 승격 행의 materialized_code(승격 당시 병합본
    스냅샷). 스냅샷이 없는 과거 이력이면 replay 폴백 — 단 strict_sha로 재생해,
    재생 결과가 승격 당시 병합본과 다르면 진행하지 않고 raise한다 — 평가와
    다른 base로 제출하면 크래시하거나 조용히 열화된 예측이 제출된다.

    반환: (base_source | None, 출처 설명). 승격 이력이 없으면 (None, ...).
    """
    query = """
        SELECT p.materialized_code, p.pipeline_sha256
        FROM raw.pipelines p
        JOIN raw.attempts a USING (attempt_id)
        WHERE p.competition_id = %s
    """
    params: list = [competition_id]
    if before_run_ts is not None:
        query += " AND a.run_ts < %s"
        params.append(before_run_ts)
    query += " ORDER BY a.run_ts DESC LIMIT 1"

    row = conn.execute(query, params).fetchone()
    if not row:
        return None, "no promoted pipeline"

    snapshot, trusted_sha = row
    if snapshot:
        actual = hashlib.sha256(snapshot.encode()).hexdigest()
        if trusted_sha and actual != trusted_sha:
            raise RuntimeError(
                f"load_base_snapshot: materialized_code 스냅샷의 sha256이 신뢰 해시와 다르다 "
                f"(competition_id={competition_id}, expected {trusted_sha[:12]}…, got {actual[:12]}…) "
                "— Postgres 스냅샷 손상. 확인 없이 진행하지 않는다."
            )
        return snapshot, "materialized_code snapshot"

    best, _, count = replay_best_pipeline(
        conn, competition_id, before_run_ts=before_run_ts, strict_sha=True
    )
    return best, f"replay fallback ({count} promoted pipeline(s), sha verified)"


def replay_best_pipeline(
    conn: "PgConn", competition_id: str, before_run_ts=None, strict_sha: bool = False
) -> tuple[str | None, str | None, int]:
    """raw.pipelines 히스토리를 시간순 재생해 materialized base pipeline을 재구성한다.

    bin/rebuild_best_pipeline.py의 복구 로직과 동일한 재생 패턴을 공용화한 것 —
    거기서는 MinIO best_pipeline.py 손상 복구용, 여기서는 attempt 제출 시 평가
    시점의 base를 정확히 재현하는 용도. before_run_ts를 주면 그 시각 이전에
    승격된 pipeline만 재생한다 — attempt의 cv_score는 그 시점까지의 base 위에서
    측정됐으므로 이후 승격분을 섞으면 안 된다.

    반환: (materialized_source, 마지막 행의 pipeline_sha256, 재생한 pipeline 수).
    승격 이력이 없으면 (None, None, 0).

    invalid_reason이 표기된(격리된) 행은 건너뛴다 — 삭제가 아니라 스킵이라 이후
    legitimate 승격이 quarantine된 hook을 참조하지 않는 한(각 승격은 자기 완결적
    patch라 일반적으로 문제 없음) 안전한 소급 롤백이 된다.
    """
    query = """
        SELECT p.pipeline_id, a.run_ts, p.code, p.pipeline_sha256
        FROM raw.pipelines p
        JOIN raw.attempts a USING (attempt_id)
        WHERE p.competition_id = %s
          AND p.invalid_reason IS NULL
    """
    params: list = [competition_id]
    if before_run_ts is not None:
        query += " AND a.run_ts < %s"
        params.append(before_run_ts)
    query += " ORDER BY a.run_ts ASC"

    rows = conn.execute(query, params).fetchall()
    if not rows:
        return None, None, 0

    best: str | None = None
    last_sha256: str | None = None
    for pipeline_id, run_ts, code, pipeline_sha256 in rows:
        try:
            best = materialize_best_pipeline(best, code)
        except Exception as exc:
            raise RuntimeError(
                f"replay failed at pipeline_id={pipeline_id} run_ts={run_ts} "
                f"(competition_id={competition_id}): {exc}"
            ) from exc
        last_sha256 = pipeline_sha256

    assert best is not None  # rows non-empty, loop always assigns
    actual_sha256 = hashlib.sha256(best.encode()).hexdigest()
    if last_sha256 and actual_sha256 != last_sha256:
        # 재현 실패의 주원인은 blob 손상이 아니라 materialize 로직 자체의 변경 —
        # 과거 레이어를 현재 로직으로 재병합하면 당시 병합본과 다른 코드가 된다.
        if strict_sha:
            raise RuntimeError(
                f"replay_best_pipeline: 재생 결과 sha256이 마지막 승격분의 신뢰 해시와 다르다 "
                f"(competition_id={competition_id}) — 평가 시점 병합본을 재현할 수 없다. "
                "이 base로 제출하면 평가와 다른 예측이 나가므로 중단한다."
            )
        logger.warning(
            "replay_best_pipeline: 재생 결과 sha256이 마지막 승격분의 신뢰 해시와 다르다 "
            "(competition_id=%s) — 복구 용도로만 사용할 것.",
            competition_id,
        )
    return best, last_sha256, len(rows)


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
