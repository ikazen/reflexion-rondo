"""최신 확정 pipeline의 params를 소급 동결한다(#349, ADR-054) — 동결본을 평가해 저장된 cv와 일치할 때만 반영.

  uv run python -m bin.freeze_base --competition playground-series-s6e8 --dry-run

승격 시 동결은 forward-only라 이미 누적된 base 풀은 다음 승격 전까지 남는다. MinIO blob과 레지스트리 sha가 잠깐 어긋나면
_baseline_source_guard가 대회를 멈추고 사람이 풀 때까지 유지하므로, 그 대회 사이클이 없을 때 실행한다.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from bin.establish_baseline import _competition_id_to_slug
from cycle.materialize import materialize_best_pipeline, with_frozen_params
from cycle.promotion import MERGE_VERIFY_TOLERANCE
from evaluator.harness import split_audit_holdout
from runtime.isolate import eval_isolated
from store.db import PgConn, connect
from store.s3_code import download_best_pipeline, upload_best_pipeline
from store.train_data import load_train

_ATTEMPT_CV_SEED = 42
_FREEZE_ONLY_PATCH = (
    "class Patch:\n"
    '    action_type = "hyperparam_search"\n'
    "    changed_stages = []\n"
    '    rationale = "freeze base params"\n'
)


def _is_frozen(source: str, params: dict) -> bool:
    # materialize는 병합할 때마다 docstring 들여쓰기가 늘어 텍스트 비교로는 동결 여부를 알 수 없다.
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "param_candidates":
            body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
                try:
                    return ast.literal_eval(body[0].value) == [params]
                except ValueError:
                    return False
    return False


def freeze_latest(conn: PgConn, comp: ModuleType, dry_run: bool) -> bool:
    row = conn.execute(
        """
        SELECT p.pipeline_id, p.code, p.cv_score, p.materialized_code, a.params
        FROM raw.pipelines p JOIN raw.attempts a USING (attempt_id)
        WHERE p.competition_id = %s AND p.invalid_reason IS NULL AND p.materialized_code IS NOT NULL
          AND COALESCE(p.materialized_origin, '') NOT LIKE 'unverifiable:%%'
        ORDER BY a.run_ts DESC LIMIT 1
        """,
        [comp.COMPETITION_ID],
    ).fetchone()
    if row is None:
        print(f"  {comp.COMPETITION_ID}: 확정 pipeline 없음")
        return False
    pipeline_id, code, stored_cv, materialized, params = row
    if isinstance(params, str):
        params = json.loads(params)
    if not params:
        print(f"  {comp.COMPETITION_ID}: selected_params 없음 — 동결 대상 아님")
        return False
    if _is_frozen(materialized, params):
        print(f"  {comp.COMPETITION_ID}: 이미 동결됨")
        return False

    frozen = materialize_best_pipeline(materialized, with_frozen_params(_FREEZE_ONLY_PATCH, params))
    train90, _ = split_audit_holdout(load_train(comp), comp.TARGET, comp.IS_CLASSIFICATION)
    result = eval_isolated(
        source=frozen, train=train90, target_col=comp.TARGET, metric=comp.METRIC, prev_best=None,
        n_splits=getattr(comp, "N_SPLITS", 5), seed=_ATTEMPT_CV_SEED, is_classification=comp.IS_CLASSIFICATION,
        cpu_budget_sec=getattr(comp, "CPU_BUDGET_SECS", None),
    )
    if result.error_trace or result.cv_score is None:
        print(f"  {comp.COMPETITION_ID}: 동결본 평가 실패 — 중단\n{result.error_trace}")
        return False
    delta = abs(result.cv_score - stored_cv)
    print(f"  {comp.COMPETITION_ID} pipeline={pipeline_id[:8]}: stored cv {stored_cv!r} / frozen cv "
          f"{result.cv_score!r} / delta {delta:.3e} (tolerance {MERGE_VERIFY_TOLERANCE})")
    if delta > MERGE_VERIFY_TOLERANCE:
        print("  동결본이 저장된 cv를 재현하지 못함 — 반영하지 않는다")
        return False
    if dry_run:
        print("  dry-run — 미반영")
        return True

    new_sha = hashlib.sha256(frozen.encode()).hexdigest()
    upload_best_pipeline(comp.COMPETITION_ID, frozen)
    # upload_best_pipeline은 MinIO 실패 시 예외 없이 로컬 파일로 폴백하므로 다시 읽어 실제 반영을 확인한다.
    if download_best_pipeline(comp.COMPETITION_ID) != frozen:
        print(f"  {comp.COMPETITION_ID}: MinIO에 반영되지 않음 — DB는 갱신하지 않는다")
        return False
    try:
        # code에도 표식을 남겨야 replay_best_pipeline/rebuild_best_pipeline이 동결을 되돌리지 않는다.
        conn.execute(
            "UPDATE raw.pipelines SET code = %s, materialized_code = %s, pipeline_sha256 = %s,"
            " materialized_sha256 = %s WHERE pipeline_id = %s",
            [with_frozen_params(code, params), frozen, new_sha, new_sha, pipeline_id],
        )
    except Exception:
        upload_best_pipeline(comp.COMPETITION_ID, materialized)
        raise
    print(f"  {comp.COMPETITION_ID}: 동결 반영 (sha {new_sha[:12]})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True, help="competition_id (예: playground-series-s6e8)")
    parser.add_argument("--dry-run", action="store_true", help="평가·검증만 하고 반영하지 않는다")
    args = parser.parse_args()

    slug = _competition_id_to_slug().get(args.competition)
    if not slug:
        parser.error(f"unknown competition: {args.competition}")
    comp = importlib.import_module(f"config.competitions.{slug}")
    conn = connect(apply_schema=False)
    try:
        freeze_latest(conn, comp, args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
