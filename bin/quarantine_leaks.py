"""
raw.pipelines 타깃 누수 격리 스캐너. 실행 순서·운영 절차는 docs/runbook.md §4-1.

evaluator.harness._check_preprocess_target_leak과 동일한 실측 동등성 검사로 기존
승격 행을 스캔해 invalid_reason='target_leak_preprocess: ...'을 표기한다(삭제 아님).
materialized_code 스냅샷이 없는 행은 이 행 자신의 code(patch delta)로만 검사하므로
이전 승격에서 물려받아 preprocess를 재정의하지 않는 행은 검사 대상이 없어 통과로
판정된다 — bin/backfill_materialized_code.py를 먼저 돌리면 더 정확하다.

Dry-run (표기 안 함, 대상만 출력):
  uv run python -m bin.quarantine_leaks --dry-run
  uv run python -m bin.quarantine_leaks --competition playground-series-s5e10 --dry-run

실제 반영:
  uv run python -m bin.quarantine_leaks --competition playground-series-s5e10

EXTRA_TRAIN_PATHS twin 중복 격리(#228/#287, 실측 검사가 아니라 시점 기준):
  uv run python -m bin.quarantine_leaks --twin-extra-train playground-series-s4e11 --since 2026-08-24 --dry-run
  uv run python -m bin.quarantine_leaks --twin-extra-train playground-series-s4e11 --since 2026-08-24
"""
from __future__ import annotations

import argparse
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent

_MIN_ROWS_FOR_SCAN = 20  # 표본이 너무 작으면 leak 판정 자체가 불안정 — 스킵


def _competition_id_to_slug() -> dict[str, str]:
    """config/competitions/*.py 스캔 → {competition_id: module_slug} 맵."""
    result: dict[str, str] = {}
    for path in (ROOT / "config" / "competitions").glob("*.py"):
        if path.stem.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"config.competitions.{path.stem}")
        except Exception:
            continue
        cid = getattr(mod, "COMPETITION_ID", None)
        if cid:
            result[cid] = path.stem
    return result


def _scan_pipeline(comp: object, code: str) -> str | None:
    """이 코드가 valid-target 의존 preprocess를 갖는지 실측.

    반환: 'target_leak_preprocess: ...'(실제 누수 확정) 또는 None(깨끗함 또는
    판정 불가). exec/데이터 로드 실패는 누수의 증거가 아니므로 반드시 None을
    반환한다 — 호출부 scan()이 non-None을 곧바로 격리 대상으로 집계하기 때문에,
    "판정 불가"를 "누수 확정"과 같은 값으로 반환하면 안 된다. 판정 불가 사례는
    stdout에 남겨 운영자가 재시도 필요 여부를 알 수 있게 한다.
    """
    from evaluator.harness import (
        BasePipeline, PatchedPipeline, PipelineContext,
        _check_preprocess_target_leak,
    )
    from store.train_data import load_train

    ns: dict = {}
    try:
        exec(compile(code, "<pipeline>", "exec"), ns)  # noqa: S102 — bin/submit.py, bin/rebuild_best_pipeline.py와 동일한 신뢰 경계(운영 스캐너)
        patch_cls = ns.get("Patch")
        if patch_cls is None:
            return None
        pipeline = PatchedPipeline(BasePipeline(), patch_cls())
    except Exception as exc:
        print(f"    [skip] 코드 로드 실패(판정 불가, 격리 안 함): {exc!r}"[:500])
        return None

    try:
        train = load_train(comp)
    except Exception as exc:
        print(f"    [skip] 데이터 로드 실패(판정 불가, 격리 안 함): {exc!r}"[:500])
        return None

    if train.height < _MIN_ROWS_FOR_SCAN:
        return None

    split = max(1, train.height // 5)
    va, tr = train[:split], train[split:]

    ctx = PipelineContext(
        target_col=comp.TARGET, metric=comp.METRIC, n_splits=5, seed=42,
        is_classification=comp.IS_CLASSIFICATION,
    )
    try:
        _check_preprocess_target_leak(pipeline, tr, va, ctx)
    except ValueError as exc:
        return f"target_leak_preprocess: {exc}"[:500]
    except Exception:
        return None  # 다른 실패(모델 미적합 등)는 이 스캔의 관심사 밖 — 판정 보류

    return None


def scan(competition_id: str | None, dry_run: bool) -> None:
    sys.path.insert(0, str(ROOT))
    from store.db import connect

    slug_map = _competition_id_to_slug()

    conn = connect(apply_schema=False)
    try:
        query = """
            SELECT p.pipeline_id, p.competition_id, coalesce(p.materialized_code, p.code)
            FROM raw.pipelines p
            WHERE p.invalid_reason IS NULL
        """
        params: list = []
        if competition_id:
            query += " AND p.competition_id = %s"
            params.append(competition_id)
        rows = conn.execute(query, params).fetchall()

        checked = flagged = skipped = 0
        affected: set[str] = set()
        for pipeline_id, cid, code in rows:
            slug = slug_map.get(cid)
            if not slug or not code:
                skipped += 1
                continue
            comp = importlib.import_module(f"config.competitions.{slug}")
            checked += 1
            reason = _scan_pipeline(comp, code)
            if not reason:
                continue
            flagged += 1
            affected.add(cid)
            if dry_run:
                print(f"[dry-run] would quarantine {cid} pipeline_id={pipeline_id[:8]}: {reason}")
            else:
                conn.execute(
                    "UPDATE raw.pipelines SET invalid_reason = %s WHERE pipeline_id = %s",
                    [reason, pipeline_id],
                )
                print(f"quarantined {cid} pipeline_id={pipeline_id[:8]}: {reason}")
    finally:
        conn.close()

    print(f"\nchecked={checked} flagged={flagged} skipped(no-slug/no-code)={skipped}")
    if affected:
        print("영향받은 대회 — MinIO best_pipeline.py도 재구성 필요:")
        for cid in sorted(affected):
            print(f"  uv run python -m bin.rebuild_best_pipeline --competition {cid}")


def quarantine_twin_extra_train(conn, competition_id: str, since_dt: datetime, dry_run: bool) -> tuple[int, int]:
    """EXTRA_TRAIN_PATHS twin 중복(#228/#287) 오염 기간의 confirmed pipeline을
    격리하고 그 기간 reflection을 archive한다. 반환: (격리된 pipeline 수, archive된
    reflection 수).

    실측 동등성 검사(_scan_pipeline)가 아니라 시점 기준 표기다 — twin 오염은
    코드가 아니라 그 시점의 load_train() 결과 자체가 오염됐던 것이라 코드 재실행
    검사로는 재현·판정할 수 없다(store/train_data.py가 이미 고쳐졌으므로 지금
    재실행하면 깨끗하게 나온다).
    """
    pipeline_rows = conn.execute(
        """
        SELECT p.pipeline_id FROM raw.pipelines p
        JOIN raw.attempts a USING (attempt_id)
        WHERE p.competition_id = %s AND p.invalid_reason IS NULL AND a.run_ts >= %s
        """,
        [competition_id, since_dt],
    ).fetchall()
    reflection_rows = conn.execute(
        """
        SELECT reflection_id FROM raw.reflections
        WHERE competition_id = %s AND archived = false AND created_at >= %s
        """,
        [competition_id, since_dt],
    ).fetchall()

    reason = f"contaminated_twin_extra_train: EXTRA_TRAIN_PATHS twin 중복 (#287), since={since_dt.isoformat()}"
    if dry_run:
        print(f"[dry-run] would quarantine {len(pipeline_rows)} pipeline(s), archive {len(reflection_rows)} reflection(s)")
        for (pid,) in pipeline_rows:
            print(f"  pipeline_id={pid[:8]}")
        return len(pipeline_rows), len(reflection_rows)

    if pipeline_rows:
        conn.execute(
            "UPDATE raw.pipelines SET invalid_reason = %s WHERE pipeline_id = ANY(%s::text[])",
            [reason, [r[0] for r in pipeline_rows]],
        )
    if reflection_rows:
        conn.execute(
            "UPDATE raw.reflections SET archived = true WHERE reflection_id = ANY(%s::text[])",
            [[r[0] for r in reflection_rows]],
        )
    print(f"quarantined {len(pipeline_rows)} pipeline(s), archived {len(reflection_rows)} reflection(s)")
    if pipeline_rows:
        print(
            f"\n다음 실행 필요:\n"
            f"  uv run python -m bin.establish_baseline --competition {competition_id} --remeasure\n"
            f"  uv run python -m bin.rebuild_best_pipeline --competition {competition_id}"
        )
    return len(pipeline_rows), len(reflection_rows)


def _run_twin_extra_train_cli(competition_id: str, since: str, dry_run: bool) -> None:
    sys.path.insert(0, str(ROOT))
    from store.db import connect

    since_dt = datetime.fromisoformat(since)
    if since_dt.tzinfo is None:
        since_dt = since_dt.replace(tzinfo=timezone.utc)

    conn = connect(apply_schema=False)
    try:
        quarantine_twin_extra_train(conn, competition_id, since_dt, dry_run)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default=None, help="특정 competition_id만 스캔 (기본: invalid_reason NULL 전체)")
    parser.add_argument("--dry-run", action="store_true", help="표기 없이 대상만 출력")
    parser.add_argument(
        "--twin-extra-train", metavar="COMPETITION_ID", default=None,
        help="실측 검사 대신 시점 기준으로 EXTRA_TRAIN_PATHS twin 오염(#228/#287)을 격리 — --since 필수",
    )
    parser.add_argument("--since", default=None, help="--twin-extra-train과 함께 사용, ISO 형식(예: 2026-08-24)")
    args = parser.parse_args()

    if args.twin_extra_train:
        if not args.since:
            parser.error("--twin-extra-train requires --since")
        _run_twin_extra_train_cli(args.twin_extra_train, args.since, args.dry_run)
        return

    scan(args.competition, args.dry_run)


if __name__ == "__main__":
    main()
