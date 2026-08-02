"""
raw.pipelines 타깃 누수 격리 스캐너 (GH #96/#97 이후 소급, #99).

#97 이전에 승격된 파이프라인은 preprocess 훅의 valid-target 의존 누수(#96, s5e10
quantile-bin 패턴)를 걸러내는 게이트 없이 통과했을 수 있다. 이 스크립트는 그 기준
(evaluator.harness._check_preprocess_target_leak과 동일한 실측 동등성 검사)으로
기존 승격 행을 스캔해 invalid_reason='target_leak_preprocess: ...'을 표기한다.

삭제하지 않는다 — cycle/run.py._prev_best 등 조회 경로가 invalid_reason IS NULL로
제외하면 이력 보존과 격리를 동시에 만족한다. 격리 후 해당 대회는 MinIO
best_pipeline.py도 다시 맞춰야 한다(이 스크립트는 raw.pipelines 행만 표기하고
MinIO는 건드리지 않음) — bin/rebuild_best_pipeline.py를 이어서 실행할 것
(cycle/materialize.replay_best_pipeline이 invalid_reason 있는 행을 건너뛰도록
같이 수정됨).

materialized_code(#89 스냅샷)가 있으면 그걸로, 없으면 이 행 자신의 code(해당
attempt의 patch delta)로 검사한다 — 후자는 이 행이 그 자체로 preprocess를
재정의하지 않으면(이전 승격에서 물려받는 경우) 검사 대상이 없어 통과로
판정되므로, materialized_code 백필(bin/backfill_materialized_code.py)을
먼저 실행하면 더 정확하다.

대회당 전체 train 데이터를 로드하므로(store.train_data.load_train) 대형 대회
(s4e7 등)는 스캔 자체가 느릴 수 있다 — 1회성 운영 스크립트라 감내.

Dry-run (표기 안 함, 대상만 출력):
  uv run python -m bin.quarantine_leaks --dry-run
  uv run python -m bin.quarantine_leaks --competition playground-series-s5e10 --dry-run

실제 반영:
  uv run python -m bin.quarantine_leaks --competition playground-series-s5e10
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

_MIN_ROWS_FOR_SCAN = 20  # 표본이 너무 작으면 leak 판정 자체가 불안정 — 스킵


def _competition_id_to_slug() -> dict[str, str]:
    """config/competitions/*.py 스캔 → {competition_id: module_slug} 맵.

    bin/api.py._competition_id_to_slug와 동일 목적이나, 그쪽의 요청-스코프 캐시
    (_cache, 3600s TTL)는 1회성 CLI 스크립트에 불필요해 독립 구현한다.
    """
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

    반환: 'target_leak_preprocess: ...'(실제 누수 확정) 또는 None(깨끗함 **또는
    판정 불가**). exec 실패·데이터 로드 실패(#120 — 로컬에 train.csv 없음/MinIO
    미설정 등 순수 환경 문제)는 누수의 증거가 아니므로 격리하지 않는다 — 호출부
    scan()이 non-None 반환을 곧바로 격리 대상으로 집계하기 때문에, 여기서
    "판정 불가"를 "누수 확정"과 절대 같은 값으로 반환하면 안 된다(실측: 이
    구분이 없던 버전으로 실제 스캔했다면 s4e1/s5e3 정상 파이프라인 27개가
    로컬 데이터 캐시 부재만으로 부당하게 격리될 뻔했다). 판정 불가 사례는
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default=None, help="특정 competition_id만 스캔 (기본: invalid_reason NULL 전체)")
    parser.add_argument("--dry-run", action="store_true", help="표기 없이 대상만 출력")
    args = parser.parse_args()
    scan(args.competition, args.dry_run)


if __name__ == "__main__":
    main()
