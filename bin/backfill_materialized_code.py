"""raw.pipelines.materialized_code 행동 검증 백필 (#254).

승격 attempt를 제출하려면 그 attempt **직전** 승격분의 병합본 스냅샷(materialized_code)이
필요한데, 2026-08 이전 승격분엔 스냅샷이 없다(#89 이전). replay 폴백은 materialize 합성
규칙이 바뀐 뒤(ADR-037/#232) 승격 당시와 다른 코드를 만들어 sha 검증(strict_sha)에 걸린다.

#166-168의 원칙("소스 해시는 로직 변경을 못 넘긴다, 행동 재현만 유효")을 적용한다. 스냅샷
없는 승격 행마다 체인을 오늘 로직으로 재생 → 재평가 → verdict:

  backfill:sha    replay 결과 sha가 pipeline_sha256과 일치 (eval 불필요)
  backfill:cv     drift probe 통과(데이터 비교 가능) + 재평가 cv가 기록값과 tolerance 안
  backfill:chain  데이터 이동됨(--allow-chain) + 재평가가 직전 검증층 대비 유의한 회귀 아님
  unverifiable:*  그 외 — materialized_origin에만 기록, invalid_reason은 안 건드림
                  (단 데이터 비교 가능한데 cv가 어긋나면 materialize_unreproducible로 격리)

드리프트한 대회(예: s4e10 — 전 이력이 2026-06 승격인데 원본 병합은 2026-08-24)는 어떤
tolerance로도 기록 cv를 재현 못 한다. 그건 정상이다 — quarantine이 정답이고, 정직하게
제출 가능한 것만 백로그에 남긴다.

기존 스크립트(MinIO 앵커로 대회당 마지막 행만 복구)는 backfill:minio tier로 흡수했다.

Dry-run (기본, DB 쓰기 없음):
  uv run python -m bin.backfill_materialized_code --competition playground-series-s4e12
실제 반영 + 약한 tier + cv_score 재작성:
  uv run python -m bin.backfill_materialized_code --competition playground-series-s4e10 \
      --apply --allow-chain --remeasure
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent


@dataclass(frozen=True, slots=True)
class ChainRow:
    pipeline_id: str
    attempt_id: str
    run_ts: object
    code: str
    cv_score: float | None
    pipeline_sha256: str | None
    materialized_code: str | None
    materialized_sha256: str | None
    materialized_origin: str | None
    invalid_reason: str | None


@dataclass(frozen=True, slots=True)
class Verdict:
    origin: str
    snapshot: str | None       # materialized_code로 쓸 값 (None = code 컬럼 안 건드림)
    new_cv: float | None       # --remeasure 시 cv_score에 쓸 값
    invalid_reason: str | None  # 격리 (mismatch 계열만)
    note: str


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _eval_base(comp: object, source: str, train90, cpu_budget):
    from runtime.isolate import eval_isolated
    return eval_isolated(
        source=source,
        train=train90,
        target_col=comp.TARGET,
        metric=comp.METRIC,
        prev_best=None,
        n_splits=getattr(comp, "N_SPLITS", 5),
        seed=42,
        is_classification=comp.IS_CLASSIFICATION,
        cpu_budget_sec=cpu_budget,
    )


def _drift_probe(comp: object, chain: list[ChainRow], train90, cpu_budget) -> bool | None:
    """대회의 최신 materialized_code 행을 오늘 데이터로 재평가해 기록 cv와 대조한다.

    재현되면 cv tier가 이 대회에서 신뢰 가능(True), 안 되면 학습 데이터가 이동한 것
    (False), 프로브할 행이 없거나 eval이 죽으면 None. None/False면 cv tier는 못 쓴다.
    """
    from cycle.promotion import MERGE_VERIFY_TOLERANCE
    probe = next((r for r in reversed(chain) if r.materialized_code and r.cv_score is not None), None)
    if probe is None:
        return None
    res = _eval_base(comp, probe.materialized_code, train90, cpu_budget)
    if res.error_trace or res.cv_score is None:
        print(f"  drift probe: eval 실패 ({res.error_trace}) — cv tier 비활성")
        return None
    ok = abs(res.cv_score - probe.cv_score) <= MERGE_VERIFY_TOLERANCE
    print(f"  drift probe: {probe.pipeline_id[:8]} 기록 cv {probe.cv_score} vs 재평가 {res.cv_score} "
          f"→ {'비교 가능' if ok else '데이터 이동됨'}")
    return ok


def _verdict_for_row(
    conn, comp: object, row: ChainRow, train90, comparable: bool | None,
    prev_eval: tuple[float, list[float] | None] | None, allow_chain: bool, cpu_budget,
    is_latest: bool,
) -> tuple[Verdict, tuple[float, list[float] | None] | None]:
    """한 행의 verdict + (다음 층에 물려줄 prev_eval). prev_eval은 (cv, fold_scores)."""
    from cycle.materialize import replay_best_pipeline
    from cycle.promotion import MERGE_VERIFY_TOLERANCE
    from evaluator.harness import is_significant_gain

    metric_sign = getattr(comp, "METRIC_SIGN", 1)

    # 이미 스냅샷이 있는 행 — 독립적으로 신뢰 가능(sha 대조). code는 안 건드리고
    # materialized_sha256/origin만 채운다. 레거시 행은 pipeline_sha256도 NULL이라
    # sha(자기 code)를 materialized_sha256에 넣어 "self-trust"로 load_base_snapshot
    # 가드를 통과시킨다(스냅샷이 곧 신뢰 사본이라는 원래 계약 그대로).
    if row.materialized_code:
        actual = _sha(row.materialized_code)
        trusted = row.materialized_sha256 or row.pipeline_sha256
        if trusted and actual != trusted:
            return Verdict("unverifiable:snapshot_corrupt", None, None,
                           "materialize_unreproducible: stored snapshot sha mismatch",
                           f"expected {trusted[:12]} got {actual[:12]}"), None
        nxt = prev_eval
        if allow_chain:
            r = _eval_base(comp, row.materialized_code, train90, cpu_budget)
            if not r.error_trace and r.cv_score is not None:
                nxt = (r.cv_score, r.fold_scores)
        # 실제로 재현하지 않았으므로(sha 대조만) origin은 기존 값 보존, NULL이면 promote.
        return Verdict(row.materialized_origin or "promote", row.materialized_code, None, None,
                       "기존 스냅샷 유지"), nxt

    # 스냅샷 없음. 먼저 MinIO 앵커 — 이 행이 대회 최신 승격분이고 현 best_pipeline.py의
    # sha가 pipeline_sha256과 일치하면 그게 승격 당시 병합본이다(구 스크립트의 유일 경로).
    if is_latest and row.pipeline_sha256:
        from store.s3_code import download_best_pipeline
        blob = download_best_pipeline(comp.COMPETITION_ID)
        if blob and _sha(blob) == row.pipeline_sha256:
            return Verdict("backfill:minio", blob, None, None, "MinIO best_pipeline.py sha 일치"), None

    # 이 행까지(포함) 재생
    try:
        candidate, _, n = replay_best_pipeline(
            conn, comp.COMPETITION_ID, strict_sha=False, stop_at_pipeline_id=row.pipeline_id,
        )
    except Exception as exc:
        return Verdict("unverifiable:eval_error", None, None, None,
                       f"replay 실패: {exc}"), None
    if candidate is None:
        return Verdict("unverifiable:eval_error", None, None, None, "replay가 빈 결과"), None

    # tier exact — eval 없이
    if row.pipeline_sha256 and _sha(candidate) == row.pipeline_sha256:
        return Verdict("backfill:sha", candidate, None, None, f"replay {n}층 sha 일치"), None

    res = _eval_base(comp, candidate, train90, cpu_budget)
    if res.error_trace or res.cv_score is None:
        return Verdict("unverifiable:eval_error", None, None, None,
                       f"재평가 실패: {res.error_trace}"), None
    this_eval = (res.cv_score, res.fold_scores)

    # tier cv — 데이터 비교 가능 + 기록 cv 재현
    if comparable and row.cv_score is not None and abs(res.cv_score - row.cv_score) <= MERGE_VERIFY_TOLERANCE:
        return Verdict("backfill:cv", candidate, res.cv_score, None,
                       f"재평가 cv {res.cv_score} ≈ 기록 {row.cv_score}"), this_eval

    # 데이터 비교 가능한데 cv 어긋남 = 진짜 mismatch → 격리
    if comparable and row.cv_score is not None:
        return Verdict("unverifiable:cv_mismatch", None, None,
                       f"materialize_unreproducible: 재평가 cv {res.cv_score} vs 기록 {row.cv_score}",
                       "데이터 비교 가능한데 cv 불일치"), None

    # tier chain — 데이터 이동됨, 직전 검증층 대비 회귀 아니면 수용
    if allow_chain:
        if prev_eval is None:
            return Verdict("backfill:chain", candidate, res.cv_score, None,
                           f"체인 앵커 (cv {res.cv_score}, 비교 대상 없음)"), this_eval
        prev_cv, prev_folds = prev_eval
        prev_beats_cand = is_significant_gain(
            gain_vs_best=metric_sign * (prev_cv - res.cv_score),
            cv_fold_var=res.cv_fold_var or 0.0,
            candidate_fold_scores=prev_folds,
            baseline_fold_scores=res.fold_scores,
            metric_sign=metric_sign,
        )
        if not prev_beats_cand:
            return Verdict("backfill:chain", candidate, res.cv_score, None,
                           f"직전층 대비 회귀 아님 (cv {prev_cv} → {res.cv_score})"), this_eval
        return Verdict("unverifiable:chain_regression", None, None, None,
                       f"직전층 {prev_cv} → 재평가 {res.cv_score}, 유의한 회귀"), None

    return Verdict("unverifiable:train_drift", None, None, None,
                   f"데이터 이동됨, --allow-chain 없음 (재평가 cv {res.cv_score})"), None


def _write(conn, row: ChainRow, v: Verdict, remeasure: bool, apply: bool) -> None:
    sets = ["materialized_origin = %s"]
    params: list = [v.origin]
    if v.snapshot is not None:
        sets.append("materialized_code = %s")
        sets.append("materialized_sha256 = %s")
        params += [v.snapshot, _sha(v.snapshot)]
    if v.invalid_reason is not None:
        sets.append("invalid_reason = coalesce(invalid_reason, %s)")
        params.append(v.invalid_reason)
    if remeasure and v.new_cv is not None:
        sets.append("cv_score = %s")
        params.append(v.new_cv)
    params.append(row.pipeline_id)
    tag = "반영" if apply else "dry-run"
    print(f"  [{tag}] {row.pipeline_id[:8]} → {v.origin}  ({v.note})")
    if apply:
        conn.execute(f"UPDATE raw.pipelines SET {', '.join(sets)} WHERE pipeline_id = %s", params)


def backfill_competition(
    conn, comp: object, apply: bool, allow_chain: bool, remeasure: bool,
) -> dict[str, int]:
    from cycle.materialize import promotion_chain
    from evaluator.harness import split_audit_holdout
    from store.train_data import load_train

    cid = comp.COMPETITION_ID
    chain = [ChainRow(*r) for r in promotion_chain(conn, cid)]
    if not chain:
        print(f"{cid}: 승격 이력 없음")
        return {}
    print(f"\n{cid}: 승격 {len(chain)}행, 스냅샷 보유 "
          f"{sum(1 for r in chain if r.materialized_code)}행")

    cpu_budget = getattr(comp, "CPU_BUDGET_SECS", None)
    train = load_train(comp)
    train90, _ = split_audit_holdout(train, comp.TARGET, comp.IS_CLASSIFICATION)
    comparable = _drift_probe(comp, chain, train90, cpu_budget)

    # 각 행은 독립적으로 판정한다 — replay_best_pipeline(stop_at_pipeline_id)이 매번
    # 체인 앞부분을 통째로 재생하고 invalid_reason 걸린 행은 건너뛰므로, 어떤 행이
    # 격리돼도 뒤 행의 재생에서 자동으로 빠진다("격리 = 롤백"). prev_eval(직전 검증층
    # 행동)만 수용 시에 전진시킨다.
    tally: dict[str, int] = {}
    prev_eval: tuple[float, list[float] | None] | None = None
    for i, row in enumerate(chain):
        if row.invalid_reason is not None:
            print(f"  [스킵] {row.pipeline_id[:8]} 이미 격리됨: {row.invalid_reason[:60]}")
            tally["already_invalid"] = tally.get("already_invalid", 0) + 1
            continue
        v, nxt = _verdict_for_row(
            conn, comp, row, train90, comparable, prev_eval, allow_chain, cpu_budget,
            is_latest=(i == len(chain) - 1),
        )
        if not v.origin.startswith("unverifiable:"):
            prev_eval = nxt
        _write(conn, row, v, remeasure, apply)
        tally[v.origin] = tally.get(v.origin, 0) + 1
    return tally


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", action="append", default=None,
                        help="competition_id (반복 가능). 기본: active_competition_ids()")
    parser.add_argument("--apply", action="store_true", help="DB 반영 (기본은 dry-run)")
    parser.add_argument("--allow-chain", action="store_true",
                        help="데이터 이동된 대회에 chain tier(약한 보장) 허용")
    parser.add_argument("--remeasure", action="store_true",
                        help="수용된 행의 cv_score를 재평가값으로 갱신 (--allow-chain 권장)")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    import importlib

    from config.competitions import active_competition_ids, competition_id_to_slug
    from store.db import connect

    slug_map = competition_id_to_slug()
    targets = args.competition or sorted(active_competition_ids())

    conn = connect(apply_schema=False)
    try:
        grand: dict[str, int] = {}
        for cid in targets:
            slug = slug_map.get(cid)
            if not slug:
                print(f"{cid}: config 없음 — 스킵")
                continue
            comp = importlib.import_module(f"config.competitions.{slug}")
            tally = backfill_competition(conn, comp, args.apply, args.allow_chain, args.remeasure)
            for k, n in tally.items():
                grand[k] = grand.get(k, 0) + n
    finally:
        conn.close()

    summary = ", ".join(f"{k}={n}" for k, n in sorted(grand.items())) or "(대상 없음)"
    print(f"\n{'반영' if args.apply else 'dry-run'} 합계: {summary}")


if __name__ == "__main__":
    main()
