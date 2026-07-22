"""Cold-start: fingerprint 계산 → 유사 대회 검색 → 시드/교훈 추출 → JSON 저장.

Usage:
    uv run python bin/start_competition.py \\
        --id playground-series-s5e3 \\
        --name "..." --task binary --metric auc
"""
import argparse
import json
from pathlib import Path

import polars as pl

from store.db import connect, ensure_competition
from store.fingerprint import compute as compute_fingerprint
from evaluator.metrics import get as get_metric
from memory.transfer import find_similar_competitions, cold_start_lessons, bootstrap_seeds

DATA_ROOT = Path(__file__).parent.parent / "data"
COLD_START_DIR = Path(__file__).parent.parent / "runs" / "cold_start"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",     required=True, help="Kaggle competition slug")
    parser.add_argument("--name",   required=True)
    parser.add_argument("--task",   required=True, choices=["binary", "multiclass", "regression"])
    parser.add_argument("--metric", required=True, help="auc / rmse / accuracy / ...")
    parser.add_argument("--target", default=None,  help="target column name (기본: 마지막 컬럼)")
    parser.add_argument("--k-similar", type=int, default=3)
    args = parser.parse_args()

    _, metric_sign, _ = get_metric(args.metric)

    train_path = DATA_ROOT / args.id / "train.csv"
    if not train_path.exists():
        print(f"train.csv not found: {train_path}")
        print(f"run: uv run kaggle competitions download -c {args.id} -p data/{args.id}")
        return

    train = pl.read_csv(train_path)
    target = args.target or train.columns[-1]
    print(f"train shape: {train.shape}  target: '{target}'")

    fp = compute_fingerprint(train, target, args.task, args.metric, metric_sign)
    print("fingerprint:", json.dumps(fp, indent=2))

    conn = connect()
    ensure_competition(
        conn,
        competition_id=args.id,
        name=args.name,
        task_type=args.task,
        metric=args.metric,
        metric_sign=metric_sign,
        fingerprint=fp,
    )
    print(f"\nraw.competitions 등록 완료: {args.id}")

    similar_with_dist = find_similar_competitions(conn, fp, exclude_id=args.id, k=args.k_similar)
    similar = [c for c, _ in similar_with_dist]

    print(f"\n유사 대회 (top-{args.k_similar}):")
    if similar_with_dist:
        for comp_id, dist in similar_with_dist:
            print(f"  {comp_id}  dist={dist:.1f}")
    else:
        print("  (없음 — 첫 대회)")

    lessons = cold_start_lessons(conn, similar, k=10)
    seeds = bootstrap_seeds(conn, similar, n=2)

    l2 = sum(1 for l in lessons if l["generality"] == "L2_class")
    l3 = sum(1 for l in lessons if l["generality"] == "L3_general")
    print(f"\n사용 가능한 교훈: L2_class={l2}, L3_general={l3} (총 {len(lessons)}개)")
    print(f"시드 파이프라인: {len(seeds)}개")
    for s in seeds:
        print(f"  {s['pipeline_id'][:8]}  cv={s['cv_score']:.5f}  from={s['competition_id']}")

    COLD_START_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COLD_START_DIR / f"{args.id}.json"
    out_path.write_text(json.dumps({
        "competition_id":      args.id,
        "similar_competitions": similar,
        "lesson_ids":          [l["reflection_id"] for l in lessons],
        "seed_pipeline_ids":   [s["pipeline_id"] for s in seeds],
    }, indent=2))
    print(f"\ncold-start 정보 저장: {out_path}")
    conn.close()


if __name__ == "__main__":
    main()
