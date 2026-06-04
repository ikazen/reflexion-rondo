"""Reflexion 루프 연속 실행.

Usage:
    uv run python -m bin.run_reflexion --competition s4e1 --stage bootstrap --cycles 5
    uv run python -m bin.run_reflexion --competition s5e3 --stage bootstrap --cycles 5 --cold-start
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import polars as pl

from cycle.run import CycleConfig, run_cycle
from memory.transfer import cold_start_lessons, bootstrap_seeds
from store.db import connect, ensure_competition

ROOT = Path(__file__).parent.parent
COLD_START_DIR = ROOT / "runs" / "cold_start"


def _load_cold_start(competition_id: str, conn) -> tuple[list[dict], str | None]:
    """cold-start JSON에서 교훈 목록과 첫 시드 코드를 로드."""
    path = COLD_START_DIR / f"{competition_id}.json"
    if not path.exists():
        return [], None

    data = json.loads(path.read_text())
    similar = data.get("similar_competitions", [])
    seed_ids = data.get("seed_pipeline_ids", [])

    lessons = cold_start_lessons(conn, similar, k=10)

    seed_code: str | None = None
    if seed_ids:
        row = conn.execute(
            "select code from raw.pipelines where pipeline_id = ? limit 1",
            [seed_ids[0]],
        ).fetchone()
        if row:
            seed_code = row[0]

    return lessons, seed_code


def _format_cold_start_context(lessons: list[dict]) -> str:
    if not lessons:
        return ""
    parts = ["\n## Cold-Start Lessons (from similar competitions)"]
    for i, l in enumerate(lessons, 1):
        parts.append(f"{i}. [{l['generality']}] {l['full_lesson']}")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", default="s4e1",
                        help="competition config name (config/competitions/*.py)")
    parser.add_argument("--stage", default="bootstrap",
                        choices=["bootstrap", "reflexion", "exploitation"])
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--cold-start", action="store_true",
                        help="runs/cold_start/{competition_id}.json 에서 교훈+시드 주입")
    args = parser.parse_args()

    comp = importlib.import_module(f"config.competitions.{args.competition}")

    train = pl.read_csv(comp.DATA_DIR / "train.csv").drop(comp.DROP_COLS)
    conn = connect()
    ensure_competition(
        conn,
        competition_id=comp.COMPETITION_ID,
        name=comp.NAME,
        task_type=comp.TASK_TYPE,
        metric=comp.METRIC,
        metric_sign=comp.METRIC_SIGN,
    )

    # cold-start 준비
    cold_lessons: list[dict] = []
    seed_code: str | None = None
    if args.cold_start:
        cold_lessons, seed_code = _load_cold_start(comp.COMPETITION_ID, conn)
        print(f"cold-start: {len(cold_lessons)}개 교훈, 시드={'있음' if seed_code else '없음'}")

    eda_card = comp.EDA_CARD
    if cold_lessons:
        eda_card = eda_card + _format_cold_start_context(cold_lessons)

    failed = 0
    for i in range(args.cycles):
        print(f"\n--- cycle {i + 1}/{args.cycles} (stage={args.stage}) ---")

        # 첫 bootstrap 사이클에만 seed_code 주입
        this_seed = seed_code if (args.stage == "bootstrap" and i == 0) else None

        config = CycleConfig(
            competition_id=comp.COMPETITION_ID,
            train=train,
            target_col=comp.TARGET,
            metric=comp.METRIC,
            stage=args.stage,
            eda_card=eda_card,
            n_splits=getattr(comp, "N_SPLITS", 5),
            seed=42,
            k_retrieve=5,
            is_classification=comp.IS_CLASSIFICATION,
            seed_code=this_seed,
        )
        try:
            result = run_cycle(conn, config)
        except Exception as exc:
            failed += 1
            print(f"[cycle {i + 1} FAILED] {exc}")
            continue

        print(f"attempt_id:    {result.attempt_id}")
        print(f"cv_score:      {result.cv_score}")
        print(f"label:         {result.label}")
        print(f"gain_vs_best:  {result.gain_vs_best}")
        print(f"retries:       {result.retries}")
        print(f"reflection_id: {result.reflection_id}")
        print(f"code:          {result.code_path}")
        if result.error_trace:
            print(f"error:\n{result.error_trace}")

    if failed:
        print(f"\n{failed}/{args.cycles} cycle(s) failed.")

    conn.close()


if __name__ == "__main__":
    main()
