"""핵심 가설 검증 CSV 내보내기.

DB 뷰(score_progression, cold_start_progression) → results/ 디렉터리.

Usage:
    uv run python -m bin.export_results [--competition s4e1] [--out results/]

--competition 미지정 시 DB 내 모든 대회를 처리한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl

from store.db import connect


def export_score_progression(conn, competition_id: str, out_dir: Path) -> Path:
    rows = conn.execute(
        """
        SELECT attempt_no, stage, cv_score, best_so_far
        FROM score_progression
        WHERE competition_id = %s
        ORDER BY attempt_no
        """,
        [competition_id],
    ).fetchall()
    df = pl.DataFrame(rows, schema=["attempt_no", "stage", "cv_score", "best_so_far"], orient="row")
    path = out_dir / f"{competition_id}_score_progression.csv"
    df.write_csv(path)
    return path


def export_holdout_divergence(conn, competition_id: str, out_dir: Path) -> Path | None:
    cols = {
        r[0]
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='raw' AND table_name='attempts'"
        ).fetchall()
    }
    if "holdout_score" not in cols:
        return None
    rows = conn.execute(
        """
        SELECT
            row_number() OVER (ORDER BY run_ts) AS attempt_no,
            cv_score,
            holdout_score
        FROM raw.attempts
        WHERE competition_id = %s
          AND holdout_score IS NOT NULL
        ORDER BY run_ts
        """,
        [competition_id],
    ).fetchall()
    if not rows:
        return None
    df = pl.DataFrame(rows, schema=["attempt_no", "cv_score", "holdout_score"], orient="row")
    df = df.with_columns((pl.col("cv_score") - pl.col("holdout_score")).alias("cv_minus_holdout"))
    path = out_dir / f"{competition_id}_holdout_divergence.csv"
    df.write_csv(path)
    return path


def export_cold_start_summary(conn, out_dir: Path) -> Path:
    rows = conn.execute(
        """
        SELECT
            c.competition_id,
            c.name,
            max(CASE WHEN p.stage = 'bootstrap' THEN p.best_so_far ELSE NULL END) AS bootstrap_best,
            max(p.best_so_far) AS overall_best,
            count(*) AS total_attempts
        FROM cold_start_progression p
        JOIN raw.competitions c USING (competition_id)
        GROUP BY c.competition_id, c.name
        ORDER BY c.competition_id
        """
    ).fetchall()
    df = pl.DataFrame(
        rows,
        schema=["competition_id", "name", "bootstrap_best", "overall_best", "total_attempts"],
        orient="row",
    )
    if not df.is_empty():
        df = df.with_columns(
            pl.when(pl.col("overall_best") != 0)
            .then((pl.col("bootstrap_best") / pl.col("overall_best")).round(4))
            .otherwise(None)
            .alias("warm_start_ratio")
        )
    path = out_dir / "cold_start_summary.csv"
    df.write_csv(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export experiment results to results/")
    parser.add_argument("--competition", "-c", default=None,
                        help="대회 ID (미지정 시 전체)")
    parser.add_argument("--out", default=str(ROOT / "results"),
                        help="출력 디렉터리 (기본: results/)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect(apply_schema=False)

    if args.competition:
        competition_ids = [args.competition]
    else:
        competition_ids = [
            r[0] for r in conn.execute(
                "SELECT competition_id FROM raw.competitions ORDER BY competition_id"
            ).fetchall()
        ]

    if not competition_ids:
        print("[export_results] 등록된 대회 없음")
        conn.close()
        return

    for comp_id in competition_ids:
        p1 = export_score_progression(conn, comp_id, out_dir)
        print(f"  {comp_id}: {p1.name}")

        p2 = export_holdout_divergence(conn, comp_id, out_dir)
        if p2:
            print(f"  {comp_id}: {p2.name}")

    p3 = export_cold_start_summary(conn, out_dir)
    print(f"  summary: {p3.name}")

    conn.close()
    print(f"[export_results] done → {out_dir}")


if __name__ == "__main__":
    main()
