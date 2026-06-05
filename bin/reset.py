"""
Run state reset — Postgres tables, generated code, submission CSVs.

Full reset (모든 대회):
  uv run python bin/reset.py

Competition-specific reset:
  uv run python bin/reset.py --competition playground-series-s4e1

Skip confirmation:
  uv run python bin/reset.py --yes
  uv run python bin/reset.py -c playground-series-s4e1 --yes
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CODE_DIR = ROOT / "runs" / "code"
RUNS_DIR = ROOT / "runs"


def _confirm(msg: str) -> bool:
    return input(f"{msg} [y/N] ").strip().lower() == "y"


def _submission_csvs() -> list[Path]:
    return list(RUNS_DIR.glob("submission_*.csv"))


def reset_full(yes: bool) -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from store.db import connect

    lines: list[str] = ["  DB:          all tables (TRUNCATE CASCADE)"]
    code_files = list(CODE_DIR.rglob("*.py")) if CODE_DIR.exists() else []
    if code_files:
        lines.append(f"  code:        {len(code_files)} file(s) in runs/code/")
    csvs = _submission_csvs()
    if csvs:
        lines.append(f"  submissions: {len(csvs)} CSV(s)")

    print("Full reset — will delete:")
    for l in lines:
        print(l)

    if not yes and not _confirm("Proceed?"):
        print("Aborted.")
        sys.exit(0)

    conn = connect(apply_schema=False)
    for table in ("raw.reflections", "raw.attempts", "raw.pipelines",
                  "raw.submission_budget", "raw.cycle_queue", "raw.competitions"):
        conn.execute(f"TRUNCATE {table}")
    conn.close()
    print("Truncated all tables.")

    if CODE_DIR.exists():
        shutil.rmtree(CODE_DIR)
        CODE_DIR.mkdir()
        print(f"Cleared {CODE_DIR.relative_to(ROOT)}")

    for csv in csvs:
        csv.unlink()
    if csvs:
        print(f"Deleted {len(csvs)} submission CSV(s)")

    print("Done.")


def reset_competition(competition_id: str, yes: bool) -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from store.db import connect

    if not DB_PATH.exists():
        print("DB not found — nothing to reset.")
        return

    conn = connect()

    exists = conn.execute(
        "select count(*) from raw.competitions where competition_id = %s",
        [competition_id],
    ).fetchone()
    if not exists or exists[0] == 0:
        print(f"Competition '{competition_id}' not found in DB.")
        conn.close()
        return

    attempt_count = conn.execute(
        "select count(*) from raw.attempts where competition_id = %s",
        [competition_id],
    ).fetchone()[0]
    reflection_count = conn.execute(
        "select count(*) from raw.reflections where competition_id = %s",
        [competition_id],
    ).fetchone()[0]
    code_uris = [
        row[0]
        for row in conn.execute(
            "select code_path from raw.attempts where competition_id = %s and code_path is not null",
            [competition_id],
        ).fetchall()
    ]

    print(f"Competition '{competition_id}' reset — will delete:")
    print(f"  {attempt_count} attempt(s), {reflection_count} reflection(s)")
    print(f"  {len(code_uris)} code file(s)")
    print(f"  competition record + submission budget")

    if not yes and not _confirm("Proceed?"):
        print("Aborted.")
        conn.close()
        sys.exit(0)

    from store.s3_code import delete as _code_delete
    deleted_code = sum(1 for uri in code_uris if _code_delete(uri))
    # 로컬 fallback 디렉토리도 정리
    comp_code_dir = CODE_DIR / competition_id
    if comp_code_dir.exists():
        shutil.rmtree(comp_code_dir)

    conn.execute("delete from raw.reflections where competition_id = %s", [competition_id])
    conn.execute("delete from raw.attempts where competition_id = %s", [competition_id])
    conn.execute("delete from raw.submission_budget where competition_id = %s", [competition_id])
    conn.execute("delete from raw.competitions where competition_id = %s", [competition_id])
    conn.close()

    print(f"Deleted {attempt_count} attempts, {reflection_count} reflections, {deleted_code} code files.")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset reflexion run state.")
    parser.add_argument("--competition", "-c", metavar="ID", help="Reset only this competition")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    if args.competition:
        reset_competition(args.competition, args.yes)
    else:
        reset_full(args.yes)


if __name__ == "__main__":
    main()
