"""
Run state reset — Postgres tables, generated code, submission CSVs.

Full reset (데이터만 삭제, 스키마 유지):
  uv run python bin/reset.py

Full reset (스키마 drop 후 재생성 — 개발 중 스키마 변경 시):
  uv run python bin/reset.py --hard

Competition-specific reset:
  uv run python bin/reset.py --competition playground-series-s4e1

Confirmed pipeline만 무효화 (attempts/reflections 보존, issue #3):
  uv run python bin/reset.py --competition playground-series-s4e1 --pipelines-only

Skip confirmation:
  uv run python bin/reset.py --yes
  uv run python bin/reset.py --hard --yes
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


def reset_hard(yes: bool) -> None:
    """raw 스키마 전체 drop 후 schema.sql로 재생성."""
    sys.path.insert(0, str(ROOT))
    from store.db import connect

    code_files = list(CODE_DIR.rglob("*.py")) if CODE_DIR.exists() else []
    csvs = _submission_csvs()

    print("Hard reset — will delete:")
    print("  DB:          DROP SCHEMA raw CASCADE + recreate from schema.sql")
    if code_files:
        print(f"  code:        {len(code_files)} file(s) in runs/code/")
    if csvs:
        print(f"  submissions: {len(csvs)} CSV(s)")

    if not yes and not _confirm("Proceed?"):
        print("Aborted.")
        sys.exit(0)

    conn = connect(apply_schema=False)
    raw = conn._conn
    raw.autocommit = False
    with raw.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS raw CASCADE")
        cur.execute("CREATE SCHEMA raw")
    raw.commit()
    raw.autocommit = True
    conn.close()

    connect(apply_schema=True).close()
    print("Schema dropped and recreated.")

    if CODE_DIR.exists():
        shutil.rmtree(CODE_DIR)
        CODE_DIR.mkdir()
        print(f"Cleared {CODE_DIR.relative_to(ROOT)}")

    for csv in csvs:
        csv.unlink()
    if csvs:
        print(f"Deleted {len(csvs)} submission CSV(s)")

    print("Done.")


def reset_full(yes: bool) -> None:
    """전체 테이블 TRUNCATE (스키마 유지)."""
    sys.path.insert(0, str(ROOT))
    from store.db import connect

    code_files = list(CODE_DIR.rglob("*.py")) if CODE_DIR.exists() else []
    csvs = _submission_csvs()

    print("Full reset — will delete:")
    print("  DB:          all tables (TRUNCATE CASCADE)")
    if code_files:
        print(f"  code:        {len(code_files)} file(s) in runs/code/")
    if csvs:
        print(f"  submissions: {len(csvs)} CSV(s)")

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
    sys.path.insert(0, str(ROOT))
    from store.db import connect
    from store.s3_code import delete as _code_delete, delete_best_pipeline as _best_delete

    conn = connect(apply_schema=False)

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

    deleted_code = sum(1 for uri in code_uris if _code_delete(uri))
    _best_delete(competition_id)
    comp_code_dir = CODE_DIR / competition_id
    if comp_code_dir.exists():
        shutil.rmtree(comp_code_dir)

    conn.execute("delete from raw.reflections where competition_id = %s", [competition_id])
    conn.execute("delete from raw.attempts where competition_id = %s", [competition_id])
    conn.execute("delete from raw.pipelines where competition_id = %s", [competition_id])
    conn.execute("delete from raw.submission_budget where competition_id = %s", [competition_id])
    conn.execute("delete from raw.action_bandit where scope_key = %s", [competition_id])
    conn.execute("delete from raw.super_cycle_context where competition_id = %s", [competition_id])
    conn.execute("delete from raw.competitions where competition_id = %s", [competition_id])
    conn.close()

    print(f"Deleted {attempt_count} attempts, {reflection_count} reflections, {deleted_code} code files.")
    print("Done.")


def reset_pipelines(competition_id: str, yes: bool) -> None:
    """confirmed pipeline(raw.pipelines + MinIO best_pipeline.py)만 무효화.

    attempts/reflections/competition 레코드는 보존한다 — 대회를 처음부터 다시 시작하지
    않고 "지금까지의 confirmed best만 리셋해 정상 사이클이 정직하게 재승격하도록" 유도할
    때 쓴다(issue #7의 phantom pipeline 정리가 정확히 이 케이스였다).

    issue #3 근본원인: 그 정리를 raw.pipelines만 지우는 수동 SQL로 처리해서 MinIO
    best_pipeline.py가 고아로 남았다(대응하는 Postgres 행이 없어 다음 confirmed-pipeline
    제출 시 sha256 mismatch로 발현). reset_competition()은 이미 _best_delete를
    raw.pipelines DELETE보다 먼저 호출해 이 문제가 없다 — 이 함수는 그 안전한 순서를
    "부분 리셋"에도 재사용해, 앞으로 유사 정리가 수동 SQL로 새지 않도록 한다.
    """
    sys.path.insert(0, str(ROOT))
    from store.db import connect
    from store.s3_code import delete_best_pipeline as _best_delete

    conn = connect(apply_schema=False)
    count = conn.execute(
        "select count(*) from raw.pipelines where competition_id = %s",
        [competition_id],
    ).fetchone()[0]

    if count == 0:
        print(f"'{competition_id}': no raw.pipelines rows — nothing to invalidate.")
        conn.close()
        return

    print(f"Pipelines-only reset for '{competition_id}' — will delete:")
    print(f"  {count} raw.pipelines row(s)")
    print(f"  MinIO best_pipeline.py (if present)")
    print(f"  attempts/reflections/competition record are PRESERVED")

    if not yes and not _confirm("Proceed?"):
        print("Aborted.")
        conn.close()
        sys.exit(0)

    conn.execute("delete from raw.pipelines where competition_id = %s", [competition_id])
    conn.close()
    deleted_blob = _best_delete(competition_id)

    print(f"Deleted {count} raw.pipelines row(s).")
    print(f"MinIO best_pipeline.py {'deleted' if deleted_blob else 'was already absent'}.")
    print("Done. Competition is cold-start for confirmed-pipeline purposes — "
          "next successful cross-seed confirm will re-promote honestly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset reflexion run state.")
    parser.add_argument("--competition", "-c", metavar="ID", help="Reset only this competition")
    parser.add_argument("--hard", action="store_true", help="Drop raw schema and recreate (dev용 스키마 변경 시)")
    parser.add_argument(
        "--pipelines-only", action="store_true",
        help="raw.pipelines + MinIO best_pipeline.py만 무효화, attempts/reflections 보존 (--competition 필수)",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    if args.pipelines_only:
        if not args.competition:
            parser.error("--pipelines-only requires --competition")
        reset_pipelines(args.competition, args.yes)
    elif args.competition:
        reset_competition(args.competition, args.yes)
    elif args.hard:
        reset_hard(args.yes)
    else:
        reset_full(args.yes)


if __name__ == "__main__":
    main()
