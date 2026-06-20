"""대회 데이터를 Kaggle에서 다운로드 후 MinIO에 시딩.

Usage:
    uv run python -m bin.seed_competition_data s4e1
    uv run python -m bin.seed_competition_data s5e3

train.csv / test.csv / sample_submission.csv 를 MinIO kaggle/<S3_DATA_PATH>에 PUT.
멱등 — 재실행 시 덮어쓰기.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent

_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")
_BUCKET = "kaggle"
_FILES = ["train.csv", "test.csv", "sample_submission.csv"]


def _put(s3_path: str, name: str, data: bytes) -> None:
    key = f"{s3_path}{name}"
    url = f"{_MINIO_ENDPOINT}/{_BUCKET}/{key}"
    resp = requests.put(
        url,
        data=data,
        headers={"Content-Type": "text/csv"},
        timeout=120,
    )
    resp.raise_for_status()
    print(f"  PUT {url} ({len(data):,} bytes)")


def seed(slug: str) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        comp = importlib.import_module(f"config.competitions.{slug}")
    except ModuleNotFoundError:
        print(f"[seed] unknown competition slug: {slug}", file=sys.stderr)
        sys.exit(1)

    competition_id: str = comp.COMPETITION_ID
    s3_path: str | None = getattr(comp, "S3_DATA_PATH", None)

    if not s3_path:
        print(f"[seed] {slug} has no S3_DATA_PATH — local-only mode, skipping", file=sys.stderr)
        sys.exit(1)

    if not _MINIO_ENDPOINT:
        print("[seed] MINIO_ENDPOINT not set", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[seed] downloading {competition_id} ...")
        result = subprocess.run(
            ["uv", "run", "kaggle", "competitions", "download",
             "-c", competition_id, "--path", tmpdir],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

        zip_files = list(Path(tmpdir).glob("*.zip"))
        if not zip_files:
            print("[seed] no zip found after download", file=sys.stderr)
            sys.exit(1)

        with zipfile.ZipFile(zip_files[0]) as zf:
            zf.extractall(tmpdir)

        tmp = Path(tmpdir)
        for name in _FILES:
            candidates = list(tmp.rglob(name))
            if not candidates:
                print(f"  [skip] {name} not found in archive")
                continue
            data = candidates[0].read_bytes()
            _put(s3_path, name, data)

    print(f"[seed] done: {slug} ({competition_id}) → {_MINIO_ENDPOINT}/{_BUCKET}/{s3_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m bin.seed_competition_data <slug>")
        sys.exit(1)
    seed(sys.argv[1])
