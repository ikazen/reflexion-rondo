"""MinIO S3 code file storage — runs/code 저장·조회.

MinIO kaggle 버킷은 익명 read/write 허용 — 인증 불필요.
MINIO_ENDPOINT 환경변수 미설정 시 http://minio.internal 기본값 사용.
S3 접근 실패 시 로컬 파일시스템 fallback.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

_BUCKET = "kaggle"
_S3_PREFIX = "runs/code"
_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")


def upload(competition_id: str, filename: str, content: str) -> str:
    """코드를 저장하고 URI를 반환한다. S3 성공 시 s3:// URI, 실패 시 로컬 경로."""
    key = f"{competition_id}/{_S3_PREFIX}/{filename}"
    try:
        resp = requests.put(
            f"{_ENDPOINT}/{_BUCKET}/{key}",
            data=content.encode(),
            headers={"Content-Type": "text/plain"},
            timeout=30,
        )
        resp.raise_for_status()
        return f"s3://{_BUCKET}/{key}"
    except Exception:
        pass
    local_dir = Path(__file__).parent.parent / "runs" / "code" / competition_id
    local_dir.mkdir(parents=True, exist_ok=True)
    path = local_dir / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def download(uri: str) -> str | None:
    """URI(s3:// 또는 로컬 경로)로 코드 내용을 반환. 없으면 None."""
    if uri.startswith("s3://"):
        bucket, key = uri[len("s3://"):].split("/", 1)
        try:
            resp = requests.get(f"{_ENDPOINT}/{bucket}/{key}", timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception:
            return None
    path = Path(uri)
    return path.read_text(encoding="utf-8") if path.exists() else None


_BEST_KEY = "best_pipeline.py"


def upload_best_pipeline(competition_id: str, content: str) -> str:
    """Materialized best pipeline 저장 → URI 반환."""
    key = f"{competition_id}/{_BEST_KEY}"
    try:
        resp = requests.put(
            f"{_ENDPOINT}/{_BUCKET}/{key}",
            data=content.encode(),
            headers={"Content-Type": "text/plain"},
            timeout=30,
        )
        resp.raise_for_status()
        return f"s3://{_BUCKET}/{key}"
    except Exception:
        pass
    local_dir = Path(__file__).parent.parent / "runs" / "best"
    local_dir.mkdir(parents=True, exist_ok=True)
    path = local_dir / f"{competition_id}_best_pipeline.py"
    path.write_text(content, encoding="utf-8")
    return str(path)


def download_best_pipeline(competition_id: str) -> str | None:
    """Materialized best pipeline 읽기. 없으면 None."""
    key = f"{competition_id}/{_BEST_KEY}"
    try:
        resp = requests.get(f"{_ENDPOINT}/{_BUCKET}/{key}", timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception:
        pass
    path = Path(__file__).parent.parent / "runs" / "best" / f"{competition_id}_best_pipeline.py"
    return path.read_text(encoding="utf-8") if path.exists() else None


def delete_best_pipeline(competition_id: str) -> bool:
    """competition_id에 귀속된 best_pipeline.py 블롭 삭제. MinIO + 로컬 폴백 모두 처리."""
    key = f"{competition_id}/{_BEST_KEY}"
    deleted = False
    try:
        requests.delete(f"{_ENDPOINT}/{_BUCKET}/{key}", timeout=30).raise_for_status()
        deleted = True
    except Exception:
        pass
    local = Path(__file__).parent.parent / "runs" / "best" / f"{competition_id}_best_pipeline.py"
    if local.exists():
        local.unlink()
        deleted = True
    return deleted


def delete(uri: str) -> bool:
    """URI가 가리키는 파일 삭제. 성공 여부 반환."""
    if uri.startswith("s3://"):
        bucket, key = uri[len("s3://"):].split("/", 1)
        try:
            requests.delete(f"{_ENDPOINT}/{bucket}/{key}", timeout=30).raise_for_status()
            return True
        except Exception:
            return False
    path = Path(uri)
    if path.exists():
        path.unlink()
        return True
    return False
