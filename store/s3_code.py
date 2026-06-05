"""MinIO S3 code file storage — runs/code 저장·조회.

MINIO_ENDPOINT / MINIO_ACCESS_KEY_ID / MINIO_SECRET_ACCESS_KEY 환경변수가 없으면
로컬 파일시스템 fallback (direct 모드·로컬 개발용).
"""
from __future__ import annotations

import io
import os
from pathlib import Path

_BUCKET = "kaggle"
_S3_PREFIX = "runs/code"


def _client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["MINIO_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def _s3_available() -> bool:
    return bool(os.getenv("MINIO_ENDPOINT") and os.getenv("MINIO_ACCESS_KEY_ID"))


def upload(competition_id: str, filename: str, content: str) -> str:
    """코드를 저장하고 URI를 반환한다. S3 환경이면 s3:// URI, 아니면 로컬 경로."""
    if _s3_available():
        key = f"{competition_id}/{_S3_PREFIX}/{filename}"
        _client().put_object(
            Bucket=_BUCKET,
            Key=key,
            Body=content.encode(),
            ContentType="text/plain",
        )
        return f"s3://{_BUCKET}/{key}"
    # fallback: 로컬
    local_dir = Path(__file__).parent.parent / "runs" / "code" / competition_id
    local_dir.mkdir(parents=True, exist_ok=True)
    path = local_dir / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def download(uri: str) -> str | None:
    """URI(s3:// 또는 로컬 경로)로 코드 내용을 반환. 없으면 None."""
    if uri.startswith("s3://"):
        # s3://bucket/key
        without_scheme = uri[len("s3://"):]
        bucket, key = without_scheme.split("/", 1)
        try:
            obj = _client().get_object(Bucket=bucket, Key=key)
            return obj["Body"].read().decode()
        except Exception:
            return None
    path = Path(uri)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def delete(uri: str) -> bool:
    """URI가 가리키는 파일 삭제. 성공 여부 반환."""
    if uri.startswith("s3://"):
        without_scheme = uri[len("s3://"):]
        bucket, key = without_scheme.split("/", 1)
        try:
            _client().delete_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    path = Path(uri)
    if path.exists():
        path.unlink()
        return True
    return False
