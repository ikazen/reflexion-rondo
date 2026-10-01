"""MinIO S3 code file storage — runs/code 저장·조회.

MinIO kaggle 버킷은 익명 read/write 허용 — 인증 불필요. S3 접근 실패 시 로컬 파일시스템 fallback.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

_BUCKET = "kaggle"
_S3_PREFIX = "runs/code"
_BEST_KEY = "best_pipeline.py"
_SUBMISSIONS_PREFIX = "submissions"
_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")
_RUNS_DIR = Path(__file__).parent.parent / "runs"

# 저장된 코드 파일의 헤더와 본문 경계 줄 — cycle/run.py:_save_code가 쓰고 strip_code_header가 떼낸다.
CODE_HEADER_SEP = "# " + "-" * 60


def strip_code_header(content: str) -> str:
    sep = CODE_HEADER_SEP + "\n"
    return (content.split(sep, 1)[1] if sep in content else content).strip()


def _put(path: str, data: bytes, content_type: str | None = None) -> None:
    headers = {"Content-Type": content_type} if content_type else None
    requests.put(f"{_ENDPOINT}/{path}", data=data, headers=headers, timeout=30).raise_for_status()


def _get(path: str) -> bytes:
    """bytes로 돌려준다 — resp.text는 charset 없는 text/*를 ISO-8859-1로 디코드해 비ASCII 소스가 깨진다(#92)."""
    resp = requests.get(f"{_ENDPOINT}/{path}", timeout=30)
    resp.raise_for_status()
    return resp.content


def _delete(path: str) -> None:
    requests.delete(f"{_ENDPOINT}/{path}", timeout=30).raise_for_status()


def upload(competition_id: str, filename: str, content: str) -> str:
    """코드를 저장하고 URI를 반환한다. S3 성공 시 s3:// URI, 실패 시 로컬 경로."""
    key = f"{competition_id}/{_S3_PREFIX}/{filename}"
    try:
        _put(f"{_BUCKET}/{key}", content.encode(), "text/plain")
        return f"s3://{_BUCKET}/{key}"
    except Exception:
        pass
    path = _RUNS_DIR / "code" / competition_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def download(uri: str) -> str | None:
    """URI(s3:// 또는 로컬 경로)로 코드 내용을 반환. 없으면 None."""
    if uri.startswith("s3://"):
        try:
            return _get(uri[len("s3://"):]).decode("utf-8")
        except Exception:
            return None
    path = Path(uri)
    return path.read_text(encoding="utf-8") if path.exists() else None


class BestPipelineUploadError(RuntimeError):
    pass


def _best_local_path(competition_id: str) -> Path:
    return _RUNS_DIR / "best" / f"{competition_id}_best_pipeline.py"


def upload_best_pipeline(competition_id: str, content: str, strict: bool = False) -> str:
    """Materialized best pipeline 저장 → URI 반환.

    DB에 새 sha를 기록하는 쓰기 경로는 insert와 같은 트랜잭션 안에서 strict=True로 부른다 — MinIO 실패를 로컬 폴백으로 삼키면 DB는
    새 sha인데 blob은 옛 내용이라 _baseline_source_guard가 대회를 정지시킨다. 폴백은 MinIO 없이 도는 로컬 개발용이다.
    """
    key = f"{competition_id}/{_BEST_KEY}"
    try:
        _put(f"{_BUCKET}/{key}", content.encode(), "text/plain")
        return f"s3://{_BUCKET}/{key}"
    except Exception as exc:
        if strict:
            raise BestPipelineUploadError(f"best_pipeline upload failed for {competition_id}: {exc}") from exc
    path = _best_local_path(competition_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def download_best_pipeline(competition_id: str) -> str | None:
    """Materialized best pipeline 읽기. 없으면 None."""
    try:
        return _get(f"{_BUCKET}/{competition_id}/{_BEST_KEY}").decode("utf-8")
    except Exception:
        pass
    path = _best_local_path(competition_id)
    return path.read_text(encoding="utf-8") if path.exists() else None


def delete_best_pipeline(competition_id: str) -> bool:
    """competition_id에 귀속된 best_pipeline.py 블롭 삭제. MinIO + 로컬 폴백 모두 처리."""
    deleted = False
    try:
        _delete(f"{_BUCKET}/{competition_id}/{_BEST_KEY}")
        deleted = True
    except Exception:
        pass
    local = _best_local_path(competition_id)
    if local.exists():
        local.unlink()
        deleted = True
    return deleted


def _submission_local_path(competition_id: str, attempt_id: str) -> Path:
    return _RUNS_DIR / "submissions" / competition_id / f"{attempt_id}.csv"


def upload_submission_csv(competition_id: str, attempt_id: str, content: bytes) -> str:
    """promote 시점에 생성한 제출 CSV 캐시.

    auto-submit(매일 06:00)이 이 attempt_id로 캐시 히트하면 fit 없이 그대로 업로드한다
    — 캐시 키에 attempt_id를 넣어 "이 attempt에 대한 예측"임을 명확히 한다.
    """
    key = f"{_SUBMISSIONS_PREFIX}/{competition_id}/{attempt_id}.csv"
    try:
        _put(f"{_BUCKET}/{key}", content, "text/csv")
        return f"s3://{_BUCKET}/{key}"
    except Exception:
        pass
    path = _submission_local_path(competition_id, attempt_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def download_submission_csv(competition_id: str, attempt_id: str) -> bytes | None:
    """캐시된 제출 CSV 읽기. 없으면 None(캐시 미스 — 호출측이 fit 경로로 폴백)."""
    try:
        return _get(f"{_BUCKET}/{_SUBMISSIONS_PREFIX}/{competition_id}/{attempt_id}.csv")
    except Exception:
        pass
    path = _submission_local_path(competition_id, attempt_id)
    return path.read_bytes() if path.exists() else None


def mark_submission_csv_timed_out(competition_id: str, attempt_id: str) -> None:
    """이 attempt의 제출 CSV fit이 wall 상한을 넘었다는 표식(#355). best-effort — 실패하면 다음 promote가 다시 시도한다."""
    try:
        _put(f"{_BUCKET}/{_SUBMISSIONS_PREFIX}/{competition_id}/{attempt_id}.timeout", b"timeout")
    except Exception:
        pass


def submission_csv_timed_out(competition_id: str, attempt_id: str) -> bool:
    url = f"{_ENDPOINT}/{_BUCKET}/{_SUBMISSIONS_PREFIX}/{competition_id}/{attempt_id}.timeout"
    try:
        return requests.get(url, timeout=30).status_code == 200
    except Exception:
        return False


def delete(uri: str) -> bool:
    """URI가 가리키는 파일 삭제. 성공 여부 반환."""
    if uri.startswith("s3://"):
        try:
            _delete(uri[len("s3://"):])
            return True
        except Exception:
            return False
    path = Path(uri)
    if path.exists():
        path.unlink()
        return True
    return False
