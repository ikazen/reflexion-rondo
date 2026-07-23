"""submission CSV 캐시 헬퍼 — 로컬 폴백 round-trip.

MINIO_ENDPOINT 미설정 테스트 환경에서는 upload/download가 항상 로컬
runs/submissions/{competition_id}/{attempt_id}.csv 폴백을 탄다(네트워크 요청은
잘못된 URL로 실패 → except → 로컬 write/read). 이 경로만 검증한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from store.s3_code import download_submission_csv, upload_submission_csv


def test_upload_then_download_roundtrips_local_fallback() -> None:
    competition_id = "test-comp-31"
    attempt_id = "attempt-abc"
    content = b"id,target\n1,0.5\n2,0.7\n"
    local_path = ROOT / "runs" / "submissions" / competition_id / f"{attempt_id}.csv"
    try:
        uri = upload_submission_csv(competition_id, attempt_id, content)
        assert uri.endswith(f"{competition_id}/{attempt_id}.csv") or Path(uri) == local_path
        result = download_submission_csv(competition_id, attempt_id)
        assert result == content
    finally:
        if local_path.exists():
            local_path.unlink()
            local_path.parent.rmdir()


def test_download_missing_returns_none() -> None:
    assert download_submission_csv("test-comp-31-missing", "no-such-attempt") is None
