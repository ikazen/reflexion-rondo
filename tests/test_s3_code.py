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

from unittest.mock import MagicMock, patch

from store import s3_code
from store.s3_code import (
    download_submission_csv,
    mark_submission_csv_timed_out,
    submission_csv_timed_out,
    upload_submission_csv,
)


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


def test_download_decodes_utf8_regardless_of_content_type_charset() -> None:
    """#92: charset 없는 text/plain을 requests가 ISO-8859-1로 디코드해 비ASCII
    소스가 mojibake 되던 회귀 — sha256 검증(승격/제출/백필)이 전부 어긋난다."""
    from unittest.mock import MagicMock, patch

    source = "class Patch:\n    pass\n# 한글 주석 — mojibake 검증\n"
    resp = MagicMock()
    resp.content = source.encode("utf-8")
    resp.raise_for_status.return_value = None
    # 실제 requests처럼 charset 부재 시 latin-1로 잘못 디코드된 .text를 흉내낸다
    resp.text = source.encode("utf-8").decode("iso-8859-1")

    from store import s3_code

    with patch.object(s3_code.requests, "get", return_value=resp):
        assert s3_code.download("s3://bucket/key.py") == source
        assert s3_code.download_best_pipeline("comp-x") == source


def test_timeout_marker_is_stored_next_to_the_csv_cache_key() -> None:
    """#355: 표식은 CSV 캐시와 같은 접두(submissions/{competition_id}/) 아래 `{attempt_id}.timeout`이다."""
    resp = MagicMock(status_code=200)
    with patch.object(s3_code.requests, "put", return_value=resp) as put:
        mark_submission_csv_timed_out("playground-series-s5e4", "attempt-abc")
    assert put.call_args.args[0].endswith("/kaggle/submissions/playground-series-s5e4/attempt-abc.timeout")


def test_timeout_marker_lookup_reflects_the_object_existing() -> None:
    with patch.object(s3_code.requests, "get", return_value=MagicMock(status_code=200)) as get:
        assert submission_csv_timed_out("playground-series-s5e4", "attempt-abc") is True
    assert get.call_args.args[0].endswith("/kaggle/submissions/playground-series-s5e4/attempt-abc.timeout")
    for status in (404, 403, 500):  # 표식을 확인하지 못한 응답은 모두 "표식 없음"이다(fail-open)
        with patch.object(s3_code.requests, "get", return_value=MagicMock(status_code=status)):
            assert submission_csv_timed_out("playground-series-s5e4", "attempt-abc") is False


def test_timeout_marker_helpers_are_best_effort() -> None:
    """MinIO 장애로 표식을 못 읽거나 쓰면 fit을 막지도, promote를 죽이지도 않는다(읽기는 없음으로 취급)."""
    with patch.object(s3_code.requests, "get", side_effect=ConnectionError("down")):
        assert submission_csv_timed_out("playground-series-s5e4", "attempt-abc") is False
    with patch.object(s3_code.requests, "put", side_effect=ConnectionError("down")):
        mark_submission_csv_timed_out("playground-series-s5e4", "attempt-abc")
