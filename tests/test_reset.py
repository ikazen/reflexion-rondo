"""issue #3: bin/reset.py --pipelines-only 회귀 테스트.

근본원인: raw.pipelines 부분 정리(issue #7)를 수동 SQL로 처리해 MinIO
best_pipeline.py가 고아로 남았다. reset_pipelines()는 DB delete와 MinIO delete를
항상 함께 수행해 이 정리 경로를 하나로 강제한다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.reset import main, reset_pipelines


def _conn_with_count(count: int) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (count,)
    return conn


def test_reset_pipelines_deletes_db_rows_and_minio_blob() -> None:
    """행이 있으면 raw.pipelines DELETE + MinIO delete_best_pipeline이 모두 호출된다."""
    conn = _conn_with_count(3)
    with patch("store.db.connect", return_value=conn), \
         patch("store.s3_code.delete_best_pipeline", return_value=True) as mock_delete_blob:
        reset_pipelines("s5e5", yes=True)

    delete_calls = [c for c in conn.execute.call_args_list if "delete" in c.args[0].lower()]
    assert len(delete_calls) == 1
    assert "raw.pipelines" in delete_calls[0].args[0]
    assert delete_calls[0].args[1] == ["s5e5"]
    mock_delete_blob.assert_called_once_with("s5e5")


def test_reset_pipelines_noop_when_no_rows() -> None:
    """행이 0건이면 DELETE도 MinIO 삭제도 호출하지 않는다(불필요한 요청 방지)."""
    conn = _conn_with_count(0)
    with patch("store.db.connect", return_value=conn), \
         patch("store.s3_code.delete_best_pipeline") as mock_delete_blob:
        reset_pipelines("s5e5", yes=True)

    delete_calls = [c for c in conn.execute.call_args_list if "delete" in c.args[0].lower()]
    assert len(delete_calls) == 0
    mock_delete_blob.assert_not_called()


def test_reset_pipelines_requires_competition_flag() -> None:
    """--pipelines-only만 주고 --competition을 빠뜨리면 argparse가 에러로 막는다."""
    with patch("sys.argv", ["reset.py", "--pipelines-only", "--yes"]):
        with pytest.raises(SystemExit):
            main()
