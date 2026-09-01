"""bin/establish_baseline.py 단위 테스트 (#101).

top-k 후보를 순회하며 첫 confirmed 후보만 승격하는 배선을 검증한다 —
confirm_and_measure/materialize_best_pipeline/S3 IO/insert_pipeline은 전부
monkeypatch해 순수 로직만 확인한다. 실제 confirm 로직 자체(cross-seed/holdout
게이트)는 tests/test_promotion_gate.py가 이미 커버한다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl

from bin.establish_baseline import (
    _top_k_attempts,
    _valid_confirmed_pipelines,
    competitions_without_baseline,
    establish_for_competition,
    remeasure_competition,
)
from cycle.promotion import ConfirmResult


class _Comp:
    COMPETITION_ID = "playground-series-s5e7"
    TARGET = "y"
    METRIC = "auc"
    IS_CLASSIFICATION = True


_TRAIN = pl.DataFrame({"x": [1.0, 2.0], "y": [0.0, 1.0]})


def _patches(confirm_results, code_by_attempt=None):
    """confirm_and_measure를 attempt마다 순서대로 다른 결과로 반환하도록 구성."""
    code_by_attempt = code_by_attempt or {}

    def _confirm_side_effect(*args, **kwargs):
        return confirm_results.pop(0)

    return (
        patch("bin.establish_baseline.load_train", return_value=_TRAIN),
        patch("bin.establish_baseline.split_audit_holdout", return_value=(_TRAIN, _TRAIN)),
        patch("bin.establish_baseline._code_download", side_effect=lambda path: code_by_attempt.get(path, "code")),
        patch("bin.establish_baseline.confirm_and_measure", side_effect=_confirm_side_effect),
        patch("bin.establish_baseline.materialize_best_pipeline", return_value="materialized"),
        # eval_isolated는 실제 서브프로세스를 띄우므로(#145 merge-verify OOF 수집)
        # 다른 IO와 마찬가지로 monkeypatch — 순수 로직만 확인한다는 이 파일의
        # 설계 원칙 유지.
        patch("bin.establish_baseline.eval_isolated", return_value=MagicMock(error_trace=None, cv_score=0.9, oof_preds=[0.1, 0.2])),
        patch("bin.establish_baseline.insert_pipeline"),
        patch("bin.establish_baseline.upload_best_pipeline"),
    )


def test_competitions_without_baseline_extracts_ids():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [("s4e2",), ("s5e7",)]
    result = competitions_without_baseline(conn)
    assert result == ["s4e2", "s5e7"]
    sql = conn.execute.call_args.args[0]
    assert "NOT EXISTS" in sql
    assert "invalid_reason IS NULL" in sql


def test_top_k_attempts_passes_limit_param():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    _top_k_attempts(conn, "s4e1", 7)
    params = conn.execute.call_args.args[1]
    assert params == ["s4e1", 7]


def test_no_candidates_returns_none():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    result = establish_for_competition(conn, _Comp(), top_k=5, dry_run=False)
    assert result is None


def test_first_candidate_confirmed_promotes_and_stops():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("attempt-1", 0.9, "path1", [0.9]),
        ("attempt-2", 0.85, "path2", [0.85]),
    ]
    conn.execute.return_value.fetchone.return_value = (None,)
    patches = _patches([ConfirmResult(confirmed=True, holdout_score=0.88, seed_gains=None)])
    with patches[0], patches[1], patches[2], patches[3] as mock_confirm, patches[4], patches[5], patches[6] as mock_insert, patches[7] as mock_upload:
        result = establish_for_competition(conn, _Comp(), top_k=5, dry_run=False)

    assert result == "attempt-1"
    assert mock_confirm.call_count == 1  # 두 번째 후보는 시도하지 않음
    mock_insert.assert_called_once()
    mock_upload.assert_called_once_with("playground-series-s5e7", "materialized")


def test_first_candidate_fails_falls_back_to_second():
    """phantom(첫 후보)이 게이트를 통과 못 하면 다음 순위로 넘어가야 한다."""
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("attempt-1", 0.99, "path1", [0.99]),  # phantom — 재현 안 됨
        ("attempt-2", 0.85, "path2", [0.85]),  # 실제 confirm됨
    ]
    conn.execute.return_value.fetchone.return_value = (None,)
    patches = _patches([
        ConfirmResult(confirmed=False, holdout_score=None, seed_gains=None),
        ConfirmResult(confirmed=True, holdout_score=0.84, seed_gains=None),
    ])
    with patches[0], patches[1], patches[2], patches[3] as mock_confirm, patches[4], patches[5], patches[6] as mock_insert, patches[7]:
        result = establish_for_competition(conn, _Comp(), top_k=5, dry_run=False)

    assert result == "attempt-2"
    assert mock_confirm.call_count == 2
    mock_insert.assert_called_once()
    assert mock_insert.call_args.kwargs["attempt_id"] == "attempt-2"
    assert mock_insert.call_args.kwargs["cv_score"] == 0.85


def test_all_candidates_fail_returns_none():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("attempt-1", 0.99, "path1", [0.99]),
        ("attempt-2", 0.95, "path2", [0.95]),
    ]
    patches = _patches([
        ConfirmResult(confirmed=False, holdout_score=None, seed_gains=None),
        ConfirmResult(confirmed=False, holdout_score=None, seed_gains=None),
    ])
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6] as mock_insert, patches[7]:
        result = establish_for_competition(conn, _Comp(), top_k=5, dry_run=False)

    assert result is None
    mock_insert.assert_not_called()


def test_dry_run_does_not_write_but_returns_attempt_id():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [("attempt-1", 0.9, "path1", [0.9])]
    patches = _patches([ConfirmResult(confirmed=True, holdout_score=0.88, seed_gains=None)])
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6] as mock_insert, patches[7] as mock_upload:
        result = establish_for_competition(conn, _Comp(), top_k=5, dry_run=True)

    assert result == "attempt-1"
    mock_insert.assert_not_called()
    mock_upload.assert_not_called()


def test_empty_source_after_header_strip_skips_candidate():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("attempt-1", 0.9, "path1", [0.9]),
        ("attempt-2", 0.85, "path2", [0.85]),
    ]
    conn.execute.return_value.fetchone.return_value = (None,)
    patches_list = list(_patches([ConfirmResult(confirmed=True, holdout_score=0.8, seed_gains=None)],
                                  code_by_attempt={"path1": "", "path2": "code"}))
    with patches_list[0], patches_list[1], patches_list[2], patches_list[3] as mock_confirm, \
         patches_list[4], patches_list[5], patches_list[6], patches_list[7]:
        result = establish_for_competition(conn, _Comp(), top_k=5, dry_run=False)

    assert result == "attempt-2"
    assert mock_confirm.call_count == 1  # attempt-1은 코드가 없어 confirm 자체를 안 부름


# --- remeasure_competition: load_train 설정 변경 후 확정 pipeline 전체 재측정 (#135/#258/#262) ---

def _remeasure_conn(rows):
    """_valid_confirmed_pipelines(fetchall)만 데이터를 돌려주는 conn mock."""
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows
    conn.execute.return_value.fetchone.return_value = None
    return conn


def test_valid_confirmed_pipelines_excludes_unverifiable_and_invalid():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    _valid_confirmed_pipelines(conn, "s4e12")
    sql = conn.execute.call_args.args[0]
    assert "invalid_reason IS NULL" in sql
    assert "unverifiable:%%" in sql
    assert "materialized_code IS NOT NULL" in sql


def test_remeasure_no_confirmed_pipeline_returns_false():
    conn = _remeasure_conn([])
    with (
        patch("bin.establish_baseline.load_train", return_value=_TRAIN),
        patch("bin.establish_baseline.split_audit_holdout", return_value=(_TRAIN, _TRAIN)),
    ):
        assert remeasure_competition(conn, _Comp(), dry_run=False) is False


def test_remeasure_updates_all_confirmed_and_stamps_fingerprint():
    conn = _remeasure_conn([("pipe-1", "code A", 0.7), ("pipe-2", "code B", 0.6)])

    with (
        patch("bin.establish_baseline.load_train", return_value=_TRAIN),
        patch("bin.establish_baseline.split_audit_holdout", return_value=(_TRAIN, _TRAIN)),
        patch("bin.establish_baseline.eval_isolated",
              return_value=MagicMock(error_trace=None, cv_score=0.55)) as mock_eval,
    ):
        result = remeasure_competition(conn, _Comp(), dry_run=False)

    assert result is True
    assert mock_eval.call_count == 2  # #262: 최고점 1건이 아니라 전체
    pipe_updates = [c for c in conn.execute.call_args_list if "UPDATE raw.pipelines SET cv_score" in c.args[0]]
    assert {tuple(c.args[1]) for c in pipe_updates} == {(0.55, "pipe-1"), (0.55, "pipe-2")}
    fp_updates = [c for c in conn.execute.call_args_list if "SET train_fingerprint" in c.args[0]]
    assert len(fp_updates) == 1


def test_remeasure_dry_run_does_not_write():
    conn = _remeasure_conn([("pipe-1", "code", 0.7)])

    with (
        patch("bin.establish_baseline.load_train", return_value=_TRAIN),
        patch("bin.establish_baseline.split_audit_holdout", return_value=(_TRAIN, _TRAIN)),
        patch("bin.establish_baseline.eval_isolated",
              return_value=MagicMock(error_trace=None, cv_score=0.55)),
    ):
        result = remeasure_competition(conn, _Comp(), dry_run=True)

    assert result is True
    writes = [c for c in conn.execute.call_args_list
              if c.args[0].strip().upper().startswith("UPDATE")]
    assert writes == []


def test_remeasure_eval_failure_quarantines_and_skips_fingerprint():
    conn = _remeasure_conn([("pipe-1", "code", 0.7)])

    with (
        patch("bin.establish_baseline.load_train", return_value=_TRAIN),
        patch("bin.establish_baseline.split_audit_holdout", return_value=(_TRAIN, _TRAIN)),
        patch("bin.establish_baseline.eval_isolated",
              return_value=MagicMock(error_trace="boom", cv_score=None)),
    ):
        result = remeasure_competition(conn, _Comp(), dry_run=False)

    assert result is False  # 재측정 성공 0건 → 지문 미갱신
    quarantine = [c for c in conn.execute.call_args_list if "remeasure_failed:train_fingerprint" in c.args[0]]
    assert len(quarantine) == 1 and quarantine[0].args[1] == ["pipe-1"]
    fp_updates = [c for c in conn.execute.call_args_list if "SET train_fingerprint" in c.args[0]]
    assert fp_updates == []


def test_remeasure_prints_rebuild_hint_when_any_quarantined(capsys):
    """격리로 유효집합이 바뀌면 MinIO 재구성 안내를 출력한다 (#278)."""
    conn = _remeasure_conn([("pipe-ok", "code A", 0.7), ("pipe-bad", "code B", 0.6)])

    with (
        patch("bin.establish_baseline.load_train", return_value=_TRAIN),
        patch("bin.establish_baseline.split_audit_holdout", return_value=(_TRAIN, _TRAIN)),
        patch("bin.establish_baseline.eval_isolated", side_effect=[
            MagicMock(error_trace=None, cv_score=0.55),
            MagicMock(error_trace="boom", cv_score=None),
        ]),
    ):
        result = remeasure_competition(conn, _Comp(), dry_run=False)

    assert result is True
    assert "rebuild_best_pipeline --competition" in capsys.readouterr().out
