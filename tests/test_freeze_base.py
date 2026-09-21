"""bin.freeze_base.freeze_latest — 동결본이 저장된 cv를 재현할 때만 MinIO/DB에 반영하는지 검증."""
from __future__ import annotations

import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bin import freeze_base
from cycle.materialize import materialize_best_pipeline, with_frozen_params

_ROUND1 = textwrap.dedent("""
    class Patch:
        action_type = "hyperparam_search"
        changed_stages = ["param_candidates"]
        rationale = "round1"

        def param_candidates(self, ctx):
            return [{"lr": 0.1}, {"lr": 0.2}]
""").strip()

_ROUND2 = textwrap.dedent("""
    class Patch:
        action_type = "hyperparam_search"
        changed_stages = ["param_candidates"]
        rationale = "round2"

        def param_candidates(self, ctx):
            return [{"lr": 0.4}, {"lr": 0.5}]
""").strip()

_PARAMS = {"lr": 0.4}
_STORED_CV = 0.9662104086
_COMP = SimpleNamespace(COMPETITION_ID="c1", TARGET="y", IS_CLASSIFICATION=True, METRIC="auc")


class _Conn:
    def __init__(self, row, fail_update: bool = False) -> None:
        self.row = row
        self.fail_update = fail_update
        self.updates: list[list] = []

    def execute(self, sql, params=None):
        if sql.lstrip().startswith("UPDATE"):
            if self.fail_update:
                raise RuntimeError("db down")
            self.updates.append(params)
            return MagicMock()
        return SimpleNamespace(fetchone=lambda: self.row)


def _row(materialized: str, params=_PARAMS):
    return ("pid-0001", _ROUND2, _STORED_CV, materialized, params)


def _run(conn, eval_cv, dry_run=False, blob_reflects_upload=True, uploads=None):
    uploads = [] if uploads is None else uploads
    result = SimpleNamespace(cv_score=eval_cv, error_trace=None)
    with patch.object(freeze_base, "load_train", return_value=MagicMock()), \
            patch.object(freeze_base, "split_audit_holdout", return_value=(MagicMock(), MagicMock())), \
            patch.object(freeze_base, "eval_isolated", return_value=result) as ev, \
            patch.object(freeze_base, "upload_best_pipeline", side_effect=lambda cid, s: uploads.append(s)), \
            patch.object(freeze_base, "download_best_pipeline",
                         side_effect=lambda cid: uploads[-1] if blob_reflects_upload else "old blob"):
        done = freeze_base.freeze_latest(conn, _COMP, dry_run)
    return done, ev, uploads


def test_freeze_applies_when_frozen_cv_matches_stored():
    materialized = materialize_best_pipeline(_ROUND1, _ROUND2)
    conn = _Conn(_row(materialized))
    done, ev, uploads = _run(conn, _STORED_CV)
    assert done is True
    frozen = uploads[-1]
    ns: dict = {}
    exec(compile(frozen, "<test>", "exec"), ns)  # noqa: S102
    assert ns["Patch"]().param_candidates(None) == [_PARAMS]
    assert ev.call_args.kwargs["source"] == frozen
    (code, mat, sha1, sha2, pid), = conn.updates
    assert mat == frozen and sha1 == sha2 and pid == "pid-0001"
    assert code == with_frozen_params(_ROUND2, _PARAMS)


def test_freeze_dry_run_writes_nothing():
    conn = _Conn(_row(materialize_best_pipeline(_ROUND1, _ROUND2)))
    done, _, uploads = _run(conn, _STORED_CV, dry_run=True)
    assert done is True
    assert uploads == []
    assert conn.updates == []


def test_freeze_aborts_when_frozen_cv_differs():
    conn = _Conn(_row(materialize_best_pipeline(_ROUND1, _ROUND2)))
    done, _, uploads = _run(conn, _STORED_CV - 1e-3)
    assert done is False
    assert uploads == []
    assert conn.updates == []


def test_freeze_skips_already_frozen_without_evaluating():
    frozen = materialize_best_pipeline(_ROUND1, with_frozen_params(_ROUND2, _PARAMS))
    done, ev, uploads = _run(_Conn(_row(frozen)), _STORED_CV)
    assert done is False
    ev.assert_not_called()
    assert uploads == []


@pytest.mark.parametrize("params", [None, {}])
def test_freeze_skips_without_selected_params(params):
    done, ev, _ = _run(_Conn(_row(materialize_best_pipeline(_ROUND1, _ROUND2), params)), _STORED_CV)
    assert done is False
    ev.assert_not_called()


def test_freeze_leaves_db_untouched_when_upload_did_not_land():
    """upload_best_pipeline은 MinIO 실패 시 예외 없이 로컬 폴백하므로, 재조회로 확인하지 못하면 DB를 갱신하면 안 된다."""
    conn = _Conn(_row(materialize_best_pipeline(_ROUND1, _ROUND2)))
    done, _, _ = _run(conn, _STORED_CV, blob_reflects_upload=False)
    assert done is False
    assert conn.updates == []


def test_freeze_restores_old_blob_when_db_update_fails():
    materialized = materialize_best_pipeline(_ROUND1, _ROUND2)
    conn = _Conn(_row(materialized), fail_update=True)
    uploads: list[str] = []
    with pytest.raises(RuntimeError, match="db down"):
        _run(conn, _STORED_CV, uploads=uploads)
    assert len(uploads) == 2 and uploads[1] == materialized and uploads[0] != materialized
