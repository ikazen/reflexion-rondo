"""promote task uses slug (--competition) for module import, not DB competition_id.
queued promote 경로 버그 — A-1(미확인 시 reflect 전멸).
cross-seed 기준점 버그(A-2)는 paired 비교 도입으로 구조적으로 제거됨 —
confirm_and_measure가 outside prev_best를 아예 받지 않고 seed마다 baseline을
직접 재평가한다(tests/test_promotion_gate.py에서 회귀 검증).
"""
from __future__ import annotations

import contextlib
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

ROOT = Path(__file__).parent.parent


def _fake_comp_module(slug: str, full_id: str) -> types.ModuleType:
    mod = types.ModuleType(f"config.competitions.{slug}")
    mod.COMPETITION_ID = full_id
    mod.S3_DATA_PATH = None
    mod.DATA_DIR = Path("/nonexistent")
    mod.DROP_COLS = []
    mod.TARGET = "target"
    mod.IS_CLASSIFICATION = True
    mod.N_SPLITS = 5
    mod.METRIC = "auc"
    mod.METRIC_SIGN = 1
    return mod


def _run_main(slug: str, monkeypatched_import) -> None:
    from unittest.mock import patch

    argv = ["run_promote_task", "--queue-id", "test-queue-id", "--run-id", "test-run-id", "--competition", slug]
    with patch.object(sys, "argv", argv):
        with patch("store.db.connect") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = None
            mock_conn.return_value.execute.return_value = mock_cur
            with patch("importlib.import_module", side_effect=monkeypatched_import):
                sys.path.insert(0, str(ROOT))
                try:
                    from bin import run_promote_task
                    importlib.reload(run_promote_task)
                    run_promote_task.main()
                except SystemExit:
                    pass
                finally:
                    if str(ROOT) in sys.path:
                        sys.path.remove(str(ROOT))


def test_promote_task_accepts_competition_arg() -> None:
    """--competition 인자가 파싱되어야 한다."""
    import argparse
    sys.path.insert(0, str(ROOT))
    try:
        from bin.run_promote_task import main
        # argparse만 테스트 — 실제 main 실행은 DB 필요
        parser = argparse.ArgumentParser()
        parser.add_argument("--queue-id", required=True)
        parser.add_argument("--run-id", required=True)
        parser.add_argument("--competition", required=True)
        args = parser.parse_args(["--queue-id", "qid", "--run-id", "rid", "--competition", "s4e1"])
        assert args.queue_id == "qid"
        assert args.run_id == "rid"
        assert args.competition == "s4e1"
    finally:
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))


def test_promote_task_imports_by_slug_not_full_id() -> None:
    """import_module이 slug(s4e1)를 쓰고 full competition_id(playground-series-s4e1)를 쓰지 않는다."""
    slug = "s4e1"
    full_id = "playground-series-s4e1"
    imported_modules: list[str] = []

    original_import = importlib.import_module

    def tracking_import(name: str, *args, **kwargs):
        if name.startswith("config.competitions."):
            imported_modules.append(name)
            mod = _fake_comp_module(slug, full_id)
            return mod
        return original_import(name, *args, **kwargs)

    sys.path.insert(0, str(ROOT))
    try:
        argv = ["run_promote_task", "--queue-id", "test-qid", "--run-id", "test-rid", "--competition", slug]
        with patch.object(sys, "argv", argv):
            with patch("importlib.import_module", side_effect=tracking_import):
                with patch("store.db.connect") as mock_conn:
                    mock_cur = MagicMock()
                    # no context found → early exit
                    mock_cur.fetchone.return_value = None
                    mock_conn.return_value.execute.return_value = mock_cur
                    try:
                        import bin.run_promote_task as rpt
                        importlib.reload(rpt)
                        rpt.main()
                    except SystemExit:
                        pass
    finally:
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))

    comp_imports = [m for m in imported_modules if m.startswith("config.competitions.")]
    for mod_name in comp_imports:
        assert full_id not in mod_name, (
            f"import_module called with full competition_id ({full_id}), should use slug ({slug}). "
            f"Got: {mod_name}"
        )
        assert slug in mod_name or not comp_imports, (
            f"Expected slug {slug!r} in import path, got {mod_name!r}"
        )


# --- A-1 / A-2 --------------------------------------------------------------

_FULL_ID = "playground-series-s4e1"
_SLUG = "s4e1"
_WINNER_CV = 0.85

# attempts SELECT 컬럼 순서:
# attempt_id, gain_vs_best, cv_score, label, error_trace,
# hypothesis, action_type, reflection_ids, cv_fold_var, code_path
_ATTEMPT_ROWS = [
    ("w0000000", 0.05, _WINNER_CV, "jump", None, "hyp-w", "model_swap", [], 0.0, "s3://w", None),
    ("l1111111", 0.01, 0.81, "neutral", None, "hyp-1", "feature_engineering", [], 0.0, "s3://l1", None),
    ("l2222222", -0.01, 0.79, "neutral", None, "hyp-2", "preprocessing", [], 0.0, "s3://l2", None),
]


class _Cur:
    def __init__(self, one=None, all_=None):
        self._one, self._all = one, all_

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._all


class _Conn:
    """SQL 텍스트로 라우팅하는 최소 conn mock."""

    def __init__(self):
        self.executed: list[str] = []

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        self.executed.append(s)
        if "FROM raw.super_cycle_context" in s:
            return _Cur(one=("sc-id", _FULL_ID))
        if "FROM raw.attempts" in s and "ORDER BY run_ts" in s:
            return _Cur(all_=_ATTEMPT_ROWS)
        if "task_type, metric from raw.competitions" in s:
            return _Cur(one=("binary", "auc"))
        if "select fingerprint" in s:
            return _Cur(one=({},))
        return _Cur()

    @contextlib.contextmanager
    def transaction(self):
        yield

    def close(self):
        pass


def _run_promote_with_mocks(
    confirm_result, reflect_mock, confirm_mock, eval_isolated_mock=None, generate_csv_mock=None
) -> "_Conn":
    """모든 외부 의존을 mock 처리하고 run_promote_task.main()을 1회 실행. 사용된 conn을 반환.

    conn.insert_pipeline_mock / conn.upload_best_pipeline_mock / conn.eval_isolated_mock /
    conn.generate_csv_mock / conn.upload_submission_csv_mock에 각 mock을 붙여둔다 — 호출
    여부를 검증할 수 있도록(merge-verify, submission 캐싱).
    eval_isolated_mock 미지정 시 winner_cv(_WINNER_CV)와 일치하는 기본 성공 응답으로 채운다.
    generate_csv_mock 미지정 시 (fake_path, winner_attempt_id, _WINNER_CV) 성공 응답으로 채운다.
    """
    fake_df = MagicMock(name="train_df")
    fake_df.drop.return_value = fake_df

    def fake_import(name, *a, **k):
        if name.startswith("config.competitions."):
            return _fake_comp_module(_SLUG, _FULL_ID)
        return importlib.import_module(name, *a, **k)

    confirm_mock.return_value = confirm_result
    conn = _Conn()

    if eval_isolated_mock is None:
        eval_isolated_mock = MagicMock(
            return_value=SimpleNamespace(cv_score=_WINNER_CV, error_trace=None, oof_preds=None)
        )

    fake_csv_path = MagicMock(name="csv_path")
    fake_csv_path.read_bytes.return_value = b"csv-bytes"
    generate_csv_mock = generate_csv_mock or MagicMock(
        return_value=(fake_csv_path, _ATTEMPT_ROWS[0][0], _WINNER_CV)
    )

    argv = ["run_promote_task", "--queue-id", "qid", "--run-id", "rid", "--competition", _SLUG]
    sys.path.insert(0, str(ROOT))
    try:
        with patch.object(sys, "argv", argv), \
             patch("store.db.connect", return_value=conn), \
             patch("store.db.insert_pipeline") as insert_pipeline_mock, \
             patch("store.s3_code.download", return_value="def f():\n    return 1\n"), \
             patch("store.s3_code.download_best_pipeline", return_value=None), \
             patch("store.s3_code.upload_best_pipeline") as upload_best_pipeline_mock, \
             patch("bin.submit.generate_submission_csv", generate_csv_mock), \
             patch("store.s3_code.upload_submission_csv") as upload_submission_csv_mock, \
             patch("cycle.materialize.materialize_best_pipeline", return_value="code"), \
             patch("cycle.promotion.confirm_and_measure", confirm_mock), \
             patch("agents.reflector.reflect", reflect_mock), \
             patch("evaluator.harness.is_significant_gain", return_value=True), \
             patch("evaluator.harness.split_audit_holdout", return_value=(fake_df, fake_df)), \
             patch("runtime.isolate.eval_isolated", eval_isolated_mock), \
             patch("polars.read_csv", return_value=fake_df), \
             patch("importlib.import_module", side_effect=fake_import):
            import bin.run_promote_task as rpt
            importlib.reload(rpt)
            try:
                rpt.main()
            except SystemExit:
                pass
    finally:
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))
    conn.insert_pipeline_mock = insert_pipeline_mock
    conn.upload_best_pipeline_mock = upload_best_pipeline_mock
    conn.eval_isolated_mock = eval_isolated_mock
    conn.generate_csv_mock = generate_csv_mock
    conn.upload_submission_csv_mock = upload_submission_csv_mock
    return conn


def test_a1_reflect_runs_even_when_unconfirmed() -> None:
    """A-1: cross-seed 미확인이어도 모든 attempt에 reflect가 실행돼야 한다."""
    reflect_mock = MagicMock(return_value=SimpleNamespace(reflection_id="rid"))
    confirm_mock = MagicMock()
    _run_promote_with_mocks(
        SimpleNamespace(confirmed=False, holdout_score=None, seed_gains=None),
        reflect_mock,
        confirm_mock,
    )
    # winner(jump) + loser x2 = 3회. 과거엔 return 때문에 0회였다.
    assert reflect_mock.call_count == len(_ATTEMPT_ROWS), (
        f"reflect가 {reflect_mock.call_count}회 호출됨 — 미확인 시에도 {len(_ATTEMPT_ROWS)}회여야 함"
    )


def test_confirm_and_measure_takes_no_outside_prev_best() -> None:
    """confirm_and_measure는 outside prev_best를 받지 않는다 (paired 비교가
    seed마다 baseline을 직접 재평가하므로 winner 자기 CV를 기준점으로 쓰는 경로 자체가 없다).
    """
    reflect_mock = MagicMock(return_value=SimpleNamespace(reflection_id="rid"))
    confirm_mock = MagicMock()
    _run_promote_with_mocks(
        SimpleNamespace(confirmed=False, holdout_score=None, seed_gains=None),
        reflect_mock,
        confirm_mock,
    )
    assert confirm_mock.call_count == 1
    assert "prev_best" not in confirm_mock.call_args.kwargs


def test_super_cycle_context_deleted_after_read() -> None:
    """컨텍스트를 읽은 뒤 해당 run_id 행을 삭제해야 한다."""
    reflect_mock = MagicMock(return_value=SimpleNamespace(reflection_id="rid"))
    confirm_mock = MagicMock()
    conn = _run_promote_with_mocks(
        SimpleNamespace(confirmed=False, holdout_score=None, seed_gains=None),
        reflect_mock,
        confirm_mock,
    )
    delete_calls = [s for s in conn.executed if "DELETE FROM raw.super_cycle_context" in s]
    assert len(delete_calls) == 1


# ---------------------------------------------------------------------------
# merge-verify eval
# ---------------------------------------------------------------------------

def test_merge_verify_matching_cv_allows_promotion() -> None:
    """병합본 cv_score가 winner cv_score와 (허용오차 내) 일치하면 정상 승격된다."""
    reflect_mock = MagicMock(return_value=SimpleNamespace(reflection_id="rid"))
    confirm_mock = MagicMock()
    eval_isolated_mock = MagicMock(
        return_value=SimpleNamespace(cv_score=_WINNER_CV, error_trace=None, oof_preds=[0.1, 0.2, 0.3])
    )
    conn = _run_promote_with_mocks(
        SimpleNamespace(confirmed=True, holdout_score=None, seed_gains=None),
        reflect_mock,
        confirm_mock,
        eval_isolated_mock=eval_isolated_mock,
    )
    assert conn.insert_pipeline_mock.call_count == 1
    assert conn.upload_best_pipeline_mock.call_count == 1


def test_merge_verify_passes_oof_preds_to_insert_pipeline() -> None:
    """merge-verify eval에서 뽑은 oof_preds가 insert_pipeline까지 전달돼야 한다."""
    reflect_mock = MagicMock(return_value=SimpleNamespace(reflection_id="rid"))
    confirm_mock = MagicMock()
    eval_isolated_mock = MagicMock(
        return_value=SimpleNamespace(cv_score=_WINNER_CV, error_trace=None, oof_preds=[0.1, 0.2, 0.3])
    )
    conn = _run_promote_with_mocks(
        SimpleNamespace(confirmed=True, holdout_score=None, seed_gains=None),
        reflect_mock,
        confirm_mock,
        eval_isolated_mock=eval_isolated_mock,
    )
    _, call_kwargs = conn.insert_pipeline_mock.call_args
    assert call_kwargs["oof_preds"] == [0.1, 0.2, 0.3]
    # collect_oof=True로 호출해야 OOF가 실제로 채워진다.
    _, eval_kwargs = conn.eval_isolated_mock.call_args
    assert eval_kwargs["collect_oof"] is True


def test_merge_verify_mismatched_cv_blocks_promotion() -> None:
    """병합본 cv_score가 winner cv_score와 크게 다르면(병합 손상 의심) 승격을 스킵한다."""
    reflect_mock = MagicMock(return_value=SimpleNamespace(reflection_id="rid"))
    confirm_mock = MagicMock()
    eval_isolated_mock = MagicMock(
        return_value=SimpleNamespace(cv_score=_WINNER_CV - 0.3, error_trace=None, oof_preds=None)
    )
    conn = _run_promote_with_mocks(
        SimpleNamespace(confirmed=True, holdout_score=None, seed_gains=None),
        reflect_mock,
        confirm_mock,
        eval_isolated_mock=eval_isolated_mock,
    )
    assert conn.insert_pipeline_mock.call_count == 0
    assert conn.upload_best_pipeline_mock.call_count == 0


def test_merge_verify_eval_error_blocks_promotion() -> None:
    """병합본 평가 자체가 실패하면(예: undefined-name) 승격을 스킵한다."""
    reflect_mock = MagicMock(return_value=SimpleNamespace(reflection_id="rid"))
    confirm_mock = MagicMock()
    eval_isolated_mock = MagicMock(
        return_value=SimpleNamespace(cv_score=None, error_trace="NameError: WeightedEnsemble", oof_preds=None)
    )
    conn = _run_promote_with_mocks(
        SimpleNamespace(confirmed=True, holdout_score=None, seed_gains=None),
        reflect_mock,
        confirm_mock,
        eval_isolated_mock=eval_isolated_mock,
    )
    assert conn.insert_pipeline_mock.call_count == 0
    assert conn.upload_best_pipeline_mock.call_count == 0


# ---------------------------------------------------------------------------
# promote 시점 submission CSV 캐싱 — ops-vm 아침 CPU 스파이크 원인
# 제거(auto-submit이 이 자리에서 fit하는 대신 캐시를 재사용하도록)
# ---------------------------------------------------------------------------

def test_submission_csv_cached_on_successful_promotion() -> None:
    """승격 성공 시 winner attempt_id로 generate_submission_csv → upload_submission_csv가
    호출돼야 한다 — 다음 06:00 auto-submit이 fit 없이 이 캐시를 쓸 수 있도록.
    """
    reflect_mock = MagicMock(return_value=SimpleNamespace(reflection_id="rid"))
    confirm_mock = MagicMock()
    eval_isolated_mock = MagicMock(
        return_value=SimpleNamespace(cv_score=_WINNER_CV, error_trace=None, oof_preds=None)
    )
    conn = _run_promote_with_mocks(
        SimpleNamespace(confirmed=True, holdout_score=None, seed_gains=None),
        reflect_mock,
        confirm_mock,
        eval_isolated_mock=eval_isolated_mock,
    )
    winner_attempt_id = _ATTEMPT_ROWS[0][0]
    conn.generate_csv_mock.assert_called_once_with(_SLUG, attempt_id=winner_attempt_id)
    conn.upload_submission_csv_mock.assert_called_once_with(_FULL_ID, winner_attempt_id, b"csv-bytes")


def test_submission_csv_caching_failure_does_not_block_promotion() -> None:
    """캐싱이 실패해도(best-effort) 승격 자체는 정상 진행돼야 한다 — 캐시 미스 시
    auto-submit이 기존 fit 경로로 폴백하므로 안전.
    """
    reflect_mock = MagicMock(return_value=SimpleNamespace(reflection_id="rid"))
    confirm_mock = MagicMock()
    eval_isolated_mock = MagicMock(
        return_value=SimpleNamespace(cv_score=_WINNER_CV, error_trace=None, oof_preds=None)
    )
    failing_generate_csv_mock = MagicMock(side_effect=RuntimeError("train data unavailable"))
    conn = _run_promote_with_mocks(
        SimpleNamespace(confirmed=True, holdout_score=None, seed_gains=None),
        reflect_mock,
        confirm_mock,
        eval_isolated_mock=eval_isolated_mock,
        generate_csv_mock=failing_generate_csv_mock,
    )
    assert conn.insert_pipeline_mock.call_count == 1
    assert conn.upload_best_pipeline_mock.call_count == 1
    assert conn.upload_submission_csv_mock.call_count == 0
