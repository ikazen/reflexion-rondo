"""bin/backfill_materialized_code.py — #254 행동 검증 백필.

verdict 사다리(sha/cv/chain/unverifiable)와 그 부작용(materialized_code/
materialized_sha256/materialized_origin/invalid_reason 쓰기)을 검증한다. replay/
eval/train IO는 전부 monkeypatch — 순수 판정 로직만 본다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl

import bin.backfill_materialized_code as bm
from bin.backfill_materialized_code import ChainRow, backfill_competition


class _Comp:
    COMPETITION_ID = "playground-series-s4e10"
    TARGET = "y"
    METRIC = "auc"
    METRIC_SIGN = 1
    IS_CLASSIFICATION = True


_TRAIN = pl.DataFrame({"x": [1.0, 2.0], "y": [0.0, 1.0]})


def _row(pid, *, cv=0.9, sha=None, mcode=None, msha=None, origin=None, invalid=None):
    return ChainRow(
        pipeline_id=pid, attempt_id=f"a-{pid}", run_ts=pid, code=f"code-{pid}",
        cv_score=cv, pipeline_sha256=sha, materialized_code=mcode,
        materialized_sha256=msha, materialized_origin=origin, invalid_reason=invalid,
    )


def _eval(cv, folds=None):
    return MagicMock(error_trace=None, cv_score=cv, fold_scores=folds or [cv, cv], cv_fold_var=0.0001)


def _run(chain, *, eval_map=None, replay_map=None, drift=None, allow_chain=False, remeasure=False):
    """chain을 백필. eval_map: {source_text: IsolatedResult}, replay_map: {pipeline_id: candidate_src}."""
    conn = MagicMock()
    eval_map = eval_map or {}
    replay_map = replay_map or {}

    def _fake_eval(comp, source, train90, cpu_budget):
        return eval_map.get(source, _eval(0.9))

    def _fake_replay(c, cid, strict_sha=False, stop_at_pipeline_id=None, **kw):
        return replay_map.get(stop_at_pipeline_id, f"replayed-{stop_at_pipeline_id}"), None, 1

    with patch("cycle.materialize.promotion_chain",
               return_value=[tuple(getattr(r, f) for f in ChainRow.__slots__) for r in chain]), \
         patch("store.train_data.load_train", return_value=_TRAIN), \
         patch("evaluator.harness.split_audit_holdout", return_value=(_TRAIN, _TRAIN)), \
         patch("bin.backfill_materialized_code._eval_base", side_effect=_fake_eval), \
         patch("bin.backfill_materialized_code._drift_probe", return_value=drift), \
         patch("cycle.materialize.replay_best_pipeline", side_effect=_fake_replay):
        tally = backfill_competition(conn, _Comp(), apply=True, allow_chain=allow_chain, remeasure=remeasure)
    updates = [c for c in conn.execute.call_args_list if "UPDATE raw.pipelines" in c.args[0]]
    return tally, updates


def _update_for(updates, pid_prefix):
    for c in updates:
        if c.args[1][-1].startswith(pid_prefix) or c.args[1][-1] == pid_prefix:
            return c
    return None


# --- import 배선 ---

def test_promotion_chain_and_replay_stop_are_importable():
    from cycle.materialize import promotion_chain, replay_best_pipeline
    import inspect
    assert "stop_at_pipeline_id" in inspect.signature(replay_best_pipeline).parameters
    assert callable(promotion_chain)


# --- tier: sha exact ---

def test_replay_sha_exact_writes_snapshot_without_cv_tier():
    import hashlib
    cand = "replayed-p1"
    sha = hashlib.sha256(cand.encode()).hexdigest()
    chain = [_row("p1", sha=sha)]
    tally, updates = _run(chain, replay_map={"p1": cand})
    assert tally == {"backfill:sha": 1}
    u = updates[0]
    assert "materialized_code = %s" in u.args[0]
    assert "materialized_sha256 = %s" in u.args[0]
    assert "invalid_reason" not in u.args[0]


# --- tier: cv (drift probe 통과) ---

def test_cv_within_tolerance_writes_snapshot():
    chain = [_row("p1", cv=0.9123, sha="x")]
    tally, updates = _run(
        chain, drift=True, replay_map={"p1": "cand1"},
        eval_map={"cand1": _eval(0.9123)},
    )
    assert tally == {"backfill:cv": 1}
    assert "invalid_reason" not in updates[0].args[0]


def test_cv_mismatch_with_comparable_data_quarantines():
    chain = [_row("p1", cv=0.9123, sha="x")]
    tally, updates = _run(
        chain, drift=True, replay_map={"p1": "cand1"},
        eval_map={"cand1": _eval(0.8000)},
    )
    assert tally == {"unverifiable:cv_mismatch": 1}
    u = updates[0]
    assert "invalid_reason = coalesce(invalid_reason, %s)" in u.args[0]
    assert "materialize_unreproducible" in u.args[1][1]
    assert "materialized_code = %s" not in u.args[0]


# --- drift probe 실패 → cv tier 비활성, 격리 안 함 (핵심 회귀 테스트) ---

def test_drift_probe_failure_never_quarantines_and_marks_train_drift():
    chain = [_row("p1", cv=0.9123, sha="x"), _row("p2", cv=0.9200, sha="y")]
    tally, updates = _run(
        chain, drift=False,  # 데이터 이동됨
        replay_map={"p1": "c1", "p2": "c2"},
        eval_map={"c1": _eval(0.80), "c2": _eval(0.79)},
        allow_chain=False,
    )
    assert set(tally) == {"unverifiable:train_drift"}
    assert all("invalid_reason" not in u.args[0] for u in updates)
    assert all("materialized_code = %s" not in u.args[0] for u in updates)


# --- tier: chain (--allow-chain) ---

def test_chain_tier_accepts_when_not_significantly_worse():
    chain = [_row("p1", cv=0.90, sha="x"), _row("p2", cv=0.91, sha="y")]
    tally, updates = _run(
        chain, drift=False, allow_chain=True,
        replay_map={"p1": "c1", "p2": "c2"},
        eval_map={"c1": _eval(0.80, [0.80, 0.80]), "c2": _eval(0.801, [0.801, 0.801])},
    )
    assert tally == {"backfill:chain": 2}
    assert all("materialized_code = %s" in u.args[0] for u in updates)


def test_chain_tier_rejects_significant_regression():
    chain = [_row("p1", cv=0.90, sha="x"), _row("p2", cv=0.91, sha="y")]
    tally, _ = _run(
        chain, drift=False, allow_chain=True,
        replay_map={"p1": "c1", "p2": "c2"},
        eval_map={
            "c1": _eval(0.80, [0.80, 0.80, 0.80, 0.80, 0.80]),
            "c2": _eval(0.50, [0.50, 0.50, 0.50, 0.50, 0.50]),
        },
    )
    assert tally.get("backfill:chain") == 1
    assert tally.get("unverifiable:chain_regression") == 1


def test_remeasure_updates_cv_only_with_flag():
    chain = [_row("p1", cv=0.9123, sha="x")]
    _, updates_no = _run(chain, drift=True, replay_map={"p1": "c1"}, eval_map={"c1": _eval(0.9123)}, remeasure=False)
    assert "cv_score = %s" not in updates_no[0].args[0]
    _, updates_yes = _run(chain, drift=True, replay_map={"p1": "c1"}, eval_map={"c1": _eval(0.9123)}, remeasure=True)
    assert "cv_score = %s" in updates_yes[0].args[0]


# --- 기존 스냅샷 행 ---

def test_existing_snapshot_row_left_alone_but_sha_backfilled():
    import hashlib
    code = "class Patch: pass"
    chain = [_row("p1", mcode=code, msha=None, sha=None, origin=None)]
    tally, updates = _run(chain)
    assert tally == {"promote": 1}
    u = updates[0]
    assert hashlib.sha256(code.encode()).hexdigest() in u.args[1]


def test_corrupt_existing_snapshot_quarantines():
    chain = [_row("p1", mcode="tampered", msha="expected_hash_that_wont_match")]
    tally, updates = _run(chain)
    assert tally == {"unverifiable:snapshot_corrupt": 1}
    assert "materialize_unreproducible" in updates[0].args[1][1]


def test_already_quarantined_row_is_skipped():
    chain = [_row("p1", invalid="target_leak_preprocess: foo")]
    tally, updates = _run(chain)
    assert tally == {"already_invalid": 1}
    assert updates == []


# --- dry-run ---

def test_dry_run_writes_nothing():
    conn = MagicMock()
    chain = [_row("p1", cv=0.9, sha="x")]
    with patch("cycle.materialize.promotion_chain",
               return_value=[tuple(getattr(r, f) for f in ChainRow.__slots__) for r in chain]), \
         patch("store.train_data.load_train", return_value=_TRAIN), \
         patch("evaluator.harness.split_audit_holdout", return_value=(_TRAIN, _TRAIN)), \
         patch("bin.backfill_materialized_code._eval_base", return_value=_eval(0.9)), \
         patch("bin.backfill_materialized_code._drift_probe", return_value=True), \
         patch("cycle.materialize.replay_best_pipeline", return_value=("cand", None, 1)):
        backfill_competition(conn, _Comp(), apply=False, allow_chain=True, remeasure=True)
    assert not [c for c in conn.execute.call_args_list if "UPDATE" in c.args[0]]
