"""cycle.promotion.eval_semantics_fingerprint + cycle.run._eval_fingerprint_guard 단위 테스트 (#348, ADR-053).

데이터가 그대로여도 후보 캡/fold 수/조기중단 같은 평가 노브가 바뀌면 확정 baseline의 cv_score와 새 attempt의
cv_score가 비교 불가능해진다 — 지문이 어긋나면 게이트가 멈춰야 한다.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from bin import establish_baseline
from cycle import promotion
from cycle.promotion import eval_semantics_fingerprint
from cycle.run import EvalFingerprintMismatchError, _eval_fingerprint_guard

ROOT = Path(__file__).parent.parent

_NUMERIC_EXEMPT = {
    "evaluator/harness.py": {
        "_PI_REPEATS": "permutation importance 전용이라 cv_score에 영향 없음",
        "_PI_TOP_N": "permutation importance 전용이라 cv_score에 영향 없음",
        "_LEAK_PERFECT_HIGH": "누수 판정 임계값 — 점수가 아니라 에러 여부를 바꾼다",
        "_LEAK_PERFECT_LOW": "누수 판정 임계값 — 점수가 아니라 에러 여부를 바꾼다",
        "_REGRESSION_IMPLAUSIBLE_BASELINE_RATIO": "비현실 점수 판정 임계값 — 점수가 아니라 에러 여부를 바꾼다",
    },
    "store/train_data.py": {
        "_TWIN_ABORT_FRAC": "twin 중복 비율 초과 시 중단 임계값 — 표본이 아니라 에러 여부를 바꾼다",
    },
}


def _numeric_module_constants(relpath: str) -> set[str]:
    names: set[str] = set()
    for node in ast.parse((ROOT / relpath).read_text()).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if isinstance(node.value.value, bool) or not isinstance(node.value.value, (int, float)):
            continue
        names.update(t.id for t in node.targets if isinstance(t, ast.Name) and re.fullmatch(r"_?[A-Z][A-Z0-9_]*", t.id))
    return names


def test_fingerprint_is_stable():
    assert eval_semantics_fingerprint(5, 42) == eval_semantics_fingerprint(5, 42)


@pytest.mark.parametrize("n_splits, seed", [(3, 42), (5, 7)])
def test_fingerprint_changes_with_fold_count_and_seed(n_splits, seed):
    assert eval_semantics_fingerprint(n_splits, seed) != eval_semantics_fingerprint(5, 42)


_KNOBS = [(mod, name) for mod, names in promotion._EVAL_KNOB_CONSTANTS.items() for name in names]


@pytest.mark.parametrize("mod, name", _KNOBS, ids=[f"{m.__name__}.{n}" for m, n in _KNOBS])
def test_fingerprint_changes_when_each_knob_changes(monkeypatch, mod, name):
    before = eval_semantics_fingerprint(5, 42)
    monkeypatch.setattr(mod, name, getattr(mod, name) + 1)
    assert eval_semantics_fingerprint(5, 42) != before


@pytest.mark.parametrize("relpath", sorted(_NUMERIC_EXEMPT))
def test_every_numeric_constant_is_classified_as_knob_or_exempt(relpath):
    """새 수치 상수가 지문 대상도 예외도 아니면 실패한다 — #311처럼 '점수에 영향 있는 줄 몰랐던' 노브를 막는다."""
    module_name = relpath.removesuffix(".py").replace("/", ".")
    knobs = {n for m, names in promotion._EVAL_KNOB_CONSTANTS.items() if m.__name__ == module_name for n in names}
    classified = knobs | set(_NUMERIC_EXEMPT[relpath])
    defined = _numeric_module_constants(relpath)
    assert defined == classified, (
        f"{relpath}: 미분류 {sorted(defined - classified)} / 존재하지 않는 분류 {sorted(classified - defined)}"
    )


def test_baseline_cache_key_changes_when_eval_knob_changes(monkeypatch):
    kwargs = dict(
        competition_id="c1", best_source="src", train90=pl.DataFrame({"a": [1.0], "y": [0.0]}),
        target_col="y", metric="auc", n_splits=5, is_classification=True,
    )
    before = promotion._eval_context_key(**kwargs)
    monkeypatch.setattr(promotion.harness, "_MAX_PARAM_CANDIDATES", promotion.harness._MAX_PARAM_CANDIDATES + 1)
    assert promotion._eval_context_key(**kwargs) != before


def _conn(stored):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (stored,)
    return conn


def test_guard_stamps_when_unset():
    conn = _conn(None)
    _eval_fingerprint_guard(conn, "s6e8", 5, 42)
    writes = [c for c in conn.execute.call_args_list if "set eval_fingerprint" in c.args[0]]
    assert len(writes) == 1
    assert writes[0].args[1] == [eval_semantics_fingerprint(5, 42), "s6e8"]


def test_guard_passes_silently_when_match():
    conn = _conn(eval_semantics_fingerprint(5, 42))
    _eval_fingerprint_guard(conn, "s6e8", 5, 42)
    assert [c for c in conn.execute.call_args_list if "set eval_fingerprint" in c.args[0]] == []


def test_guard_clears_only_its_own_stale_pause_when_match():
    conn = _conn(eval_semantics_fingerprint(5, 42))
    _eval_fingerprint_guard(conn, "s6e8", 5, 42)
    clears = [c for c in conn.execute.call_args_list if "auto_submit_paused_reason = null" in c.args[0]]
    assert len(clears) == 1
    assert clears[0].args[1] == ["s6e8", "eval_fingerprint 불일치%"]


def test_guard_raises_and_pauses_when_a_knob_changed(monkeypatch):
    conn = _conn(eval_semantics_fingerprint(5, 42))
    monkeypatch.setattr(promotion.harness, "_MAX_PARAM_CANDIDATES", 6)
    with pytest.raises(EvalFingerprintMismatchError):
        _eval_fingerprint_guard(conn, "s6e8", 5, 42)
    pauses = [c for c in conn.execute.call_args_list if "coalesce(auto_submit_paused_reason" in c.args[0]]
    assert len(pauses) == 1
    reason = pauses[0].args[1][0]
    assert reason.startswith("eval_fingerprint 불일치") and "establish_baseline --remeasure --competition s6e8" in reason


def test_remeasure_stamps_eval_fingerprint_and_clears_both_pauses():
    comp = SimpleNamespace(COMPETITION_ID="c1", TARGET="y", METRIC="auc", IS_CLASSIFICATION=True, N_SPLITS=3)
    conn = MagicMock()
    eval_result = SimpleNamespace(cv_score=0.91, fold_scores=[0.9, 0.92], error_trace=None)
    train90 = pl.DataFrame({"a": [1.0, 2.0], "y": [0.0, 1.0]})
    with patch.object(establish_baseline, "_valid_confirmed_pipelines", return_value=[("pid-1", "src", 0.9)]), \
            patch.object(establish_baseline, "load_train", return_value=MagicMock()), \
            patch.object(establish_baseline, "split_audit_holdout", return_value=(train90, train90)), \
            patch.object(establish_baseline, "eval_isolated", return_value=eval_result):
        assert establish_baseline.remeasure_competition(conn, comp, dry_run=False) is True
    calls = [(c.args[0], c.args[1]) for c in conn.execute.call_args_list]
    stamp = [p for s, p in calls if "SET train_fingerprint = %s, eval_fingerprint = %s" in s]
    assert stamp == [[promotion.train_data_fingerprint(train90), eval_semantics_fingerprint(3, 42), "c1"]]
    cleared = [p[1] for s, p in calls if "auto_submit_paused_reason = NULL" in s]
    assert cleared == ["train_fingerprint 불일치%", "eval_fingerprint 불일치%"]
