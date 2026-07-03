"""bin/run_retrieve_task.py — _merge_lessons 단위 테스트 (BON-134),
main() TTL sweep 회귀 테스트 (BON-242)."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.run_retrieve_task import _merge_lessons


def _l(rid: str, lesson_type: str = "recommend") -> dict:
    return {"reflection_id": rid, "embedded_text": rid, "full_lesson": rid,
            "generality": "L1_local", "gain_vs_best": 0.0, "lesson_type": lesson_type}


def test_merge_appends_new_failure_lessons():
    lessons = [_l("a"), _l("b")]
    extra = [_l("c", "failure")]
    merged = _merge_lessons(lessons, extra)
    assert [l["reflection_id"] for l in merged] == ["a", "b", "c"]


def test_merge_dedupes_by_reflection_id_keeping_original():
    lessons = [_l("a"), _l("b", "recommend")]
    extra = [_l("b", "failure"), _l("c", "failure")]
    merged = _merge_lessons(lessons, extra)
    ids = [l["reflection_id"] for l in merged]
    assert ids == ["a", "b", "c"]
    # 중복은 원래(lessons) 쪽 값 유지
    assert next(l for l in merged if l["reflection_id"] == "b")["lesson_type"] == "recommend"


def test_merge_empty_extra_is_noop():
    lessons = [_l("a")]
    assert _merge_lessons(lessons, []) == lessons


def test_merge_empty_lessons_returns_extra():
    extra = [_l("a", "failure")]
    assert _merge_lessons([], extra) == extra


_SLUG = "s4e1"
_FULL_ID = "playground-series-s4e1"


def _fake_comp_module() -> types.ModuleType:
    mod = types.ModuleType(f"config.competitions.{_SLUG}")
    mod.COMPETITION_ID = _FULL_ID
    mod.NAME = "s4e1"
    mod.TASK_TYPE = "binary"
    mod.METRIC = "auc"
    mod.METRIC_SIGN = 1
    mod.EDA_CARD = "eda"
    return mod


class _Conn:
    """execute 호출만 기록하는 최소 conn mock."""

    def __init__(self):
        self.executed: list[str] = []

    def execute(self, sql, params=None):
        self.executed.append(" ".join(sql.split()))
        return MagicMock()

    def close(self):
        pass


def _run_retrieve_with_mocks() -> "_Conn":
    """모든 외부 의존을 mock 처리하고 run_retrieve_task.main()을 1회 실행. 사용된 conn을 반환."""

    def fake_import(name, *a, **k):
        if name.startswith("config.competitions."):
            return _fake_comp_module()
        return importlib.import_module(name, *a, **k)

    conn = _Conn()
    argv = ["run_retrieve_task", "--competition", _SLUG, "--stage", "reflexion",
            "--queue-id", "qid", "--run-id", "rid"]
    try:
        with patch.object(sys, "argv", argv), \
             patch("store.db.connect", return_value=conn), \
             patch("store.db.ensure_competition"), \
             patch("cycle.run._prev_best", return_value=0.5), \
             patch("cycle.run._recent_failure_summary", return_value=""), \
             patch("cycle.run._build_retrieval_query", return_value="query"), \
             patch("memory.retriever.search", return_value=[]), \
             patch("memory.retriever.search_failure_lessons", return_value=[]), \
             patch("cycle.action_optimizer.assign_super_cycle_actions", return_value={}), \
             patch("importlib.import_module", side_effect=fake_import):
            import bin.run_retrieve_task as rrt
            importlib.reload(rrt)
            try:
                rrt.main()
            except SystemExit:
                pass
    finally:
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))
    return conn


def test_ttl_sweep_runs_after_retrieve() -> None:
    """BON-242: retrieve 실행마다 오래된 super_cycle_context row TTL sweep이 나가야 한다."""
    conn = _run_retrieve_with_mocks()
    sweep_calls = [
        s for s in conn.executed
        if "DELETE FROM raw.super_cycle_context" in s and "created_at <" in s
    ]
    assert len(sweep_calls) == 1


def test_ttl_sweep_failure_does_not_abort_retrieve() -> None:
    """sweep DELETE가 예외를 던져도 retrieve 자체는 끝까지 실행돼야 한다."""

    class _FlakyConn(_Conn):
        def execute(self, sql, params=None):
            if "DELETE FROM raw.super_cycle_context" in " ".join(sql.split()):
                raise RuntimeError("db hiccup")
            return super().execute(sql, params)

    def fake_import(name, *a, **k):
        if name.startswith("config.competitions."):
            return _fake_comp_module()
        return importlib.import_module(name, *a, **k)

    conn = _FlakyConn()
    argv = ["run_retrieve_task", "--competition", _SLUG, "--stage", "reflexion",
            "--queue-id", "qid", "--run-id", "rid"]
    try:
        with patch.object(sys, "argv", argv), \
             patch("store.db.connect", return_value=conn), \
             patch("store.db.ensure_competition"), \
             patch("cycle.run._prev_best", return_value=0.5), \
             patch("cycle.run._recent_failure_summary", return_value=""), \
             patch("cycle.run._build_retrieval_query", return_value="query"), \
             patch("memory.retriever.search", return_value=[]), \
             patch("memory.retriever.search_failure_lessons", return_value=[]), \
             patch("cycle.action_optimizer.assign_super_cycle_actions", return_value={}), \
             patch("importlib.import_module", side_effect=fake_import):
            import bin.run_retrieve_task as rrt
            importlib.reload(rrt)
            # SystemExit이면 안 됨 — sweep 실패가 retrieve를 죽이면 실패
            rrt.main()
    finally:
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))
