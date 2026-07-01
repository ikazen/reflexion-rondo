"""bin/run_retrieve_task.py — _merge_lessons 단위 테스트 (BON-134)."""
from __future__ import annotations

import sys
from pathlib import Path

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
