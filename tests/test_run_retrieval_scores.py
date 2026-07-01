from __future__ import annotations

from cycle.run import _retrieval_scores


def test_all_lessons_have_score():
    lessons = [{"score": 0.9}, {"score": 0.5}]
    assert _retrieval_scores(lessons) == [0.9, 0.5]


def test_failure_lesson_without_score_key_does_not_raise():
    """BON-134 search_failure_lessons()는 score 키를 넣지 않는다 — KeyError 회귀(BON-228) 재발 방지."""
    lessons = [{"score": 0.9}, {"reflection_id": "r1", "lesson_type": "failure"}]
    assert _retrieval_scores(lessons) == [0.9, None]


def test_empty_lessons_returns_none():
    assert _retrieval_scores([]) is None
