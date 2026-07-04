from __future__ import annotations

from agents.reflector import AttemptContext, GENERALITY_VALUES, LABEL_VALUES, _format_context


def _apply_fallbacks(data: dict) -> dict:
    """agents/reflector.py의 sanitize 블록과 동일 로직 — fallback 동작 검증용."""
    if not data.get("embedded_text"):
        data["embedded_text"] = (data.get("full_lesson") or "")[:500]
    if not data.get("full_lesson"):
        data["full_lesson"] = data["embedded_text"]
    if data.get("generality") not in GENERALITY_VALUES:
        data["generality"] = "L1_local"
    if data.get("reflector_label") not in LABEL_VALUES:
        data["reflector_label"] = "neutral"
    return data


def test_invalid_generality_defaults_to_l1_local():
    data = _apply_fallbacks({"generality": "bad_value", "reflector_label": "jump",
                              "embedded_text": "x", "full_lesson": "x"})
    assert data["generality"] == "L1_local"


def test_missing_generality_defaults_to_l1_local():
    data = _apply_fallbacks({"reflector_label": "jump",
                              "embedded_text": "x", "full_lesson": "x"})
    assert data["generality"] == "L1_local"


def test_valid_generality_values_preserved():
    for val in GENERALITY_VALUES:
        data = _apply_fallbacks({"generality": val, "reflector_label": "jump",
                                  "embedded_text": "x", "full_lesson": "x"})
        assert data["generality"] == val, f"{val} was overwritten"


def test_l1_local_is_in_generality_values():
    assert "L1_local" in GENERALITY_VALUES


def test_invalid_label_defaults_to_neutral():
    data = _apply_fallbacks({"generality": "L1_local", "reflector_label": "bad",
                              "embedded_text": "x", "full_lesson": "x"})
    assert data["reflector_label"] == "neutral"


def test_missing_full_lesson_filled_from_embedded_text():
    data = _apply_fallbacks({"generality": "L1_local", "reflector_label": "jump",
                              "embedded_text": "the lesson"})
    assert data["full_lesson"] == "the lesson"


def _ctx(**overrides) -> AttemptContext:
    base = dict(
        hypothesis="h", action_type="hyperparam_search", code="class Patch: ...",
        cv_score=0.9, cv_fold_var=0.0, gain_vs_best=0.0, label="neutral",
    )
    base.update(overrides)
    return AttemptContext(**base)


def test_format_context_includes_noop_note_when_flagged():
    """BON-239: is_noop_tie=True면 프롬프트에 명시적 설명 요청 노트가 포함된다."""
    text = _format_context(_ctx(is_noop_tie=True))
    assert "bit-for-bit identical to prev_best" in text


def test_format_context_omits_noop_note_by_default():
    text = _format_context(_ctx())
    assert "bit-for-bit identical" not in text
