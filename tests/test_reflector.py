from __future__ import annotations

from agents.reflector import GENERALITY_VALUES, LABEL_VALUES


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
