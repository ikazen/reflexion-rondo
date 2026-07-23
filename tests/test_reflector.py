from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agents.reflector import AttemptContext, GENERALITY_VALUES, LABEL_VALUES, _format_context, reflect


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
    """is_noop_tie=True면 프롬프트에 명시적 설명 요청 노트가 포함된다."""
    text = _format_context(_ctx(is_noop_tie=True))
    assert "bit-for-bit identical to prev_best" in text


def test_format_context_omits_noop_note_by_default():
    text = _format_context(_ctx())
    assert "bit-for-bit identical" not in text


def _mock_chat_response() -> MagicMock:
    resp = MagicMock()
    resp.message.content = json.dumps({
        "embedded_text": "summary", "full_lesson": "lesson",
        "generality": "L1_local", "reflector_label": "neutral",
    })
    return resp


def test_reflect_passes_configured_think_setting_to_chat() -> None:
    """#51: kimi-k2.6은 thinking 모델이라 hidden thinking이 num_predict 예산을 잠식해
    JSON이 중간에 잘리는 문제가 있었다 — MODEL_REFLECTOR_THINK가 실제 chat() 호출에
    전달되는지 확인."""
    with patch("agents.reflector._client") as mock_client, \
         patch("agents.reflector.insert_reflection") as mock_insert, \
         patch("agents.reflector.settings.MODEL_REFLECTOR_THINK", False):
        mock_client.return_value.chat.return_value = _mock_chat_response()
        reflect(MagicMock(), "attempt-1", "comp-1", _ctx())
        call_kwargs = mock_client.return_value.chat.call_args.kwargs
    assert call_kwargs["think"] is False
    mock_insert.assert_called_once()
