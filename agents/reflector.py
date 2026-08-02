from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field

from ollama import Client

from agents.llm_retry import chat_with_retry
from config import settings
from memory.retriever import insert_reflection
from store.db import PgConn

_LOG = logging.getLogger(__name__)
_REFLECT_RETRIES = 3

GENERALITY_VALUES = ["L1_local", "L2_class", "L3_general"]
LABEL_VALUES = ["jump", "neutral", "regression"]

_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "embedded_text": {
            "type": "string",
            "description": "One-paragraph summary used as the search key for future retrieval.",
        },
        "full_lesson": {
            "type": "string",
            "description": "Detailed lesson explaining what worked or failed and why.",
        },
        "generality": {
            "type": "string",
            "enum": GENERALITY_VALUES,
            "description": "L1_local: specific to this competition. L2_class: applies to similar fingerprint class. L3_general: universal tabular principle.",
        },
        "reflector_label": {
            "type": "string",
            "enum": LABEL_VALUES,
            "description": "Qualitative assessment (reference only — not the ground truth label).",
        },
    },
    "required": ["embedded_text", "full_lesson", "generality", "reflector_label"],
}


@dataclass(frozen=True, slots=True)
class AttemptContext:
    hypothesis: str
    action_type: str
    code: str
    cv_score: float
    cv_fold_var: float
    gain_vs_best: float | None
    label: str
    retrieved_ids: list[str] = field(default_factory=list)
    feature_importance: dict | None = None
    error_trace: str | None = None
    is_noop_tie: bool = False


@dataclass(frozen=True, slots=True)
class ReflectionOutput:
    reflection_id: str
    embedded_text: str
    full_lesson: str
    generality: str
    reflector_label: str
    lesson_type: str


def _derive_lesson_type(label: str, error_trace: str | None) -> str:
    if error_trace:
        return "failure"
    if label == "jump":
        return "recommend"
    if label == "regression":
        return "avoid"
    return "no_op"


def _client() -> Client:
    kwargs: dict = {"host": settings.OLLAMA_CLOUD_BASE_URL}
    if settings.OLLAMA_API_KEY:
        kwargs["headers"] = {"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"}
    return Client(**kwargs)


def _tail_error(trace: str, max_chars: int = 1000) -> str:
    if len(trace) <= max_chars:
        return trace
    lines = trace.splitlines()
    kept: list[str] = []
    budget = max_chars
    for line in reversed(lines):
        cost = len(line) + 1
        if cost > budget:
            break
        kept.append(line)
        budget -= cost
    return "[...]\n" + "\n".join(reversed(kept))


def _format_context(ctx: AttemptContext) -> str:
    fi_text = json.dumps(ctx.feature_importance, indent=2) if ctx.feature_importance else "N/A"
    gain_text = f"{ctx.gain_vs_best:+.5f}" if ctx.gain_vs_best is not None else "N/A (first attempt)"
    error_text = _tail_error(ctx.error_trace) if ctx.error_trace else "none"
    noop_note = (
        "\n- NOTE: cv_score is bit-for-bit identical to prev_best. The patch made no "
        "effective change to the evaluated pipeline (e.g. a hook fell back to the base "
        "pipeline, or reimplemented logic already present in it). Explain WHY this "
        "action_type/hypothesis likely had no effect here, so the strategist avoids "
        "repeating it blindly."
        if ctx.is_noop_tie else ""
    )

    return f"""## Attempt Summary
- Hypothesis: {ctx.hypothesis}
- Action type: {ctx.action_type}
- Evaluator label: {ctx.label}
- CV score: {ctx.cv_score:.5f}
- Gain vs best: {gain_text}
- Fold variance: {ctx.cv_fold_var:.6f}{noop_note}

## Code
```python
{ctx.code}
```

## Feature Importance (top features)
{fi_text}

## Error Trace
{error_text}

## Retrieved Lesson IDs Used
{ctx.retrieved_ids if ctx.retrieved_ids else 'none'}"""


def reflect(
    conn: PgConn,
    attempt_id: str,
    competition_id: str,
    context: AttemptContext,
) -> ReflectionOutput:
    user_prompt = f"""{_format_context(context)}

## Task
Write a lesson that captures what this attempt taught us.

For generality:
- L1_local if the lesson depends on column names or statistics specific to this competition
- L2_class if it likely applies to competitions with a similar fingerprint (task/metric/size class)
- L3_general if it is a universal principle for tabular ML

Respond with ONLY a JSON object using exactly these keys:
{{"embedded_text": "one-paragraph summary for search (2-3 sentences max)", "full_lesson": "detailed lesson (4-5 sentences max)", "generality": "<L1_local|L2_class|L3_general>", "reflector_label": "<jump|neutral|regression>"}}"""

    import time as _time

    gain_str = f"{context.gain_vs_best:+.5f}" if context.gain_vs_best is not None else "N/A"
    print(f"[reflector] model={settings.MODEL_REFLECTOR} temp={settings.LLM_TEMPERATURE}"
          f" label={context.label} cv={context.cv_score:.5f} gain={gain_str}")
    _t0 = _time.monotonic()
    last_err: Exception | None = None
    for attempt in range(_REFLECT_RETRIES):
        resp = chat_with_retry(
            _client,
            model=settings.MODEL_REFLECTOR,
            messages=[{"role": "user", "content": user_prompt}],
            think=settings.MODEL_REFLECTOR_THINK,
            format=_OUTPUT_SCHEMA,
            options=settings.llm_options(num_predict=4096),
        )
        content = resp.message.content.strip()
        if not content:
            last_err = ValueError("Reflector returned empty response")
            _LOG.warning("reflect attempt %d/%d: empty response", attempt + 1, _REFLECT_RETRIES)
            _time.sleep(2)
            continue
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if m:
            content = m.group(1).strip()
        try:
            data = json.loads(content)
            break
        except json.JSONDecodeError as e:
            last_err = ValueError(f"Reflector JSON parse failed: {e}\nraw: {content[:300]}")
            _LOG.warning("reflect attempt %d/%d: %s", attempt + 1, _REFLECT_RETRIES, last_err)
    else:
        raise last_err  # type: ignore[misc]

    if not data.get("embedded_text"):
        data["embedded_text"] = (data.get("full_lesson") or "")[:500]
    if not data.get("full_lesson"):
        data["full_lesson"] = data["embedded_text"]
    if data.get("generality") not in GENERALITY_VALUES:
        data["generality"] = "L1_local"
    if data.get("reflector_label") not in LABEL_VALUES:
        data["reflector_label"] = "neutral"

    print(f"[reflector] done in {_time.monotonic() - _t0:.1f}s"
          f" generality={data['generality']} reflector_label={data['reflector_label']}"
          f" lesson_type={_derive_lesson_type(context.label, context.error_trace)}")

    reflection_id = str(uuid.uuid4())
    lesson_type = _derive_lesson_type(context.label, context.error_trace)

    insert_reflection(
        conn=conn,
        reflection_id=reflection_id,
        attempt_id=attempt_id,
        competition_id=competition_id,
        embedded_text=data["embedded_text"],
        full_lesson=data["full_lesson"],
        generality=data["generality"],
        label=context.label,
        gain_vs_best=context.gain_vs_best if context.gain_vs_best is not None else 0.0,
        reflector_label=data["reflector_label"],
        lesson_type=lesson_type,
    )

    return ReflectionOutput(
        reflection_id=reflection_id,
        embedded_text=data["embedded_text"],
        full_lesson=data["full_lesson"],
        generality=data["generality"],
        reflector_label=data["reflector_label"],
        lesson_type=lesson_type,
    )
