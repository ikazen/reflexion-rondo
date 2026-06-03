from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

import duckdb
from ollama import Client

from config import settings
from memory.retriever import insert_reflection

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


@dataclass(frozen=True, slots=True)
class ReflectionOutput:
    reflection_id: str
    embedded_text: str
    full_lesson: str
    generality: str
    reflector_label: str


def _client() -> Client:
    kwargs: dict = {"host": settings.OLLAMA_CLOUD_BASE_URL}
    if settings.OLLAMA_API_KEY:
        kwargs["headers"] = {"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"}
    return Client(**kwargs)


def _format_context(ctx: AttemptContext) -> str:
    fi_text = json.dumps(ctx.feature_importance, indent=2) if ctx.feature_importance else "N/A"
    gain_text = f"{ctx.gain_vs_best:+.5f}" if ctx.gain_vs_best is not None else "N/A (first attempt)"
    error_text = ctx.error_trace or "none"

    return f"""## Attempt Summary
- Hypothesis: {ctx.hypothesis}
- Action type: {ctx.action_type}
- Evaluator label: {ctx.label}
- CV score: {ctx.cv_score:.5f}
- Gain vs best: {gain_text}
- Fold variance: {ctx.cv_fold_var:.6f}

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
    conn: duckdb.DuckDBPyConnection,
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
{{"embedded_text": "one-paragraph summary for search", "full_lesson": "detailed lesson", "generality": "<L1_local|L2_class|L3_general>", "reflector_label": "<jump|neutral|regression>"}}"""

    resp = _client().chat(
        model=settings.MODEL_REFLECTOR,
        messages=[{"role": "user", "content": user_prompt}],
        format="json",
    )

    content = resp.message.content.strip()
    if not content:
        raise ValueError("Reflector returned empty response")
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if m:
        content = m.group(1).strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Reflector JSON parse failed: {e}\nraw: {content[:300]}") from e

    if data.get("generality") not in GENERALITY_VALUES:
        data["generality"] = "L3_general"
    if data.get("reflector_label") not in LABEL_VALUES:
        data["reflector_label"] = "neutral"

    reflection_id = str(uuid.uuid4())

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
    )

    return ReflectionOutput(
        reflection_id=reflection_id,
        embedded_text=data["embedded_text"],
        full_lesson=data["full_lesson"],
        generality=data["generality"],
        reflector_label=data["reflector_label"],
    )
