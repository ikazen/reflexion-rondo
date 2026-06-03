from __future__ import annotations

from dataclasses import dataclass, field

from ollama import Client

from config import settings

ACTION_TYPES: list[str] = [
    "feature_engineering",
    "model_swap",
    "hyperparam_search",
    "cv_strategy",
    "preprocessing",
    "ensemble",
]

_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "hypothesis": {
            "type": "string",
            "description": "One concrete hypothesis to test in this attempt.",
        },
        "action_type": {
            "type": "string",
            "enum": ACTION_TYPES,
        },
        "reflection_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "IDs of the retrieved lessons that were actually used to form this hypothesis. Empty list if none applied.",
        },
    },
    "required": ["hypothesis", "action_type", "reflection_ids"],
}


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    hypothesis: str
    action_type: str
    reflection_ids: list[str] = field(default_factory=list)


def _client() -> Client:
    kwargs: dict = {"host": settings.OLLAMA_BASE_URL}
    if settings.OLLAMA_API_KEY:
        kwargs["headers"] = {"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"}
    return Client(**kwargs)


def _format_lessons(lessons: list[dict]) -> str:
    if not lessons:
        return "(no relevant lessons retrieved)"
    parts = []
    for i, l in enumerate(lessons, 1):
        parts.append(
            f"{i}. [id={l['reflection_id']}] [{l['generality']}] {l['full_lesson']}"
        )
    return "\n".join(parts)


def strategize(
    eda_card: str,
    lessons: list[dict],
    stage: str,
    prev_best_cv: float | None = None,
) -> StrategyDecision:
    lessons_text = _format_lessons(lessons)
    valid_ids = {l["reflection_id"] for l in lessons}
    prev_best_str = f"{prev_best_cv:.5f}" if prev_best_cv is not None else "none yet"
    action_types_str = ", ".join(ACTION_TYPES)

    user_prompt = f"""## EDA Card
{eda_card}

## Retrieved Lessons
{lessons_text}

## Context
- Stage: {stage}
- Previous best CV: {prev_best_str}

## Task
Propose exactly one change to improve the CV score.
Select which retrieved lessons (if any) directly informed your hypothesis and list their IDs in reflection_ids.
Only include IDs from the list above — omit any that did not influence your reasoning.

Respond with ONLY a JSON object using exactly these keys:
{{"hypothesis": "...", "action_type": "<one of: {action_types_str}>", "reflection_ids": []}}"""

    import json
    import re

    resp = _client().chat(
        model=settings.MODEL_STRATEGIST,
        messages=[{"role": "user", "content": user_prompt}],
        format="json",
    )

    content = resp.message.content.strip()
    if not content:
        raise ValueError("Strategist returned empty response")
    # strip markdown code block if model ignores format param
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if m:
        content = m.group(1).strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Strategist JSON parse failed: {e}\nraw: {content[:300]}") from e

    if data.get("action_type") not in ACTION_TYPES:
        data["action_type"] = "feature_engineering"

    # keep only IDs that were actually provided (guard against hallucination)
    adopted = [rid for rid in data.get("reflection_ids", []) if rid in valid_ids]

    return StrategyDecision(
        hypothesis=data["hypothesis"],
        action_type=data["action_type"],
        reflection_ids=adopted,
    )
