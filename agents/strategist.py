from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ollama import Client

from config import settings

if TYPE_CHECKING:
    from cycle.stagnation import StagnationSignal

from config.settings import ACTION_TYPES

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
    kwargs: dict = {"host": settings.OLLAMA_CLOUD_BASE_URL}
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


def _format_stagnation(sig: StagnationSignal) -> str:
    lines = [
        f"- best CV stagnant for {sig.stagnant_for} attempts (jumps in last window: {sig.jumps_in_window})",
    ]
    if sig.underused_actions:
        lines.append(f"- underused action_types: {', '.join(sig.underused_actions)}")
        lines.append("- Prefer an underused action_type unless you have strong reason not to.")
    return "\n".join(lines)


def _format_action_prior(prior: dict[str, float]) -> str:
    lines = ["Estimated success probability per action_type (Thompson sample from this competition's history):"]
    for action, p in sorted(prior.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {action}: {p:.3f}")
    lines.append("Use this as a soft prior — you may override if you have strong reasoning.")
    return "\n".join(lines)


def strategize(
    eda_card: str,
    lessons: list[dict],
    stage: str,
    prev_best_cv: float | None = None,
    stagnation: StagnationSignal | None = None,
    action_prior: dict[str, float] | None = None,
    forced_action_type: str | None = None,
) -> StrategyDecision:
    lessons_text = _format_lessons(lessons)
    valid_ids = {l["reflection_id"] for l in lessons}
    prev_best_str = f"{prev_best_cv:.5f}" if prev_best_cv is not None else "none yet"
    action_types_str = ", ".join(ACTION_TYPES)

    exploration_section = ""
    if stagnation is not None and stagnation.is_stagnant:
        exploration_section = f"\n## Exploration Signal\n{_format_stagnation(stagnation)}\n"

    prior_section = ""
    if action_prior and not forced_action_type:
        prior_section = f"\n## Action Prior\n{_format_action_prior(action_prior)}\n"

    forced_section = ""
    if forced_action_type:
        forced_section = f"\n## Assigned Action\nYou MUST use action_type: {forced_action_type}. Do not choose any other action_type.\n"

    user_prompt = f"""## EDA Card
{eda_card}

## Retrieved Lessons
{lessons_text}

## Context
- Stage: {stage}
- Previous best CV: {prev_best_str}
{exploration_section}{prior_section}{forced_section}
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
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if m:
        content = m.group(1).strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Strategist JSON parse failed: {e}\nraw: {content[:300]}") from e

    if forced_action_type:
        data["action_type"] = forced_action_type
    elif data.get("action_type") not in ACTION_TYPES:
        data["action_type"] = "feature_engineering"

    adopted = [rid for rid in data.get("reflection_ids", []) if rid in valid_ids]

    return StrategyDecision(
        hypothesis=data["hypothesis"],
        action_type=data["action_type"],
        reflection_ids=adopted,
    )
