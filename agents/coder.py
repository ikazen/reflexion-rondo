from __future__ import annotations

import re

from ollama import Client

from config import settings

_CONTRACT = """\
Generate a Python class named Patch that implements exactly the hook(s) required by the given action_type.

## System-owned pipeline
The system runs the following hooks in order on each CV fold:
  preprocess(self, train, valid, target, ctx)       -> (train, valid)
  feature_transform(self, train, valid, target, ctx) -> (Xtr, Xva)   [drop target here]
  param_candidates(self, ctx)                        -> list[dict]
  build_model(self, params, ctx)                     -> sklearn estimator
  postprocess_predictions(self, preds, ctx)          -> preds

Your Patch overrides only the hook(s) assigned. All other hooks fall back to the current best pipeline.

## Allowed hooks per action_type
  feature_engineering  -> feature_transform only
  model_swap           -> build_model only
  preprocessing        -> preprocess only
  hyperparam_search    -> param_candidates only
  compound             -> at most 2 hooks from the list above

## Required Patch structure
```python
import polars as pl

class Patch:
    action_type = "<assigned action_type>"
    changed_stages = ["<stage>"]
    rationale = "<one line: what this patch changes and why>"

    def <hook>(self, ...):
        ...
```

## Hook signatures
  def preprocess(self, train: pl.DataFrame, valid: pl.DataFrame, target: str, ctx) -> tuple[pl.DataFrame, pl.DataFrame]
  def feature_transform(self, train: pl.DataFrame, valid: pl.DataFrame, target: str, ctx) -> tuple[pl.DataFrame, pl.DataFrame]
  def param_candidates(self, ctx) -> list[dict]
  def build_model(self, params: dict, ctx) -> sklearn_estimator
  def postprocess_predictions(self, preds, ctx) -> preds

## ctx attributes
  ctx.target_col: str
  ctx.metric: str
  ctx.seed: int
  ctx.is_classification: bool

## Rules
- Only implement the hook(s) allowed for your action_type
- Patch.action_type MUST exactly match the assigned action_type
- feature_transform must drop the target column before returning
- Fit all transformations on train only, apply to valid (no leakage)
- No file I/O, no network calls, no eval/exec/open

## Polars rules (do NOT use pandas-style API)
- String columns have dtype pl.String (NOT pl.Categorical)
- Correct ordinal encoding for pl.String columns:
    mapping = {v: i for i, v in enumerate(sorted(train[col].unique().to_list()))}
    train = train.with_columns(pl.col(col).replace_strict(mapping).cast(pl.Int32))
    valid = valid.with_columns(pl.col(col).replace_strict(mapping).cast(pl.Int32))
- pl.concat requires identical schemas
- No inplace mutations
- clip() takes positional args: expr.clip(lower_bound, upper_bound)"""


def _extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks)
    return text.strip()


def _client() -> Client:
    kwargs: dict = {"host": settings.OLLAMA_CLOUD_BASE_URL}
    if settings.OLLAMA_API_KEY:
        kwargs["headers"] = {"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"}
    return Client(**kwargs)


def generate_code(
    hypothesis: str,
    action_type: str,
    eda_card: str,
    prev_code: str | None = None,
    error_feedback: str | None = None,
) -> str:
    parts = [
        f"## Contract\n{_CONTRACT}",
        f"## EDA Card\n{eda_card}",
        f"## Hypothesis\nAction type: {action_type}\n{hypothesis}",
    ]
    if prev_code:
        parts.append(
            "## Current Best Pipeline\n"
            "This is the accumulated best pipeline. Understand what hooks are already implemented "
            "so your Patch complements rather than duplicates them. "
            f"You must only implement the hook(s) allowed for action_type={action_type!r}.\n"
            f"```python\n{prev_code}\n```"
        )
    if error_feedback:
        parts.append(f"## Validation Errors (fix these)\n{error_feedback}")

    parts.append("Write the Patch class now.")

    resp = _client().chat(
        model=settings.MODEL_CODER,
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
    )
    return _extract_code(resp.message.content)
