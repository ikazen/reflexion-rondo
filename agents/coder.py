from __future__ import annotations

import logging
import re

from ollama import Client

from config import settings

_LOG = logging.getLogger(__name__)

_REFLEXION_CONTRACT = """\
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
  ensemble             -> any hooks needed (combine models: build_model + postprocess_predictions typical)

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
  def param_candidates(self, ctx) -> list[dict]  # MUST return 3–12 dicts — Cartesian grid forbidden
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
- param_candidates MUST return between 3 and 12 dicts — never build a Cartesian product grid

## Polars rules (do NOT use pandas-style API)
- String columns have dtype pl.String (NOT pl.Categorical)
- Correct ordinal encoding for pl.String columns:
    mapping = {v: i for i, v in enumerate(sorted(train[col].unique().to_list()))}
    train = train.with_columns(pl.col(col).replace_strict(mapping).cast(pl.Int32))
    valid = valid.with_columns(pl.col(col).replace_strict(mapping).cast(pl.Int32))
- pl.concat requires identical schemas
- No inplace mutations
- clip() takes positional args: expr.clip(lower_bound, upper_bound)"""

_BOOTSTRAP_CONTRACT = """\
Generate a complete, self-contained Python pipeline class named Patch.
This is the very first pipeline for this competition — build it from scratch so it runs without errors.
Implement ALL hooks that are needed for the dataset to work correctly end-to-end.

## Hook execution order (each CV fold)
  preprocess(self, train, valid, target, ctx)        -> (train, valid)
  feature_transform(self, train, valid, target, ctx) -> (Xtr, Xva)   [drop target here]
  param_candidates(self, ctx)                        -> list[dict]
  build_model(self, params, ctx)                     -> sklearn estimator
  postprocess_predictions(self, preds, ctx)          -> preds

## Required structure
```python
import polars as pl

class Patch:
    action_type = "bootstrap"
    changed_stages = ["preprocess", "feature_transform", "build_model"]
    rationale = "<one line describing the baseline approach>"

    def preprocess(self, train, valid, target, ctx):
        # MUST convert every pl.String column to numeric
        ...

    def feature_transform(self, train, valid, target, ctx):
        # MUST drop the target column before returning
        ...

    def build_model(self, params, ctx):
        # Return an appropriate sklearn estimator
        ...
```

## Hook signatures
  def preprocess(self, train: pl.DataFrame, valid: pl.DataFrame, target: str, ctx) -> tuple[pl.DataFrame, pl.DataFrame]
  def feature_transform(self, train: pl.DataFrame, valid: pl.DataFrame, target: str, ctx) -> tuple[pl.DataFrame, pl.DataFrame]
  def param_candidates(self, ctx) -> list[dict]  # MUST return 3–12 dicts — Cartesian grid forbidden
  def build_model(self, params: dict, ctx) -> sklearn_estimator
  def postprocess_predictions(self, preds, ctx) -> preds

## ctx attributes
  ctx.target_col: str
  ctx.metric: str
  ctx.seed: int
  ctx.is_classification: bool

## Rules
- Patch.action_type MUST be exactly "bootstrap"
- preprocess MUST encode every pl.String column to a numeric type
- feature_transform MUST drop the target column before returning
- build_model MUST return a classifier when ctx.is_classification else a regressor
- Fit all transformations on train only, apply identically to valid (no leakage)
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
    if not blocks:
        return text.strip()
    for b in blocks:
        if "class Patch" in b:
            return b.strip()
    return blocks[0].strip()


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
    known_errors: list[str] | None = None,
) -> str:
    is_bootstrap = action_type == "bootstrap"
    contract = _BOOTSTRAP_CONTRACT if is_bootstrap else _REFLEXION_CONTRACT

    user_parts = [
        f"## EDA Card\n{eda_card}",
        f"## Hypothesis\nAction type: {action_type}\n{hypothesis}",
    ]
    if known_errors:
        pitfall_block = "## Known failure modes (past errors on this task + action — avoid these)\n"
        pitfall_block += "\n".join(f"- {e}" for e in known_errors)
        user_parts.insert(0, pitfall_block)
    if prev_code and not is_bootstrap:
        user_parts.append(
            "## Current Best Pipeline\n"
            "This is the accumulated best pipeline. Understand what hooks are already implemented "
            "so your Patch complements rather than duplicates them. "
            f"You must only implement the hook(s) allowed for action_type={action_type!r}.\n"
            f"```python\n{prev_code}\n```"
        )
    if error_feedback:
        user_parts.append(f"## Validation Errors (fix these)\n{error_feedback}")

    user_parts.append("Write the Patch class now.")

    import time

    retry_tag = " [retry/feedback]" if error_feedback else ""
    _LOG.info("model=%s action=%s prev_code=%s%s",
              settings.MODEL_CODER, action_type,
              "yes" if prev_code else "no", retry_tag)
    _t0 = time.monotonic()
    resp = _client().chat(
        model=settings.MODEL_CODER,
        messages=[
            {"role": "system", "content": contract},
            {"role": "user",   "content": "\n\n".join(user_parts)},
        ],
    )
    source = _extract_code(resp.message.content)
    _LOG.info("done in %.1fs  code_chars=%d", time.monotonic() - _t0, len(source))
    return source
