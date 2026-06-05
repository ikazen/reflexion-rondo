from __future__ import annotations

import re

from ollama import Client

from config import settings

_CONTRACT = """\
You must produce exactly two Python functions. No other top-level code.

```python
import polars as pl

def feature_fn(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    target: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Fit any statistics on train only, apply to both.
    # Drop the target column before returning.
    ...

def model_fn(params: dict) -> object:
    # Return a sklearn-compatible estimator (fit / predict[_proba]).
    ...
```

Rules:
- feature_fn must accept exactly 3 args and return a 2-tuple of polars DataFrames
- model_fn must accept exactly 1 arg and return an object with fit() and predict()
- No file I/O, no network calls, no eval/exec/open
- Fit transformations on train only (no leakage)

Polars rules (do NOT use pandas-style API):
- String columns have dtype pl.String (NOT pl.Categorical) — detect with dtype == pl.String
- series.cat.codes does not exist → use series.to_physical() (only for pl.Categorical)
- Correct ordinal encoding for pl.String columns:
    mapping = {v: i for i, v in enumerate(sorted(train[col].unique().to_list()))}
    train = train.with_columns(pl.col(col).replace_strict(mapping).cast(pl.Int32))
    valid = valid.with_columns(pl.col(col).replace_strict(mapping).cast(pl.Int32))
- pl.concat requires identical schemas → align columns before concat
- No inplace mutations (no fillna(inplace=True) or similar)
- clip() takes positional args: expr.clip(lower_bound, upper_bound) — NOT clip(lower=..., upper=...)"""


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
            "## Previous Best Pipeline\n"
            "Start from this exact code. Apply the single change the hypothesis "
            f"(action_type: {action_type}) requires and nothing else — keep every "
            "other line identical so the CV delta is attributable to one change.\n"
            f"```python\n{prev_code}\n```"
        )
    if error_feedback:
        parts.append(f"## Validation Errors (fix these)\n{error_feedback}")

    parts.append("Write the two functions now.")

    resp = _client().chat(
        model=settings.MODEL_CODER,
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
    )
    return _extract_code(resp.message.content)
