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
- Fit transformations on train only (no leakage)"""


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _client() -> Client:
    kwargs: dict = {"host": settings.OLLAMA_BASE_URL}
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
