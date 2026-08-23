"""LLM 코드 생성 — Patch 클래스 소스를 만든다(generate_code).

action_type별 허용 hook을 시스템 프롬프트에 강제하고, 응답에서 코드 블록만 추출한다.
"""
from __future__ import annotations

import logging
import re

from ollama import Client

from agents.llm_retry import chat_with_retry
from config import settings
from evaluator.contract import _ALLOWED_HOOKS, _ALL_HOOKS

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
  model_swap           -> model_spec (STRONGLY PREFERRED, see below) or build_model
  preprocessing        -> preprocess only
  hyperparam_search    -> param_candidates only
  ensemble             -> ensemble_spec (STRONGLY PREFERRED, see below) or any hooks needed for a
                          hand-written combination (build_model + postprocess_predictions typical)

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
  def ensemble_spec(self, ctx) -> dict   # ensemble action_type only, see below
  def model_spec(self, ctx) -> dict      # model_swap action_type only, see below

## ctx attributes
  ctx.target_col: str
  ctx.metric: str
  ctx.seed: int
  ctx.is_classification: bool

## Available libraries
- Available: scikit-learn, lightgbm, xgboost, catboost, imbalanced-learn (imblearn), optuna, polars
- NOT available: tabpfn, pandas — importing these fails

## ensemble action_type — use ensemble_spec, not a hand-written wrapper class
For ensemble, implement `ensemble_spec(self, ctx) -> dict` instead of writing a custom
estimator/wrapper class by hand. The harness builds, fits (with early stopping where
supported), and combines the member models itself:

```python
class Patch:
    action_type = "ensemble"
    changed_stages = ["ensemble_spec"]
    rationale = "blend lgbm and xgboost"

    def ensemble_spec(self, ctx) -> dict:
        return {
            "members": [
                {"model": "lgbm", "params": {"n_estimators": 300, "learning_rate": 0.05}},
                {"model": "xgboost", "params": {"max_depth": 6}},
            ],
            "method": "weighted_average",   # or "majority_vote" for discrete-label metrics
            "weights": [0.6, 0.4],          # optional — omit for equal weights
        }
```
- Allowed model names: lgbm, xgboost, catboost, hgb, random_forest, extra_trees, ridge,
  elastic_net — the harness picks the classifier or regressor variant automatically from
  ctx.is_classification. Do NOT write "LGBMClassifier" etc, just the short name.
- params are passed straight to that model's constructor (same values you'd put in build_model).
- method: "weighted_average" for regression or probability-based metrics (auc/logloss), or
  "majority_vote" for discrete-label metrics (accuracy/f1/qwk/balanced_accuracy). If omitted,
  the harness picks a sensible default from the metric.
- If you implement ensemble_spec, do NOT also implement build_model — ensemble_spec replaces it.
  You may still implement preprocess/feature_transform/postprocess_predictions alongside it.
- This is strongly preferred over a hand-written ensemble wrapper class. Wrapper classes have
  repeatedly failed in production from bugs the harness cannot see inside exec'd code: incorrect
  `super()` usage, stale/removed constructor kwargs hardcoded from memory, and member models
  reconstructed incorrectly inside the wrapper's own fit(). ensemble_spec avoids all of these
  because the harness — not generated code — constructs and fits every member.
- Only fall back to a hand-written build_model/wrapper class if ensemble_spec genuinely cannot
  express what you need (e.g. a stacking meta-model on out-of-fold predictions) — expect that
  path to be less reliable.

## model_swap action_type — use model_spec, not build_model
For model_swap, implement `model_spec(self, ctx) -> dict` instead of writing build_model by hand.
The harness constructs the model itself from a registry, so a swapped-in model can never suffer
from stale/removed constructor kwargs or a misremembered class name:

```python
class Patch:
    action_type = "model_swap"
    changed_stages = ["model_spec"]
    rationale = "switch to catboost, handles categoricals natively"

    def model_spec(self, ctx) -> dict:
        return {
            "model": "catboost",
            "params": {"iterations": 500, "learning_rate": 0.05, "depth": 6},
        }
```
- Allowed model names: lgbm, xgboost, catboost, hgb, random_forest, extra_trees, ridge,
  elastic_net — same registry as ensemble_spec, same short-name rule (no "CatBoostClassifier" etc).
- params are passed straight to that model's constructor (same values you'd put in build_model).
- If you implement model_spec, do NOT also implement build_model — model_spec replaces it.
- Only fall back to build_model if the model you need is not in the registry (e.g. a library
  outside the registry, or constructor logic ensemble_spec/model_spec cannot express).

## Rules
- Only implement the hook(s) allowed for your action_type
- Patch.action_type MUST exactly match the assigned action_type
- feature_transform must drop the target column before returning
- Fit all transformations on train only, apply to valid (no leakage)
- No file I/O, no network calls, no eval/exec/open
- param_candidates MUST return between 3 and 12 dicts — never build a Cartesian product grid
- Multiclass targets are often string labels (e.g. "Low"/"Medium"/"High"), not integers. The
  harness scores postprocess_predictions() output against the ORIGINAL untouched string target —
  if you encode the target to integers anywhere (preprocess/build_model), you MUST decode
  predictions back to the original string labels in postprocess_predictions before returning.
  Returning encoded integers crashes scoring (`ValueError: Mix of label input types`).
  LightGBM/CatBoost/sklearn classifiers accept string class labels directly and predict() returns
  them unchanged — simplest and safest for those. XGBoost's sklearn wrapper is the exception: it
  REQUIRES integer labels 0..n-1 and raises `ValueError: Invalid classes inferred` on strings —
  if you use XGBoost for a multiclass target, encode y to 0..n-1 (fit the mapping on train only)
  and decode predictions back to the original strings in postprocess_predictions.
- The harness always calls build_model()'s fit()/predict()/predict_proba() with numpy arrays
  (never a polars DataFrame), on every CV fold. If your Patch defines a custom estimator/wrapper
  class (e.g. for ensemble), its fit(self, X, y) and predict(self, X) MUST accept numpy arrays
  directly — do NOT call X.to_numpy() or any DataFrame-only method on them, that raises
  `AttributeError: 'numpy.ndarray' object has no attribute 'to_numpy'`.
- If your Patch defines its own class (e.g. an ensemble wrapper), never call the explicit two-arg
  `super(ClassName, self)` — use zero-arg `super()` only. The harness loads Patch dynamically and
  explicit super() calls can raise `TypeError: super(type, obj): obj must be an instance or
  subtype of type` when the class identity doesn't match at runtime.
- Do not use removed/deprecated sklearn constructor kwargs from memory (e.g. LogisticRegression's
  old `multi_class` argument no longer exists) — if unsure whether a kwarg is still valid, omit it
  and rely on the estimator's default.

## Polars rules (do NOT use pandas-style API)
- String columns have dtype pl.String (NOT pl.Categorical)
- Correct ordinal encoding for pl.String columns — drop_nulls() before unique()/sorted() (a raw
  train[col].unique().to_list() can contain None, and sorted() then raises `TypeError: '<' not
  supported between instances of 'NoneType' and 'str'`), and always pass default= to
  replace_strict (unmapped values, e.g. an unseen category or a null, otherwise crash with
  `InvalidOperationError: incomplete mapping specified for replace_strict`):
    mapping = {v: i for i, v in enumerate(sorted(train[col].drop_nulls().unique().to_list()))}
    train = train.with_columns(pl.col(col).replace_strict(mapping, default=-1).cast(pl.Int32))
    valid = valid.with_columns(pl.col(col).replace_strict(mapping, default=-1).cast(pl.Int32))
- pl.concat requires identical schemas
- No inplace mutations
- clip() takes positional args: expr.clip(lower_bound, upper_bound)
- FORBIDDEN (pandas-only, do not exist on polars — statically rejected):
    .groupby()   -> use .group_by()
    .map_dict()  -> use .replace_strict()
    .take()      -> use .gather()
    .apply()     -> use .map_elements()
    .iterrows(), .applymap(), .get_dummies() -> no polars equivalent needed, avoid entirely"""

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

## Available libraries
- Available: scikit-learn, lightgbm, xgboost, catboost, imbalanced-learn (imblearn), optuna, polars
- NOT available: tabpfn, pandas — importing these fails

## Rules
- Patch.action_type MUST be exactly "bootstrap"
- preprocess MUST encode every pl.String FEATURE column to a numeric type (do not touch the
  target column here — leave it as-is)
- feature_transform MUST drop the target column before returning
- build_model MUST return a classifier when ctx.is_classification else a regressor
- Fit all transformations on train only, apply identically to valid (no leakage)
- No file I/O, no network calls, no eval/exec/open
- Multiclass targets are often string labels (e.g. "Low"/"Medium"/"High"), not integers. The
  harness scores postprocess_predictions() output against the ORIGINAL untouched string target —
  if you encode the target to integers anywhere, you MUST decode predictions back to the
  original string labels in postprocess_predictions before returning. Returning encoded integers
  crashes scoring (`ValueError: Mix of label input types`).
  LightGBM/CatBoost/sklearn classifiers accept string class labels directly and predict() returns
  them unchanged — simplest and safest for those. XGBoost's sklearn wrapper is the exception: it
  REQUIRES integer labels 0..n-1 and raises `ValueError: Invalid classes inferred` on strings —
  if you use XGBoost for a multiclass target, encode y to 0..n-1 (fit the mapping on train only)
  and decode predictions back to the original strings in postprocess_predictions.
- The harness always calls build_model()'s fit()/predict()/predict_proba() with numpy arrays
  (never a polars DataFrame), on every CV fold. If your Patch defines a custom estimator/wrapper
  class, its fit(self, X, y) and predict(self, X) MUST accept numpy arrays directly — do NOT call
  X.to_numpy() or any DataFrame-only method on them, that raises `AttributeError: 'numpy.ndarray'
  object has no attribute 'to_numpy'`.
- If your Patch defines its own class, never call the explicit two-arg `super(ClassName, self)` —
  use zero-arg `super()` only. The harness loads Patch dynamically and explicit super() calls can
  raise `TypeError: super(type, obj): obj must be an instance or subtype of type` when the class
  identity doesn't match at runtime.
- Do not use removed/deprecated sklearn constructor kwargs from memory (e.g. LogisticRegression's
  old `multi_class` argument no longer exists) — if unsure whether a kwarg is still valid, omit it
  and rely on the estimator's default.

## Polars rules (do NOT use pandas-style API)
- String columns have dtype pl.String (NOT pl.Categorical)
- Correct ordinal encoding for pl.String columns — drop_nulls() before unique()/sorted() (a raw
  train[col].unique().to_list() can contain None, and sorted() then raises `TypeError: '<' not
  supported between instances of 'NoneType' and 'str'`), and always pass default= to
  replace_strict (unmapped values, e.g. an unseen category or a null, otherwise crash with
  `InvalidOperationError: incomplete mapping specified for replace_strict`):
    mapping = {v: i for i, v in enumerate(sorted(train[col].drop_nulls().unique().to_list()))}
    train = train.with_columns(pl.col(col).replace_strict(mapping, default=-1).cast(pl.Int32))
    valid = valid.with_columns(pl.col(col).replace_strict(mapping, default=-1).cast(pl.Int32))
- pl.concat requires identical schemas
- No inplace mutations
- clip() takes positional args: expr.clip(lower_bound, upper_bound)
- FORBIDDEN (pandas-only, do not exist on polars — statically rejected):
    .groupby()   -> use .group_by()
    .map_dict()  -> use .replace_strict()
    .take()      -> use .gather()
    .apply()     -> use .map_elements()
    .iterrows(), .applymap(), .get_dummies() -> no polars equivalent needed, avoid entirely"""


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

    # 정적 검증은 생성 이후에만 컨트랙트 위반을 잡아 반복된다 — 허용 hook을
    # 생성 이전 user 메시지에 action_type별로 명시.
    allowed_hooks = sorted(_ALLOWED_HOOKS.get(action_type, _ALL_HOOKS))
    hook_directive = (
        f"## Allowed hooks for THIS action_type={action_type!r} (STRICT)\n"
        f"You may implement ONLY: {allowed_hooks}. Any other hook will be rejected — "
        f"do not implement it even if it seems like it would help."
    )

    user_parts = [
        hook_directive,
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
    _LOG.info("model=%s action=%s prev_code=%s%s temp=%.2f",
              settings.MODEL_CODER, action_type,
              "yes" if prev_code else "no", retry_tag, settings.LLM_TEMPERATURE)
    _t0 = time.monotonic()
    resp = chat_with_retry(
        _client,
        model=settings.MODEL_CODER,
        messages=[
            {"role": "system", "content": contract},
            {"role": "user",   "content": "\n\n".join(user_parts)},
        ],
        options=settings.llm_options(reasoning_effort=settings.MODEL_CODER_REASONING_EFFORT),
    )
    source = _extract_code(resp.message.content)
    _LOG.info("done in %.1fs  code_chars=%d", time.monotonic() - _t0, len(source))
    return source
