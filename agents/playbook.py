"""Playground Series 대응 정적 playbook — Strategist/Coder 프롬프트에 큐레이션 지식으로 주입한다(#235, ADR-041).

측정 자산(reflections)이 아니라 "이 시스템에 이미 아는데 프롬프트에 안 들어간" baseline 지식.
"""
from __future__ import annotations

STRATEGIST_PLAYBOOK = """\
Curated Playground Series strategy (static baseline knowledge — not a measured lesson, weigh it against the EDA card):

- Validation first. Match the CV scheme to the data: StratifiedKFold for classification, GroupKFold when an
  id/entity repeats across rows, a time-aware split when there is an ordering column. A CV that does not track
  the leaderboard makes every downstream experiment noise.
- On tabular Playground data the largest score movers are usually feature engineering, not the estimator or its
  hyperparameters: out-of-fold target/count/frequency encoding of categoricals, pairwise ratios and products of
  the most predictive numerics, per-category aggregations (mean/std/min/max of key numerics grouped by each
  categorical), and date-part decomposition.
- Model diversity then blend. Three GBDTs trained separately (lgbm, xgboost, catboost) and averaged usually beats
  any one heavily tuned model; a stacked ridge meta-model on their out-of-fold predictions is the standard next step.
- Prefer a wide Optuna search over a hand-picked hyperparameter list. A 3-candidate param_candidates rarely finds
  anything — the useful move is widening ranges (num_leaves, learning_rate log-scale, subsample/colsample, L1/L2).
- Merging original / extra source data (already wired for some competitions) tends to help when the columns are
  compatible.
- If best CV has been flat for many attempts, change action_type away from the one tried repeatedly. A config that
  beats CV but not the audit holdout is fold overfitting, not progress — do not keep proposing it."""

CODER_PLAYBOOK = """\
## Playground Series implementation patterns (curated — apply when they fit the hypothesis)
Every frame below is a polars DataFrame — the "Polars rules" and the statically-rejected pandas-only list
above still apply to all of this code (use group_by / gather / replace_strict / map_elements, never the
pandas spellings).

- Group-aggregate features (per-category mean/std/count of key numerics), fit on train only:
    agg = train.group_by(cat).agg(
        pl.col(num).mean().alias(f"{cat}_{num}_mean"),
        pl.col(num).std().alias(f"{cat}_{num}_std"),
        pl.len().alias(f"{cat}_count"),
    )
    train = train.join(agg, on=cat, how="left")
    valid = valid.join(agg, on=cat, how="left")   # same agg computed on train, joined onto valid
  For a single stat a window is shorter: train.with_columns(pl.col(num).mean().over(cat).alias(...)).
- Target encoding must be strictly out-of-fold and never fit on full train then transform valid (that leaks).
  Loop KFold splits of the training rows; on each inner-train split compute per-category target stats with
  train[inner_tr].group_by(cat).agg(pl.col(target).mean(), pl.len()), build {category: smoothed_mean} dicts
  where smoothed = (cat_mean * n + global_mean * SMOOTH) / (n + SMOOTH), and write the encoded values back
  onto the inner-validation rows only. For valid: compute the same stats on the full train fold and map with
  the smoothed dict, default = global_mean for unseen categories.
- Add interaction features — ratios/products/differences of the most predictive numerics:
    train = train.with_columns((pl.col(a) / (pl.col(b) + 1e-6)).alias(f"{a}_over_{b}"))
- For a stacking ensemble on rmse/mae/rmsle/auc/logloss: ensemble_spec with members lgbm + xgboost + catboost
  (each with distinct params), method "stack", meta {"model": "ridge"}.
- param_candidates: return 6-12 genuinely different dicts spanning wide ranges — not a grid, not near-duplicates
  of the current params. Wide num_leaves / max_depth, log-scale learning_rate, subsample and colsample below 1.0,
  non-zero L1/L2.
- Do not re-propose a change whose only effect last time was a higher CV with a worse or flat audit holdout."""
