from __future__ import annotations

import types
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import duckdb
import polars as pl

from agents.coder import generate_code
from agents.reflector import AttemptContext, reflect
from agents.strategist import strategize
from evaluator.contract import validate_code, smoke_test
from evaluator.harness import run as eval_run
from memory.retriever import search
from store.db import insert_attempt


@dataclass(frozen=True, slots=True)
class CycleConfig:
    competition_id: str
    train: pl.DataFrame
    target_col: str
    metric: str
    stage: str
    eda_card: str
    n_splits: int = 5
    seed: int = 42
    k_retrieve: int = 5
    is_classification: bool = True


@dataclass
class CycleResult:
    attempt_id: str
    cv_score: float | None
    label: str
    gain_vs_best: float | None
    reflection_id: str | None
    error_trace: str | None


def _prev_best(conn: duckdb.DuckDBPyConnection, competition_id: str) -> float | None:
    row = conn.execute(
        """
        select max(metric_sign * a.cv_score) * any_value(c.metric_sign)
        from raw.attempts a
        join raw.competitions c using (competition_id)
        where a.competition_id = ?
          and a.cv_score is not null
        """,
        [competition_id],
    ).fetchone()
    return row[0] if row else None


def _last_hypothesis(conn: duckdb.DuckDBPyConnection, competition_id: str) -> str | None:
    row = conn.execute(
        """
        select hypothesis from raw.attempts
        where competition_id = ? and hypothesis is not null
        order by run_ts desc limit 1
        """,
        [competition_id],
    ).fetchone()
    return row[0] if row else None


def _load_functions(source: str) -> tuple[object, object] | str:
    """Exec source and return (feature_fn, model_fn) or error string."""
    ns: dict = {}
    try:
        exec(compile(source, "<generated>", "exec"), ns)  # noqa: S102
    except Exception as exc:
        return str(exc)
    missing = [n for n in ("feature_fn", "model_fn") if n not in ns]
    if missing:
        return f"missing after exec: {missing}"
    return ns["feature_fn"], ns["model_fn"]


def run_cycle(
    conn: duckdb.DuckDBPyConnection,
    config: CycleConfig,
) -> CycleResult:
    attempt_id = str(uuid.uuid4())

    # 1. Retrieve
    query = _last_hypothesis(conn, config.competition_id) or config.eda_card
    lessons = search(conn, query, config.competition_id, k=config.k_retrieve)

    # 2. Strategize
    prev_best_cv = _prev_best(conn, config.competition_id)
    decision = strategize(
        eda_card=config.eda_card,
        lessons=lessons,
        stage=config.stage,
        prev_best_cv=prev_best_cv,
    )

    # 3. Generate code (1 retry on validation failure)
    source = generate_code(
        hypothesis=decision.hypothesis,
        action_type=decision.action_type,
        eda_card=config.eda_card,
    )
    errors = validate_code(source)
    if errors:
        source = generate_code(
            hypothesis=decision.hypothesis,
            action_type=decision.action_type,
            eda_card=config.eda_card,
            error_feedback="\n".join(errors),
        )
        errors = validate_code(source)

    error_trace: str | None = None

    # 4. Load + smoke test
    loaded = None
    if not errors:
        loaded = _load_functions(source)
        if isinstance(loaded, str):
            errors = [loaded]

    if not errors and loaded is not None:
        feature_fn, model_fn = loaded
        smoke_err = smoke_test(feature_fn, model_fn, config.train, config.target_col)
        if smoke_err:
            errors = [smoke_err]

    if errors:
        error_trace = "\n".join(errors)

    # 5. Evaluate
    cv_score = None
    cv_fold_var = 0.0
    label = "regression"
    gain_vs_best = None

    if not error_trace and loaded is not None:
        feature_fn, model_fn = loaded
        result = eval_run(
            train=config.train,
            target_col=config.target_col,
            metric=config.metric,
            feature_fn=feature_fn,
            model_fn=model_fn,
            params={},
            prev_best=prev_best_cv,
            n_splits=config.n_splits,
            seed=config.seed,
            is_classification=config.is_classification,
        )
        cv_score = result.cv_score
        cv_fold_var = result.cv_fold_var
        label = result.label
        gain_vs_best = result.gain_vs_best

    # 6. Persist attempt
    insert_attempt(conn, {
        "attempt_id":       attempt_id,
        "competition_id":   config.competition_id,
        "run_ts":           datetime.now(timezone.utc),
        "stage":            config.stage,
        "hypothesis":       decision.hypothesis,
        "action_type":      decision.action_type,
        "reflection_ids":   decision.reflection_ids or None,
        "retrieval_scores": [l["score"] for l in lessons] or None,
        "cv_score":         cv_score,
        "cv_fold_var":      cv_fold_var,
        "label":            label,
        "gain_vs_best":     gain_vs_best,
        "error_trace":      error_trace,
    })

    # 7. Reflect
    reflection_id: str | None = None
    ctx = AttemptContext(
        hypothesis=decision.hypothesis,
        action_type=decision.action_type,
        code=source,
        cv_score=cv_score or 0.0,
        cv_fold_var=cv_fold_var,
        gain_vs_best=gain_vs_best,
        label=label,
        retrieved_ids=decision.reflection_ids,
        error_trace=error_trace,
    )
    output = reflect(conn, attempt_id=attempt_id, competition_id=config.competition_id, context=ctx)
    reflection_id = output.reflection_id

    return CycleResult(
        attempt_id=attempt_id,
        cv_score=cv_score,
        label=label,
        gain_vs_best=gain_vs_best,
        reflection_id=reflection_id,
        error_trace=error_trace,
    )
