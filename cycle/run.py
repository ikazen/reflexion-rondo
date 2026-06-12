from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from agents.coder import generate_code
from agents.reflector import AttemptContext, reflect
from agents.strategist import StrategyDecision, strategize
from config.settings import MODEL_CODER
from cycle.action_optimizer import get_action_prior, update_bandit
from cycle.stagnation import detect_stagnation
from cycle.materialize import materialize_best_pipeline
from evaluator.contract import validate_patch
from memory.retriever import EmbeddingUnavailableError, search
from runtime.isolate import eval_isolated
from store.db import PgConn, insert_attempt, insert_pipeline
from store.s3_code import download as _code_download
from store.s3_code import download_best_pipeline as _best_pipeline_download
from store.s3_code import upload as _code_upload
from store.s3_code import upload_best_pipeline as _best_pipeline_upload

_CODE_HEADER_SEP = "# " + "-" * 60  # 저장 헤더와 본문 경계 — _best_code가 이 줄로 헤더를 떼낸다


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
    seed_code: str | None = None
    slug: str | None = None  # S3 경로용 모듈명 (e.g. s4e1), 미설정 시 competition_id fallback


@dataclass
class CycleResult:
    attempt_id: str
    cv_score: float | None
    label: str
    gain_vs_best: float | None
    retries: int
    reflection_id: str | None
    error_trace: str | None
    code_path: str


@dataclass
class _AttemptData:
    """Internal: complete data from one attempt, before reflect."""
    attempt_id: str
    decision: StrategyDecision
    action_type: str
    source: str
    cv_score: float | None
    cv_fold_var: float
    label: str
    gain_vs_best: float | None
    retries: int
    error_trace: str | None
    code_path: str
    feature_importance: dict | None


def _prev_best(conn: PgConn, competition_id: str) -> float | None:
    row = conn.execute(
        """
        select max(c.metric_sign * a.cv_score) * max(c.metric_sign)
        from raw.attempts a
        join raw.competitions c using (competition_id)
        where a.competition_id = %s
          and a.cv_score is not null
        """,
        [competition_id],
    ).fetchone()
    return row[0] if row else None


def _last_hypothesis(conn: PgConn, competition_id: str) -> str | None:
    row = conn.execute(
        """
        select hypothesis from raw.attempts
        where competition_id = %s and hypothesis is not null
        order by run_ts desc limit 1
        """,
        [competition_id],
    ).fetchone()
    return row[0] if row else None


def _dynamic_eda_context(
    conn: PgConn,
    competition_id: str,
    prev_best_cv: float | None,
    window: int = 10,
) -> str:
    """매 사이클 DB를 조회해 EDA 카드 하단에 붙일 동적 컨텍스트를 생성한다."""
    lines: list[str] = ["\n## Current State"]

    best_str = f"{prev_best_cv:.5f}" if prev_best_cv is not None else "none yet"
    lines.append(f"- best CV so far: {best_str}")

    # 최근 window 개 attempt의 action_type 분포
    dist_rows = conn.execute(
        """
        select action_type, count(*) as cnt
        from (
            select action_type from raw.attempts
            where competition_id = %s and cv_score is not null
            order by run_ts desc
            limit %s
        )
        group by action_type
        order by cnt desc
        """,
        [competition_id, window],
    ).fetchall()
    if dist_rows:
        dist_str = ", ".join(f"{r[0]} x{r[1]}" for r in dist_rows)
        lines.append(f"- recent {window} attempts: {dist_str}")

    # 최근 실패 패턴: regression + error 합산, action_type별 횟수
    fail_rows = conn.execute(
        """
        select action_type, count(*) as cnt,
               max(hypothesis) as sample_hyp
        from (
            select action_type, hypothesis from raw.attempts
            where competition_id = %s
              and (label = 'regression' or error_trace is not null)
            order by run_ts desc
            limit %s
        )
        group by action_type
        order by cnt desc
        """,
        [competition_id, window],
    ).fetchall()
    if fail_rows:
        fail_parts = [
            f"{r[0]} ({r[1]}x, e.g. {(r[2] or '')[:50]})"
            for r in fail_rows
        ]
        lines.append(f"- recent failures: {'; '.join(fail_parts)}")

    return "\n".join(lines)


def _recent_failure_summary(
    conn: PgConn,
    competition_id: str,
    window: int = 5,
) -> str:
    rows = conn.execute(
        """
        select action_type, hypothesis
        from raw.attempts
        where competition_id = %s
          and (label = 'regression' or error_trace is not null)
        order by run_ts desc
        limit %s
        """,
        [competition_id, window],
    ).fetchall()
    if not rows:
        return ""
    return "; ".join(f"{r[0]}: {(r[1] or '')[:60]}" for r in rows)


def _build_retrieval_query(
    conn: PgConn,
    competition_id: str,
    eda_card: str,
    fail_summary: str = "",
) -> str:
    last_hyp = _last_hypothesis(conn, competition_id) or ""
    parts = [p for p in [last_hyp, f"avoid: {fail_summary}" if fail_summary else ""] if p]
    return "; ".join(parts) if parts else eda_card


def _save_code(
    source: str,
    *,
    competition_id: str,
    attempt_id: str,
    stage: str,
    hypothesis: str,
    action_type: str,
    cv_score: float | None,
    gain_vs_best: float | None,
    error_trace: str | None,
) -> str:
    """생성 코드를 저장하고 URI를 반환한다 (S3 또는 로컬 경로 fallback)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{attempt_id[:8]}.py"
    header = (
        f"# attempt_id:   {attempt_id}\n"
        f"# coder_model:  {MODEL_CODER}\n"
        f"# stage:        {stage}  action_type: {action_type}\n"
        f"# cv_score:     {cv_score}  gain_vs_best: {gain_vs_best}\n"
        f"# error:        {'yes' if error_trace else 'no'}\n"
        f"# hypothesis:   {' '.join(hypothesis.split())}\n"
        f"{_CODE_HEADER_SEP}\n"
    )
    return _code_upload(competition_id, filename, header + source)


def _load_best_pipeline(competition_id: str) -> str | None:
    """Materialized best pipeline source. None if not yet stored."""
    return _best_pipeline_download(competition_id)



def run_attempt_core(
    conn: PgConn,
    config: CycleConfig,
    lessons: list[dict],
    prev_best_cv: float | None,
    super_cycle_id: str | None = None,
    was_promoted: bool | None = None,
    attempt_index: int | None = None,
    forced_action: str | None = None,
) -> _AttemptData:
    """Strategize → Generate → Evaluate → Persist one attempt. Returns data needed for reflect."""
    attempt_id = str(uuid.uuid4())
    attempt_start = time.monotonic()

    # Strategize
    dynamic_ctx = _dynamic_eda_context(conn, config.competition_id, prev_best_cv)
    enriched_eda = config.eda_card + dynamic_ctx
    stagnation = detect_stagnation(conn, config.competition_id)
    print(f"[attempt] start attempt_id={attempt_id[:8]}"
          f" super_cycle={super_cycle_id[:8] if super_cycle_id else '-'}"
          f" idx={attempt_index if attempt_index is not None else '-'}"
          f" stage={config.stage} prev_best={prev_best_cv} n_lessons={len(lessons)}"
          f" stagnant={stagnation.is_stagnant if stagnation else False}")
    action_prior = get_action_prior(conn, config.competition_id)
    _t_strategize = time.monotonic()
    decision = strategize(
        eda_card=enriched_eda,
        lessons=lessons,
        stage=config.stage,
        prev_best_cv=prev_best_cv,
        stagnation=stagnation,
        action_prior=action_prior,
        forced_action_type=forced_action,
    )
    print(f"[attempt] strategize done in {time.monotonic() - _t_strategize:.1f}s")
    # bootstrap with no established pipeline → generate full pipeline from scratch
    if config.stage == "bootstrap" and config.seed_code:
        prev_code: str | None = config.seed_code
    else:
        prev_code = _load_best_pipeline(config.competition_id)
    action_type = "bootstrap" if (config.stage == "bootstrap" and not prev_code) else decision.action_type

    # Generate + validate
    _t_codegen = time.monotonic()
    _MAX_CODE_RETRIES = 2
    gen_kwargs: dict = dict(
        hypothesis=decision.hypothesis,
        action_type=action_type,
        eda_card=config.eda_card,
        prev_code=prev_code,
    )
    source = generate_code(**gen_kwargs)
    retries = 0
    error_trace: str | None = None

    for _i in range(_MAX_CODE_RETRIES + 1):
        errors = validate_patch(source, action_type)
        if not errors:
            break
        feedback = "\n".join(errors)
        if _i < _MAX_CODE_RETRIES:
            print(f"[attempt] static error ({len(errors)} violation(s)) → regenerating (retry {_i + 1})")
            source = generate_code(**gen_kwargs, error_feedback=feedback)
            retries += 1
        else:
            error_trace = feedback

    print(f"[attempt] codegen done in {time.monotonic() - _t_codegen:.1f}s retries={retries}")

    # Evaluate
    print(f"[attempt] evaluating (n_splits={config.n_splits} metric={config.metric})...")
    _t_eval = time.monotonic()
    cv_score = None
    cv_fold_var = 0.0
    label = "regression"
    gain_vs_best = None
    feature_importance: dict | None = None

    if not error_trace:
        for _eval_i in range(2):
            iso = eval_isolated(
                source=source,
                train=config.train,
                target_col=config.target_col,
                metric=config.metric,
                prev_best=prev_best_cv,
                n_splits=config.n_splits,
                seed=config.seed,
                is_classification=config.is_classification,
                action_type=action_type,
                best_source=prev_code,
            )
            if not iso.error_trace:
                cv_score = iso.cv_score
                cv_fold_var = iso.cv_fold_var or 0.0
                label = iso.label or "regression"
                gain_vs_best = iso.gain_vs_best
                feature_importance = iso.feature_importance
                gain_str = f"{gain_vs_best:+.6f}" if gain_vs_best is not None else "N/A"
                print(f"[attempt] eval ok in {time.monotonic() - _t_eval:.1f}s"
                      f" cv={cv_score:.6f} fold_var={cv_fold_var:.6f}"
                      f" gain={gain_str} label={label}")
                break
            print(f"[attempt] eval error (try {_eval_i + 1}) → regenerating"
                  f": {(iso.error_trace or '')[:120]}")
            if _eval_i == 0:
                source = generate_code(**gen_kwargs, error_feedback=iso.error_trace)
                retries += 1
                static_errs = validate_patch(source, action_type)
                if static_errs:
                    error_trace = "\n".join(static_errs)
                    break
            else:
                error_trace = iso.error_trace

    if error_trace:
        print(f"[attempt] failed — {error_trace[:200]}")

    # Save code
    code_path = _save_code(
        source,
        competition_id=config.slug or config.competition_id,
        attempt_id=attempt_id,
        stage=config.stage,
        hypothesis=decision.hypothesis,
        action_type=action_type,
        cv_score=cv_score,
        gain_vs_best=gain_vs_best,
        error_trace=error_trace,
    )

    # Persist attempt
    duration_sec = time.monotonic() - attempt_start
    row: dict = {
        "attempt_id":       attempt_id,
        "competition_id":   config.competition_id,
        "run_ts":           datetime.now(timezone.utc),
        "stage":            config.stage,
        "hypothesis":       decision.hypothesis,
        "action_type":      action_type,
        "reflection_ids":   decision.reflection_ids or None,
        "retrieval_scores": [l["score"] for l in lessons] or None,
        "cv_score":         cv_score,
        "cv_fold_var":      cv_fold_var,
        "label":            label,
        "gain_vs_best":     gain_vs_best,
        "error_trace":      error_trace,
        "duration_sec":     round(duration_sec, 1),
        "code_path":        str(code_path),
        "retries":          retries,
    }
    if super_cycle_id is not None:
        row["super_cycle_id"] = super_cycle_id
    if was_promoted is not None:
        row["was_promoted"] = was_promoted
    insert_attempt(conn, row)

    # Save pipeline if improved — also materialize as new best baseline
    if gain_vs_best is not None and gain_vs_best > 0 and not error_trace:
        fp_row = conn.execute(
            "select fingerprint from raw.competitions where competition_id = %s",
            [config.competition_id],
        ).fetchone()
        fp_val = fp_row[0] if fp_row and fp_row[0] else {}
        fp_dict = fp_val if isinstance(fp_val, dict) else json.loads(fp_val)
        insert_pipeline(
            conn,
            pipeline_id=str(uuid.uuid4()),
            attempt_id=attempt_id,
            competition_id=config.competition_id,
            fingerprint_snapshot=fp_dict,
            code=source,
            cv_score=cv_score,
            gain_vs_best=gain_vs_best,
        )
        materialized = materialize_best_pipeline(prev_code, source)
        _best_pipeline_upload(config.competition_id, materialized)
        print(f"[attempt] best pipeline materialized (gain={gain_vs_best:+.5f})")

    print(f"[attempt] persist done — total {round(duration_sec, 1)}s"
          f" attempt_id={attempt_id[:8]} action={action_type}"
          f" label={label} retries={retries}")

    # Action bandit update — reflexion stage only (BON-109)
    if config.stage == "reflexion":
        update_bandit(
            conn,
            competition_id=config.competition_id,
            action_type=action_type,
            label=label,
            gain_vs_best=gain_vs_best,
            error_trace=error_trace,
        )

    return _AttemptData(
        attempt_id=attempt_id,
        decision=decision,
        action_type=action_type,
        source=source,
        cv_score=cv_score,
        cv_fold_var=cv_fold_var,
        label=label,
        gain_vs_best=gain_vs_best,
        retries=retries,
        error_trace=error_trace,
        code_path=code_path,
        feature_importance=feature_importance,
    )


def _do_reflect(conn: PgConn, competition_id: str, data: _AttemptData) -> str | None:
    """Reflect if label warrants it (BON-96 gate). Returns reflection_id or None."""
    if data.label not in ("jump", "regression") and data.error_trace is None:
        return None
    ctx = AttemptContext(
        hypothesis=data.decision.hypothesis,
        action_type=data.action_type,
        code=data.source,
        cv_score=data.cv_score or 0.0,
        cv_fold_var=data.cv_fold_var,
        gain_vs_best=data.gain_vs_best,
        label=data.label,
        retrieved_ids=data.decision.reflection_ids,
        feature_importance=data.feature_importance,
        error_trace=data.error_trace,
    )
    try:
        output = reflect(conn, attempt_id=data.attempt_id, competition_id=competition_id, context=ctx)
        return output.reflection_id
    except EmbeddingUnavailableError:
        return None


def run_cycle(
    conn: PgConn,
    config: CycleConfig,
) -> CycleResult:
    # 1. Retrieve
    fail_summary = _recent_failure_summary(conn, config.competition_id)
    query = _build_retrieval_query(conn, config.competition_id, config.eda_card, fail_summary)
    lessons = search(conn, query, config.competition_id, k=config.k_retrieve)
    prev_best_cv = _prev_best(conn, config.competition_id)

    # 2-6. Strategize → Generate → Evaluate → Persist
    data = run_attempt_core(conn, config, lessons, prev_best_cv)

    # 7. Reflect (BON-96 gate inside _do_reflect)
    reflection_id = _do_reflect(conn, config.competition_id, data)

    return CycleResult(
        attempt_id=data.attempt_id,
        cv_score=data.cv_score,
        label=data.label,
        gain_vs_best=data.gain_vs_best,
        retries=data.retries,
        reflection_id=reflection_id,
        error_trace=data.error_trace,
        code_path=data.code_path,
    )
