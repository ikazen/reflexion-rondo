from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

from agents.coder import generate_code
from agents.reflector import AttemptContext, reflect
from agents.strategist import strategize
from config.settings import MODEL_CODER
from cycle.stagnation import detect_stagnation
from evaluator.contract import validate_code
from memory.retriever import search
from runtime.isolate import eval_isolated
from store.db import insert_attempt, insert_pipeline

RUNS_CODE_DIR = Path(__file__).parent.parent / "runs" / "code"
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


@dataclass
class CycleResult:
    attempt_id: str
    cv_score: float | None
    label: str
    gain_vs_best: float | None
    retries: int
    reflection_id: str | None
    error_trace: str | None
    code_path: Path


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
) -> Path:
    """생성 코드를 로컬에 떨군다 — 사람이 품질 검토 후 reflection 반영 여부 판단용.

    runs/ 는 gitignore. coder 모델 교체 전후 비교를 위해 헤더에 모델명·CV를 박는다.
    """
    out_dir = RUNS_CODE_DIR / competition_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{ts}_{attempt_id[:8]}.py"
    header = (
        f"# attempt_id:   {attempt_id}\n"
        f"# coder_model:  {MODEL_CODER}\n"
        f"# stage:        {stage}  action_type: {action_type}\n"
        f"# cv_score:     {cv_score}  gain_vs_best: {gain_vs_best}\n"
        f"# error:        {'yes' if error_trace else 'no'}\n"
        f"# hypothesis:   {' '.join(hypothesis.split())}\n"
        f"{_CODE_HEADER_SEP}\n"
    )
    path.write_text(header + source, encoding="utf-8")
    return path


def _best_code(conn: duckdb.DuckDBPyConnection, competition_id: str) -> str | None:
    """1변경 규율의 기준점: best(에러 없는) attempt의 저장 코드 본문. 없으면 None."""
    row = conn.execute(
        """
        select a.code_path
        from raw.attempts a
        join raw.competitions c using (competition_id)
        where a.competition_id = ?
          and a.cv_score is not null
          and a.error_trace is null
          and a.code_path is not null
        order by c.metric_sign * a.cv_score desc
        limit 1
        """,
        [competition_id],
    ).fetchone()
    if not row or not row[0]:
        return None
    path = Path(row[0])
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8")
    sep = _CODE_HEADER_SEP + "\n"
    if sep in content:  # 저장 시 붙인 주석 헤더 제거 → 순수 소스만
        content = content.split(sep, 1)[1]
    return content.strip() or None



def run_cycle(
    conn: duckdb.DuckDBPyConnection,
    config: CycleConfig,
) -> CycleResult:
    attempt_id = str(uuid.uuid4())
    cycle_start = time.monotonic()

    # 1. Retrieve
    query = _last_hypothesis(conn, config.competition_id) or config.eda_card
    lessons = search(conn, query, config.competition_id, k=config.k_retrieve)

    # 2. Strategize
    prev_best_cv = _prev_best(conn, config.competition_id)
    stagnation = detect_stagnation(conn, config.competition_id)
    decision = strategize(
        eda_card=config.eda_card,
        lessons=lessons,
        stage=config.stage,
        prev_best_cv=prev_best_cv,
        stagnation=stagnation,
    )

    # 3. Generate code + validate (정적 검사) + Docker 격리 실행 (최대 2회 재시도)
    # bootstrap은 1변경 규율 면제(§4) → from-scratch. 그 외엔 best 파이프라인을 한 군데만 수정.
    _MAX_CODE_RETRIES = 2
    if config.stage == "bootstrap" and config.seed_code:
        prev_code: str | None = config.seed_code
    elif config.stage == "bootstrap":
        prev_code = None
    else:
        prev_code = _best_code(conn, config.competition_id)
    gen_kwargs: dict = dict(
        hypothesis=decision.hypothesis,
        action_type=decision.action_type,
        eda_card=config.eda_card,
        prev_code=prev_code,
    )
    source = generate_code(**gen_kwargs)
    retries = 0
    error_trace: str | None = None

    # Phase 1: AST 정적 검사 — Docker 실행 전 구문/금지 패턴 차단
    for _i in range(_MAX_CODE_RETRIES + 1):
        errors = validate_code(source)
        if not errors:
            break
        feedback = "\n".join(errors)
        if _i < _MAX_CODE_RETRIES:
            source = generate_code(**gen_kwargs, error_feedback=feedback)
            retries += 1
        else:
            error_trace = feedback

    # 5. Evaluate — Docker 격리 실행 (exec + CV 평가 전부 컨테이너 안)
    cv_score = None
    cv_fold_var = 0.0
    label = "regression"
    gain_vs_best = None

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
            )
            if not iso.error_trace:
                cv_score = iso.cv_score
                cv_fold_var = iso.cv_fold_var or 0.0
                label = iso.label or "regression"
                gain_vs_best = iso.gain_vs_best
                break
            if _eval_i == 0:
                source = generate_code(**gen_kwargs, error_feedback=iso.error_trace)
                retries += 1
                static_errs = validate_code(source)
                if static_errs:
                    error_trace = "\n".join(static_errs)
                    break
            else:
                error_trace = iso.error_trace

    # 5b. 생성 코드 로컬 저장 (사람 검토용 — reflection 반영 여부는 사람이 판단)
    code_path = _save_code(
        source,
        competition_id=config.competition_id,
        attempt_id=attempt_id,
        stage=config.stage,
        hypothesis=decision.hypothesis,
        action_type=decision.action_type,
        cv_score=cv_score,
        gain_vs_best=gain_vs_best,
        error_trace=error_trace,
    )

    # 6. Persist attempt
    duration_sec = time.monotonic() - cycle_start
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
        "duration_sec":     round(duration_sec, 1),
        "code_path":        str(code_path),
        "retries":          retries,
    })

    # 6b. gain_vs_best > 0 이면 raw.pipelines 저장 (cold-start 시드 후보)
    if gain_vs_best is not None and gain_vs_best > 0 and not error_trace:
        fp_row = conn.execute(
            "select fingerprint from raw.competitions where competition_id = ?",
            [config.competition_id],
        ).fetchone()
        fp_dict = json.loads(fp_row[0]) if fp_row else {}
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
        retries=retries,
        reflection_id=reflection_id,
        error_trace=error_trace,
        code_path=code_path,
    )
