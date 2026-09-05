"""단일 attempt 실행 루프: strategize -> generate_code -> eval_isolated -> 라벨링/영속화.

CPU 예산은 attempt 전체 기준으로 집행(1회차 소진 시 2회차 스킵). run_attempt_core가
핵심 진입점 — Airflow 프로덕션 모드(defer_promotion=True)와 직접모드(run_cycle) 공유.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_LOG = logging.getLogger(__name__)

import polars as pl

from agents.coder import generate_code
from agents.reflector import AttemptContext, reflect
from agents.strategist import StrategyDecision, strategize
from config.settings import MODEL_CODER, PROMOTE_CONFIRM_SEEDS
from cycle.action_optimizer import get_action_prior, update_bandit
from cycle.error_pitfalls import normalize_error, top_error_pitfalls
from cycle.stagnation import detect_stagnation
from cycle.materialize import materialize_best_pipeline
from cycle.promotion import (
    ConfirmResult,
    PromotionCache,
    confirm_and_measure,
    effective_label,
    leaderboard_ceiling_violation,
    train_data_fingerprint,
)
from evaluator.contract import validate_patch
from evaluator.harness import is_significant_gain, split_audit_holdout
from evaluator.metrics import get as get_metric
from memory.retriever import EmbeddingUnavailableError, search
from runtime.isolate import DEFAULT_CPU_BUDGET_SECS, eval_isolated
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
    holdout: pl.DataFrame | None = None  # audit holdout 10% — 승격 시 1회 측정·기록에만 사용
    cpu_budget_secs: float | None = None  # comp.CPU_BUDGET_SECS 오버라이드, 미설정 시 env/DEFAULT_CPU_BUDGET_SECS


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
    reward_label: str
    is_noop_tie: bool = False


class TrainFingerprintMismatchError(RuntimeError):
    """현재 load_train() 결과가 확정 baseline을 측정한 학습 데이터와 다르다.

    EXTRA_TRAIN_PATHS/MAX_TRAIN_ROWS/DROP_COLS 등 대회 데이터 설정이 바뀌면
    raw.pipelines.cv_score(옛 데이터 기준)와 새 attempt의 cv_score가 비교
    불가능해진다(#258, ADR-040). `bin.establish_baseline --remeasure`로 baseline을
    새 데이터에 재측정하고 raw.competitions.train_fingerprint를 갱신해야 재개된다.
    """


class BaselineSourceMismatchError(RuntimeError):
    """MinIO best_pipeline.py가 raw.pipelines 유효행이 가리키는 병합본과 다르다.

    격리(invalid_reason)나 remeasure로 레지스트리 유효집합이 바뀌었는데 MinIO를
    재구성하지 않으면 promote 게이트(_prev_best, 레지스트리)와 confirm 게이트
    (download_best_pipeline, MinIO blob)가 서로 다른 baseline을 보게 된다(#278,
    ADR-042). `bin.rebuild_best_pipeline`로 blob을 유효행으로 재생성해야 재개된다.
    """


def _train_fingerprint_guard(conn: PgConn, competition_id: str, train90: pl.DataFrame) -> None:
    """train90의 지문이 raw.competitions.train_fingerprint와 어긋나면 승격 게이트를
    멈춘다. 최초 관측(저장값 NULL)이면 현재 지문을 심고 통과한다.

    호출부는 train90(split_audit_holdout 결과)을 넘겨야 한다 — remeasure도 같은
    분할을 거치므로 그때만 지문 비교가 성립한다. holdout이 분리되지 않은
    smoke/direct 경로는 호출하지 않는다.
    """
    fp = train_data_fingerprint(train90)
    row = conn.execute(
        "select train_fingerprint from raw.competitions where competition_id = %s",
        [competition_id],
    ).fetchone()
    stored = row[0] if row else None
    if stored is None:
        conn.execute(
            "update raw.competitions set train_fingerprint = %s where competition_id = %s",
            [fp, competition_id],
        )
        return
    if stored == fp:
        return
    cmd = f"uv run python -m bin.establish_baseline --remeasure --competition {competition_id}"
    reason = (
        f"train_fingerprint 불일치 (#258): 저장 {stored[:12]} != 현재 {fp[:12]}. "
        f"load_train 설정이 바뀌었으면 `{cmd}` 실행 후 재개."
    )
    conn.execute(
        "update raw.competitions"
        " set auto_submit_paused_reason = coalesce(auto_submit_paused_reason, %s)"
        " where competition_id = %s",
        [reason, competition_id],
    )
    _LOG.error("%s", reason)
    raise TrainFingerprintMismatchError(reason)


def _baseline_source_guard(conn: PgConn, competition_id: str) -> None:
    """MinIO best_pipeline.py의 sha256이 레지스트리 최신 유효행의 신뢰 해시와
    어긋나면 승격 게이트를 멈춘다(#278).

    blob이 없으면(콜드스타트) 통과. 최신 유효행에 신뢰 해시가 없으면(레거시 행)
    검증을 건너뛴다 — 미탐지가 오탐(정상 대회를 멈춤)보다 안전하다. submit.py가
    expected_sha256=None을 검증 스킵으로 처리하는 것과 같은 판단.
    """
    blob = _best_pipeline_download(competition_id)
    if blob is None:
        return
    row = conn.execute(
        "select coalesce(p.materialized_sha256, p.pipeline_sha256)"
        " from raw.pipelines p join raw.attempts a using (attempt_id)"
        " where p.competition_id = %s and p.invalid_reason is null"
        " order by a.run_ts desc limit 1",
        [competition_id],
    ).fetchone()
    trusted = row[0] if row else None
    if not trusted:
        return
    blob_sha = hashlib.sha256(blob.encode()).hexdigest()
    if blob_sha == trusted:
        return
    cmd = f"uv run python -m bin.rebuild_best_pipeline --competition {competition_id}"
    reason = (
        f"baseline 소스 불일치 (#278): MinIO blob {blob_sha[:12]} != 레지스트리 "
        f"유효행 {trusted[:12]}. 격리/remeasure 후 `{cmd}` 실행 후 재개."
    )
    conn.execute(
        "update raw.competitions"
        " set auto_submit_paused_reason = coalesce(auto_submit_paused_reason, %s)"
        " where competition_id = %s",
        [reason, competition_id],
    )
    _LOG.error("%s", reason)
    raise BaselineSourceMismatchError(reason)


def _prev_best(conn: PgConn, competition_id: str) -> float | None:
    """확정 파이프라인(raw.pipelines, cross-seed+holdout 확인 통과분)의 cv_score.

    확정 파이프라인이 없으면 None — 재측정 없는 attempt 최고값으로 폴백하지 않는다
    (decisions.md ADR-025). 콜드스타트 대응은 establish_bootstrap_baseline과
    bin/establish_baseline.py가 실제 재검증으로 처리한다.

    #254 백필이 재현 불가로 판정한 행(materialized_origin='unverifiable:*')은 제외한다 —
    그 cv_score는 승격 당시(옛 데이터/로직) 기준이라 baseline으로 못 쓰는데 invalid_reason은
    안 세우므로(이력 보존, ADR-039) 여기서 명시적으로 거른다. _prev_best_params/
    _prev_best_fold_scores/establish_bootstrap_baseline도 동일.
    """
    row = conn.execute(
        """
        select max(c.metric_sign * p.cv_score) * max(c.metric_sign)
        from raw.pipelines p
        join raw.competitions c using (competition_id)
        where p.competition_id = %s
          and p.cv_score is not null
          and p.invalid_reason is null
          and coalesce(p.materialized_origin, '') not like 'unverifiable:%%'
        """,
        [competition_id],
    ).fetchone()
    return row[0] if row else None


def _prev_best_params(conn: PgConn, competition_id: str) -> dict | None:
    """확정 파이프라인(raw.pipelines)에 연결된 attempt의 params.

    hyperparam_search 훅이 ctx.best_params로 현재 best 근방 로컬 서치를 할 수 있도록
    advisory로 제공. 훅이 참고 안 해도 무해 — 강제 소비 아님.
    """
    row = conn.execute(
        """
        select a.params
        from raw.pipelines p
        join raw.competitions c using (competition_id)
        join raw.attempts a using (attempt_id)
        where p.competition_id = %s
          and p.cv_score is not null
          and p.invalid_reason is null
          and coalesce(p.materialized_origin, '') not like 'unverifiable:%%'
        order by c.metric_sign * p.cv_score desc
        limit 1
        """,
        [competition_id],
    ).fetchone()
    if not row or not row[0]:
        return None
    val = row[0]
    return val if isinstance(val, dict) else json.loads(val)


def _latest_tuned_params(conn: PgConn, competition_id: str) -> dict | None:
    """가장 최근 튜닝 실행(evaluator/tuner.py, #230, raw.tuned_params)의 결과를
    ctx.tuned_params advisory로 제공한다. model_spec/build_model 훅이 참고할 수 있는
    후보 목록 — best_params와 동일하게 강제 소비 아님. 개선(improved=True)된 항목이
    하나도 없으면 advisory로 넘길 가치가 없어 None(원본보다 나쁜 params를 권하지 않음).
    """
    run_row = conn.execute(
        "select tuning_run_id from raw.tuned_params where competition_id = %s"
        " order by created_at desc limit 1",
        [competition_id],
    ).fetchone()
    if not run_row:
        return None
    rows = conn.execute(
        "select model_type, member_index, params, cv_score, improved from raw.tuned_params"
        " where tuning_run_id = %s order by member_index nulls first",
        [run_row[0]],
    ).fetchall()
    entries = [
        {
            "model": model_type,
            "member_index": member_index,
            "params": params if isinstance(params, dict) else json.loads(params),
            "cv_score": cv_score,
            "improved": improved,
        }
        for model_type, member_index, params, cv_score, improved in rows
    ]
    if not any(e["improved"] for e in entries):
        return None
    return {"entries": entries}


def _prev_best_fold_scores(conn: PgConn, competition_id: str) -> list[float] | None:
    """확정 파이프라인(raw.pipelines)에 연결된 attempt의 fold_scores.

    paired per-fold 유의성 검정(is_significant_gain)의 baseline으로 쓰인다.
    같은 seed로 생성된 fold split은 결정적이라 candidate의 fold_scores와 인덱스별로
    바로 대응시킬 수 있다.

    확정 파이프라인이 없으면 None — 재측정 없는 attempt 최고값의 fold_scores로
    폴백하지 않는다(decisions.md ADR-025).
    """
    row = conn.execute(
        """
        select a.fold_scores
        from raw.pipelines p
        join raw.competitions c using (competition_id)
        join raw.attempts a using (attempt_id)
        where p.competition_id = %s
          and p.cv_score is not null
          and p.invalid_reason is null
          and coalesce(p.materialized_origin, '') not like 'unverifiable:%%'
        order by c.metric_sign * p.cv_score desc
        limit 1
        """,
        [competition_id],
    ).fetchone()
    if not row or not row[0]:
        return None
    val = row[0]
    return val if isinstance(val, list) else json.loads(val)


def establish_bootstrap_baseline(
    conn: PgConn,
    competition_id: str,
    train: pl.DataFrame,
    target_col: str,
    metric: str,
    n_splits: int,
    is_classification: bool,
    cpu_budget_secs: float | None = None,
) -> bool:
    """bootstrap 배치 종료 시 최고 attempt를 BasePipeline 대비 검증해 확정 baseline으로 승격한다.

    확정 파이프라인이 하나도 없는 신규 대회를 위한 콜드스타트 대응(decisions.md
    ADR-025) — bootstrap 배치 끝에 최고 attempt를 실제로 cross-seed confirm +
    holdout 게이트(confirm_and_measure, best_source=None → BasePipeline 대비)를
    통과시켜야만 baseline이 된다.

    이미 확정 파이프라인이 있으면(재부트스트랩 등) 아무것도 하지 않고 False 반환.
    반환값은 이번 호출로 새로 baseline이 확립됐는지 여부.
    """
    existing = conn.execute(
        "select 1 from raw.pipelines where competition_id = %s and invalid_reason is null"
        " and coalesce(materialized_origin, '') not like 'unverifiable:%%' limit 1",
        [competition_id],
    ).fetchone()
    if existing:
        return False

    row = conn.execute(
        """
        select a.attempt_id, a.cv_score, a.code_path, a.fold_scores
        from raw.attempts a
        join raw.competitions c using (competition_id)
        where a.competition_id = %s
          and a.cv_score is not null
          and a.error_trace is null
        order by c.metric_sign * a.cv_score desc
        limit 1
        """,
        [competition_id],
    ).fetchone()
    if not row:
        return False
    attempt_id, cv_score, code_path, fold_scores = row
    if not code_path:
        return False

    content = _code_download(code_path) or ""
    sep = _CODE_HEADER_SEP + "\n"
    source = content.split(sep, 1)[1].strip() if sep in content else content.strip()
    if not source:
        return False

    train90, holdout10 = split_audit_holdout(train, target_col, is_classification)

    confirm = confirm_and_measure(
        source=source,
        best_source=None,
        train90=train90,
        holdout10=holdout10,
        target_col=target_col,
        metric=metric,
        n_splits=n_splits,
        seed=42,
        is_classification=is_classification,
        confirm_seeds=PROMOTE_CONFIRM_SEEDS,
        cache=PromotionCache(conn),
        competition_id=competition_id,
        candidate_cv=cv_score,
        candidate_fold_scores=fold_scores,
        cpu_budget_sec=cpu_budget_secs,
        conn=conn,
    )
    if confirm.holdout_score is not None:
        conn.execute(
            "update raw.attempts set holdout_score = %s where attempt_id = %s",
            [confirm.holdout_score, attempt_id],
        )
    if confirm.seed_gains:
        conn.execute(
            "update raw.attempts set confirm_seed_gains = %s where attempt_id = %s",
            [json.dumps(confirm.seed_gains), attempt_id],
        )
    if not confirm.confirmed:
        reason = "holdout 악화" if confirm.holdout_regressed else "cross-seed 미재현"
        _LOG.info("bootstrap baseline 미확립 — %s (%s)", competition_id, reason)
        return False

    fp_row = conn.execute(
        "select fingerprint from raw.competitions where competition_id = %s",
        [competition_id],
    ).fetchone()
    fp_val = fp_row[0] if fp_row and fp_row[0] else {}
    fp_dict = fp_val if isinstance(fp_val, dict) else json.loads(fp_val)

    materialized = materialize_best_pipeline(None, source)
    pipeline_sha256 = hashlib.sha256(materialized.encode()).hexdigest()
    with conn.transaction():
        insert_pipeline(
            conn,
            pipeline_id=str(uuid.uuid4()),
            attempt_id=attempt_id,
            competition_id=competition_id,
            fingerprint_snapshot=fp_dict,
            code=source,
            cv_score=cv_score,
            gain_vs_best=None,
            pipeline_sha256=pipeline_sha256,
            materialized_code=materialized,
        )
        # 이 baseline이 측정된 train90 지문을 심는다 — 이후 load_train 설정이 바뀌면
        # cycle 게이트(_train_fingerprint_guard)가 옛 cv_score 재사용을 막는다(ADR-040).
        conn.execute(
            "update raw.competitions set train_fingerprint = %s"
            " where competition_id = %s and train_fingerprint is null",
            [train_data_fingerprint(train90), competition_id],
        )
    _best_pipeline_upload(competition_id, materialized)
    _LOG.info(
        "bootstrap baseline 확립 — competition=%s cv=%.6f attempt=%s",
        competition_id, cv_score, attempt_id[:8],
    )
    return True


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


def _retrieval_scores(lessons: list[dict]) -> list[float | None] | None:
    """lessons의 score 목록. failure-lesson 채널은 score 키가 없음(.get 필수)."""
    return [l.get("score") for l in lessons] or None


def _resource_kill_feedback(error_trace: str, cpu_budget_sec: float) -> str:
    """워치독이 리소스 상한 초과로 강제종료한 에러는 원문(rc=-9 등) 대신 실행
    가능한 지시로 바꿔 재생성 피드백에 넘긴다. 원문은 "이유 모르게 죽었다"로만
    읽혀 재시도가 비슷하게 비싼 코드를 다시 쓰는 낭비를 낳았다(2026-08 실측:
    CPU kill attempt 113건이 예외 없이 이 경로로 재시도했고 전부 같은 자리에서
    다시 실패)."""
    if error_trace.startswith("cpu budget exceeded"):
        return (
            f"이 파이프라인은 CPU 예산 {cpu_budget_sec:.0f}초를 초과해 강제 종료됐다"
            "(코드 버그 아님). n_estimators/iterations, n_splits, 하이퍼파라미터"
            " 탐색 후보 수를 줄여 더 싼 파이프라인을 써라."
        )
    if error_trace.startswith("memory watchdog"):
        return (
            "이 파이프라인은 메모리 상한을 초과해 강제 종료됐다(코드 버그 아님)."
            " 배치 크기, 피처 수, 모델 복잡도를 줄이거나 청크 처리로 메모리"
            " 사용량을 낮춰라."
        )
    return error_trace



def run_attempt_core(
    conn: PgConn,
    config: CycleConfig,
    lessons: list[dict],
    prev_best_cv: float | None,
    super_cycle_id: str | None = None,
    was_promoted: bool | None = None,
    attempt_index: int | None = None,
    forced_action: str | None = None,
    defer_promotion: bool = False,
) -> _AttemptData:
    """Strategize → Generate → Evaluate → Persist one attempt. Returns data needed for reflect."""
    if config.holdout is not None:
        _train_fingerprint_guard(conn, config.competition_id, config.train)
        _baseline_source_guard(conn, config.competition_id)
    attempt_id = str(uuid.uuid4())
    attempt_start = time.monotonic()

    dynamic_ctx = _dynamic_eda_context(conn, config.competition_id, prev_best_cv)
    enriched_eda = config.eda_card + dynamic_ctx
    stagnation = detect_stagnation(conn, config.competition_id)
    _LOG.info(
        "start attempt_id=%s super_cycle=%s idx=%s stage=%s prev_best=%s n_lessons=%d stagnant=%s",
        attempt_id[:8],
        super_cycle_id[:8] if super_cycle_id else "-",
        attempt_index if attempt_index is not None else "-",
        config.stage, prev_best_cv, len(lessons),
        stagnation.is_stagnant if stagnation else False,
    )
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
    _LOG.info("strategize done in %.1fs", time.monotonic() - _t_strategize)
    if config.stage == "bootstrap" and config.seed_code:
        prev_code: str | None = config.seed_code
    else:
        prev_code = _load_best_pipeline(config.competition_id)
    action_type = "bootstrap" if (config.stage == "bootstrap" and not prev_code) else decision.action_type

    _t_codegen = time.monotonic()
    _MAX_CODE_RETRIES = 2
    pitfalls = top_error_pitfalls(conn, config.competition_id, action_type)
    known_errors = [f"{sig} (seen {cnt}x)" for sig, cnt in pitfalls] or None
    if known_errors:
        _LOG.info("pitfalls injected (%d): %s", len(known_errors), "; ".join(known_errors))
    gen_kwargs: dict = dict(
        hypothesis=decision.hypothesis,
        action_type=action_type,
        eda_card=config.eda_card,
        prev_code=prev_code,
        known_errors=known_errors,
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
            _LOG.info("static error (%d violation(s)) → regenerating (retry %d)", len(errors), _i + 1)
            source = generate_code(**gen_kwargs, error_feedback=feedback)
            retries += 1
        else:
            error_trace = feedback

    _LOG.info("codegen done in %.1fs retries=%d", time.monotonic() - _t_codegen, retries)

    _LOG.info("evaluating (n_splits=%d metric=%s)...", config.n_splits, config.metric)
    _t_eval = time.monotonic()
    cv_score = None
    cv_fold_var = 0.0
    label = "regression"
    gain_vs_best = None
    gain_vs_best_relative = None
    feature_importance: dict | None = None
    is_noop_tie = False
    fold_scores: list[float] | None = None
    selected_params: dict | None = None
    peak_rss_bytes: int | None = None
    peak_cpu_sec: float | None = None
    model_type: str | None = None

    if not error_trace:
        # CPU 예산은 eval 회차가 아니라 attempt 전체 기준으로 집행한다 — 과거엔
        # 회차마다 독립적으로 900초를 줬는데, 무의미한 rc=-9 피드백으로 재생성한
        # 2회차도 같은 자리에서 또 900초를 태워 attempt 하나가 최대 ~1800초까지
        # 갔다(2026-08 실측: CPU kill attempt 113건 중앙값 971초). 1회차가 예산을
        # 다 쓰면 2회차는 애초에 돌리지 않는다 — 최악 소모가 절반으로 줄고, 다
        # 못 쓴 나머지 예산은 그대로 2회차에 넘어가 재시도가 낭비가 아니라
        # 실제 성공 기회가 된다(피드백도 아래에서 실행 가능한 지시로 바꾼다).
        cpu_budget_total = (
            config.cpu_budget_secs
            if config.cpu_budget_secs is not None
            else float(os.environ.get("EVAL_CPU_BUDGET_SECS", str(DEFAULT_CPU_BUDGET_SECS)))
        )
        cpu_spent = 0.0
        for _eval_i in range(2):
            cpu_remaining = cpu_budget_total - cpu_spent
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
                best_params=_prev_best_params(conn, config.competition_id),
                tuned_params=_latest_tuned_params(conn, config.competition_id),
                cpu_budget_sec=cpu_remaining,
            )
            peak_rss_bytes = iso.peak_rss_bytes
            peak_cpu_sec = iso.peak_cpu_sec
            cpu_spent += iso.peak_cpu_sec or 0.0
            if not iso.error_trace:
                cv_score = iso.cv_score
                cv_fold_var = iso.cv_fold_var or 0.0
                label = iso.label or "regression"
                gain_vs_best = iso.gain_vs_best
                gain_vs_best_relative = iso.gain_vs_best_relative
                feature_importance = iso.feature_importance
                is_noop_tie = iso.is_noop_tie
                fold_scores = iso.fold_scores
                selected_params = iso.selected_params
                model_type = iso.model_type
                gain_str = f"{gain_vs_best:+.6f}" if gain_vs_best is not None else "N/A"
                _LOG.info(
                    "eval ok in %.1fs cv=%.6f fold_var=%.6f gain=%s label=%s",
                    time.monotonic() - _t_eval, cv_score, cv_fold_var, gain_str, label,
                )
                if is_noop_tie:
                    _LOG.warning(
                        "no-op tie: cv_score exactly matches prev_best "
                        "(action=%s) — patch made no effective change",
                        action_type,
                    )
                break
            _LOG.warning("eval error (try %d) → regenerating: %s",
                         _eval_i + 1, (iso.error_trace or "")[:120])
            if _eval_i == 0 and cpu_budget_total - cpu_spent <= 0:
                # 1회차가 예산을 이미 다 태웠으면 재생성 자체를 하지 않는다 —
                # 남은 예산 0으로 재시도해봐야 즉시 다시 죽으므로, 규정 위반
                # 없이도 LLM 호출과 2회차 eval을 통째로 아낀다.
                _LOG.warning(
                    "cpu budget exhausted after try 1 (spent %.0fs of %.0fs) — "
                    "skipping retry", cpu_spent, cpu_budget_total,
                )
                error_trace = iso.error_trace
                break
            if _eval_i == 0:
                feedback = _resource_kill_feedback(iso.error_trace, cpu_remaining)
                source = generate_code(**gen_kwargs, error_feedback=feedback)
                retries += 1
                static_errs = validate_patch(source, action_type)
                if static_errs:
                    error_trace = "\n".join(static_errs)
                    break
            else:
                error_trace = iso.error_trace

    if not error_trace and cv_score is not None:
        ceiling_reason = leaderboard_ceiling_violation(conn, config.competition_id, cv_score)
        if ceiling_reason is not None:
            error_trace = ceiling_reason
            _LOG.warning("%s — attempt 격리(promotion 이전 단계, #288)", ceiling_reason)

    if error_trace:
        label = "error"
        _LOG.warning("failed — %s", error_trace[:200])

    # label의 jump 판정을 promotion 게이트(is_significant_gain, paired
    # per-fold t-test)와 동일 기준으로 통일한다. harness의 절대-마진 jump
    # (LABEL_Z*fold_std)는 수렴한 대회에서 사실상 도달 불가해 실제 승격 attempt도
    # 전부 label=neutral로 남았고, 그 결과 bandit/stagnation/reflection 전부가
    # "성공 신호 0"으로 굳어 있었다. 여기서 확정해야 insert_attempt에 반영된다.
    _, _metric_sign, _ = get_metric(config.metric)
    _significant = (
        is_significant_gain(
            gain_vs_best, cv_fold_var,
            candidate_fold_scores=fold_scores,
            baseline_fold_scores=_prev_best_fold_scores(conn, config.competition_id),
            metric_sign=_metric_sign,
        )
        if not error_trace
        else False
    )
    if not error_trace:
        if _significant:
            label = "jump"
        elif label == "jump":
            # harness 절대-마진 기준은 통과했지만 paired 유의성은 미달 — 강등.
            label = "neutral"

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

    duration_sec = time.monotonic() - attempt_start
    row: dict = {
        "attempt_id":       attempt_id,
        "competition_id":   config.competition_id,
        "run_ts":           datetime.now(timezone.utc),
        "stage":            config.stage,
        "hypothesis":       decision.hypothesis,
        "action_type":      action_type,
        "reflection_ids":   decision.reflection_ids or None,
        "retrieval_scores": _retrieval_scores(lessons),
        "retrieved_ids":    [l["reflection_id"] for l in lessons] or None,
        "cv_score":         cv_score,
        "cv_fold_var":      cv_fold_var,
        "label":            label,
        "gain_vs_best":     gain_vs_best,
        "gain_vs_best_relative": gain_vs_best_relative,
        "error_trace":      error_trace,
        "error_signature":  normalize_error(error_trace) if error_trace else None,
        "duration_sec":     round(duration_sec, 1),
        "peak_rss_bytes":   peak_rss_bytes,
        "peak_cpu_sec":     peak_cpu_sec,
        "code_path":        str(code_path),
        "retries":          retries,
        # 다음 attempt/승격 게이트가 이 attempt의 fold_scores/params를
        # 참고할 수 있도록 영속화 (이전엔 EvalResult 안에서만 존재하고 버려졌음).
        "fold_scores":      json.dumps(fold_scores) if fold_scores is not None else None,
        "params":           json.dumps(selected_params) if selected_params else None,
        "model_type":       model_type,
    }
    if super_cycle_id is not None:
        row["super_cycle_id"] = super_cycle_id
    if was_promoted is not None:
        row["was_promoted"] = was_promoted
    elif super_cycle_id is not None and defer_promotion:
        # promote task가 winner만 나중에 True로 뒤집는다. NULL로 두면
        # `reflection_impact`(store/schema.sql) 뷰가 `IS NOT FALSE`로 NULL을
        # "legacy(승격됨)"로 취급하는 기존 관례 때문에, gate(#203)가 있는
        # sequence에서 늦게 도착한 attempt가 영구 NULL로 남아 승자로 잘못
        # 집계된다(#205) — False로 시작해 promote가 확정할 때만 뒤집는다.
        row["was_promoted"] = False
    insert_attempt(conn, row)

    # defer_promotion=True: caller (super_cycle / Airflow promote task) handles winner-only promotion.
    # _significant은 위에서 label 확정 시 이미 계산됨 — 재계산하지 않음.
    # confirm은 아래 update_bandit 직전에 effective_label(label, confirm)로 다시
    # 참조된다 — 블록을 안 타는 경우(defer_promotion/미유의/에러) None으로 초기화.
    confirm: ConfirmResult | None = None
    if not defer_promotion and _significant and not error_trace:
        confirm = confirm_and_measure(
            source=source,
            best_source=prev_code,
            train90=config.train,
            holdout10=config.holdout,
            target_col=config.target_col,
            metric=config.metric,
            n_splits=config.n_splits,
            seed=config.seed,
            is_classification=config.is_classification,
            confirm_seeds=PROMOTE_CONFIRM_SEEDS,
            action_type=action_type,
            cache=PromotionCache(conn),
            competition_id=config.competition_id,
            candidate_cv=cv_score,
            candidate_fold_scores=fold_scores,
            cpu_budget_sec=config.cpu_budget_secs,
            conn=conn,
        )
        if confirm.holdout_score is not None:
            conn.execute(
                "UPDATE raw.attempts SET holdout_score = %s WHERE attempt_id = %s",
                [confirm.holdout_score, attempt_id],
            )
        if confirm.seed_gains:
            conn.execute(
                "UPDATE raw.attempts SET confirm_seed_gains = %s WHERE attempt_id = %s",
                [json.dumps(confirm.seed_gains), attempt_id],
            )
        if confirm.confirmed:
            fp_row = conn.execute(
                "select fingerprint from raw.competitions where competition_id = %s",
                [config.competition_id],
            ).fetchone()
            fp_val = fp_row[0] if fp_row and fp_row[0] else {}
            fp_dict = fp_val if isinstance(fp_val, dict) else json.loads(fp_val)
            # materialize 먼저 → 해시는 실제 MinIO에 올라가는 내용(submit.py가 exec하는
            # 그 문자열) 기준이어야 한다. raw.pipelines.code(winner source)와는
            # 다른 문자열이므로 순서를 바꿔 sha256을 insert_pipeline에 함께 기록한다.
            materialized = materialize_best_pipeline(prev_code, source)
            pipeline_sha256 = hashlib.sha256(materialized.encode()).hexdigest()

            # OOF 확보 — bin/run_promote_task.py의 merge-verify와 동일 패턴
            # (materialized를 1회 재평가하는 김에 collect_oof=True로 얹는다,
            # 추가 eval 아님). 소비처였던 bin/blend.py는 #231로 폐기됐지만
            # raw.pipelines.oof_preds 자체는 유지한다(향후 분석 재료). 이 경로는
            # 기존에 merge-verify 게이트가 없었으므로 실패해도 승격 자체는 막지
            # 않고 oof_preds만 비운다 — best-effort.
            merge_oof_preds = None
            try:
                merge_eval = eval_isolated(
                    source=materialized,
                    train=config.train,
                    target_col=config.target_col,
                    metric=config.metric,
                    prev_best=None,
                    n_splits=config.n_splits,
                    seed=config.seed,
                    is_classification=config.is_classification,
                    collect_oof=True,
                    cpu_budget_sec=config.cpu_budget_secs,
                )
                if not merge_eval.error_trace and merge_eval.cv_score is not None:
                    merge_oof_preds = merge_eval.oof_preds
            except Exception as exc:
                _LOG.warning("merge-verify OOF 수집 실패(무시하고 계속): %s", exc)

            with conn.transaction():
                insert_pipeline(
                    conn,
                    pipeline_id=str(uuid.uuid4()),
                    attempt_id=attempt_id,
                    competition_id=config.competition_id,
                    fingerprint_snapshot=fp_dict,
                    code=source,
                    cv_score=cv_score,
                    gain_vs_best=gain_vs_best,
                    pipeline_sha256=pipeline_sha256,
                    oof_preds=merge_oof_preds,
                    materialized_code=materialized,
                )
            _best_pipeline_upload(config.competition_id, materialized)
            _LOG.info("best pipeline materialized (gain=%+.5f)", gain_vs_best)
        else:
            reason = "holdout 악화" if confirm.holdout_regressed else "cross-seed 미확인"
            _LOG.info("%s — 승격 스킵 (gain=%+.5f)", reason, gain_vs_best)

    _LOG.info(
        "persist done — total %.1fs attempt_id=%s action=%s label=%s retries=%d",
        duration_sec, attempt_id[:8], action_type, label, retries,
    )

    # confirm이 jump를 거부했으면(cross-seed 미재현/holdout 악화) 보상 신호를
    # regression으로 다운그레이드 — #164, effective_label 참고.
    reward_label = effective_label(label, confirm)
    if config.stage == "reflexion":
        update_bandit(
            conn,
            competition_id=config.competition_id,
            action_type=action_type,
            label=reward_label,
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
        reward_label=reward_label,
        is_noop_tie=is_noop_tie,
    )


def _do_reflect(conn: PgConn, competition_id: str, data: _AttemptData) -> str | None:
    """Reflect if label warrants it, or if a no-op tie was detected —
    otherwise these zero-effect attempts silently bypass the label gate (label=neutral,
    error_trace=None) and the strategist never learns why the action had no effect.

    data.reward_label(jump/regression 둘 다 이 게이트를 통과하므로 판정 자체는
    안 바뀐다)을 쓴다 — confirm이 jump를 거부한 경우 lesson도 "CV에서는 좋아
    보였지만 실제 검증은 통과 못 했다"는 correction된 신호를 받아야 한다(#164).
    """
    if data.reward_label not in ("jump", "regression") and data.error_trace is None and not data.is_noop_tie:
        return None
    ctx = AttemptContext(
        hypothesis=data.decision.hypothesis,
        action_type=data.action_type,
        code=data.source,
        cv_score=data.cv_score or 0.0,
        cv_fold_var=data.cv_fold_var,
        gain_vs_best=data.gain_vs_best,
        label=data.reward_label,
        retrieved_ids=data.decision.reflection_ids,
        feature_importance=data.feature_importance,
        error_trace=data.error_trace,
        is_noop_tie=data.is_noop_tie,
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
    fail_summary = _recent_failure_summary(conn, config.competition_id)
    query = _build_retrieval_query(conn, config.competition_id, config.eda_card, fail_summary)
    try:
        lessons = search(conn, query, config.competition_id, k=config.k_retrieve)
    except EmbeddingUnavailableError as exc:
        _LOG.warning("embedding unavailable, proceeding with no lessons: %s", exc)
        lessons = []
    prev_best_cv = _prev_best(conn, config.competition_id)

    data = run_attempt_core(conn, config, lessons, prev_best_cv)

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
