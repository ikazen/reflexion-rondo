"""단일 attempt 실행 루프: strategize -> generate_code -> eval_isolated -> 라벨링/영속화.

CPU 예산은 attempt 전체 기준으로 집행(1회차 소진 시 2회차 스킵). run_attempt_core가
핵심 진입점 — Airflow attempt task(bin/run_attempt_task.py)가 호출하고, 승격은 promote task가 맡는다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

_LOG = logging.getLogger(__name__)

import polars as pl

from agents.coder import generate_code
from agents.strategist import StrategyDecision, strategize
from config.settings import MODEL_CODER, PROMOTE_CONFIRM_SEEDS
from cycle.action_optimizer import get_action_prior, update_bandit
from cycle.error_pitfalls import normalize_error, top_error_pitfalls
from cycle.stagnation import StagnationSignal, detect_stagnation
from cycle.materialize import materialize_best_pipeline
from cycle.promotion import (
    PromotionCache,
    confirm_and_measure,
    eval_semantics_fingerprint,
    leaderboard_ceiling_violation,
    record_confirm,
    train_data_fingerprint,
)
from evaluator.contract import validate_patch
from evaluator.harness import is_significant_gain, split_audit_holdout
from evaluator.metrics import get as get_metric
from runtime.isolate import DEFAULT_CPU_BUDGET_SECS, IsolatedResult, eval_isolated
from store.db import PgConn, competition_fingerprint, insert_attempt, insert_pipeline
from store.s3_code import download as _code_download
from store.s3_code import download_best_pipeline as _best_pipeline_download
from store.s3_code import upload as _code_upload
from store.s3_code import CODE_HEADER_SEP, strip_code_header
from store.s3_code import upload_best_pipeline as _best_pipeline_upload


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
    slug: str | None = None  # S3 경로용 모듈명 (e.g. s4e1), 미설정 시 competition_id fallback
    holdout: pl.DataFrame | None = None  # 값이 있을 때만 run_attempt_core가 평가 전에 fingerprint/baseline 가드를 건다(attempt task가 채움)
    cpu_budget_secs: float | None = None  # comp.CPU_BUDGET_SECS 오버라이드, 미설정 시 env/DEFAULT_CPU_BUDGET_SECS


@dataclass(frozen=True, slots=True)
class _AttemptData:
    """run_attempt_task가 로그로 남기는 attempt 결과 요약."""
    attempt_id: str
    decision: StrategyDecision
    cv_score: float | None
    label: str
    gain_vs_best: float | None
    retries: int
    is_noop_tie: bool = False


class TrainFingerprintMismatchError(RuntimeError):
    """현재 load_train() 결과가 확정 baseline을 측정한 학습 데이터와 다르다.

    EXTRA_TRAIN_PATHS/MAX_TRAIN_ROWS/DROP_COLS 등 대회 데이터 설정이 바뀌면
    raw.pipelines.cv_score(옛 데이터 기준)와 새 attempt의 cv_score가 비교
    불가능해진다(#258, ADR-040). `bin.establish_baseline --remeasure`로 baseline을
    새 데이터에 재측정하고 raw.competitions.train_fingerprint를 갱신해야 재개된다.
    """


class EvalFingerprintMismatchError(RuntimeError):
    """평가 노브(preselect 후보 캡·fold 수·조기중단·분할 seed)가 확정 baseline을 측정할 때와 다르다.

    데이터가 그대로여도 이 값들이 바뀌면 raw.pipelines.cv_score(옛 노브 기준)와 새 attempt의 cv_score가
    비교 불가능해진다 — #311이 후보 캡을 줄여 s6e8이 3일간 전부 regression이었다(#347, #348, ADR-053).
    `bin.establish_baseline --remeasure`로 baseline을 재측정하고 raw.competitions.eval_fingerprint를 갱신해야 재개된다.
    """


class BaselineSourceMismatchError(RuntimeError):
    """MinIO best_pipeline.py가 raw.pipelines 유효행이 가리키는 병합본과 다르다.

    격리(invalid_reason)나 remeasure로 레지스트리 유효집합이 바뀌었는데 MinIO를
    재구성하지 않으면 promote 게이트(_prev_best, 레지스트리)와 confirm 게이트
    (download_best_pipeline, MinIO blob)가 서로 다른 baseline을 보게 된다(#278,
    ADR-042). `bin.rebuild_best_pipeline`로 blob을 유효행으로 재생성해야 재개된다.
    """


# train_fingerprint 불일치로 걸린 pause 사유의 접두어. establish_baseline --remeasure와
# 아래 가드 재일치 분기 양쪽이 이걸로 다른 사유(cv_lb_divergence 등)와 구분해 해제한다.
_FP_PAUSE_PREFIX = "train_fingerprint 불일치"
_EVAL_FP_PAUSE_PREFIX = "eval_fingerprint 불일치"


def _fingerprint_guard(
    conn: PgConn, competition_id: str, *, column: str, fp: str, pause_prefix: str, issue: str,
    error_cls: type[RuntimeError], hint: str,
) -> None:
    """raw.competitions.<column>의 저장 지문과 현재 지문 fp가 어긋나면 승격 게이트를 멈춘다.
    최초 관측(저장값 NULL)이면 현재 지문을 심고 통과한다. 일치하면 이전에 심긴 같은 접두어의 pause를 푼다.
    column은 SQL에 그대로 삽입되므로 호출부의 상수 컬럼명만 넘긴다.
    """
    row = conn.execute(
        f"select {column} from raw.competitions where competition_id = %s",
        [competition_id],
    ).fetchone()
    stored = row[0] if row else None
    if stored is None:
        conn.execute(
            f"update raw.competitions set {column} = %s where competition_id = %s",
            [fp, competition_id],
        )
        return
    if stored == fp:
        # 설정을 원복해 지문이 다시 일치하면(v1.6.14 s4e11 실사례) 이전에 심긴 fp pause는
        # 조건이 이미 해소된 것이라 여기서 해제한다 — remeasure 없이도 재개돼야 한다.
        conn.execute(
            "update raw.competitions set auto_submit_paused_reason = null"
            " where competition_id = %s and auto_submit_paused_reason like %s",
            [competition_id, pause_prefix + "%"],
        )
        return
    reason = f"{pause_prefix} ({issue}): 저장 {stored[:12]} != 현재 {fp[:12]}. {hint}"
    conn.execute(
        "update raw.competitions"
        " set auto_submit_paused_reason = coalesce(auto_submit_paused_reason, %s)"
        " where competition_id = %s",
        [reason, competition_id],
    )
    _LOG.error("%s", reason)
    raise error_cls(reason)


def _train_fingerprint_guard(conn: PgConn, competition_id: str, train90: pl.DataFrame) -> None:
    """train90의 지문이 raw.competitions.train_fingerprint와 어긋나면 승격 게이트를 멈춘다.

    호출부는 train90(split_audit_holdout 결과)을 넘겨야 한다 — remeasure도 같은
    분할을 거치므로 그때만 지문 비교가 성립한다. holdout이 분리되지 않은
    경로(CycleConfig.holdout=None)는 호출하지 않는다.
    """
    cmd = f"uv run python -m bin.establish_baseline --remeasure --competition {competition_id}"
    _fingerprint_guard(
        conn, competition_id, column="train_fingerprint", fp=train_data_fingerprint(train90),
        pause_prefix=_FP_PAUSE_PREFIX, issue="#258", error_cls=TrainFingerprintMismatchError,
        hint=f"load_train 설정이 바뀌었으면 `{cmd}` 실행 후 재개.",
    )


def _eval_fingerprint_guard(conn: PgConn, competition_id: str, n_splits: int, seed: int) -> None:
    """평가 노브 지문이 raw.competitions.eval_fingerprint와 어긋나면 승격 게이트를 멈춘다(자동 remeasure 없음).

    점수 회귀를 조용히 수용하지 않고 사람이 remeasure 여부를 판단하게 하는 게 요점이다.
    """
    cmd = f"uv run python -m bin.establish_baseline --remeasure --competition {competition_id}"
    _fingerprint_guard(
        conn, competition_id, column="eval_fingerprint", fp=eval_semantics_fingerprint(n_splits, seed),
        pause_prefix=_EVAL_FP_PAUSE_PREFIX, issue="#348", error_cls=EvalFingerprintMismatchError,
        hint=f"평가 노브(후보 캡/fold 수/조기중단/seed)가 바뀌었으면 `{cmd}` 실행 후 재개.",
    )


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
    return row[0] if row and row[0] else None


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
            "params": params,
            "cv_score": cv_score,
            "improved": improved,
        }
        for model_type, member_index, params, cv_score, improved in rows
    ]
    if not any(e["improved"] for e in entries):
        return None
    return {"entries": entries}


def _prev_best_fold_scores(conn: PgConn, competition_id: str) -> list[float] | None:
    """확정 파이프라인(raw.pipelines)의 fold_scores.

    paired per-fold 유의성 검정(is_significant_gain)의 baseline으로 쓰인다.
    같은 seed로 생성된 fold split은 결정적이라 candidate의 fold_scores와 인덱스별로
    바로 대응시킬 수 있다.

    raw.pipelines.fold_scores가 있으면 그걸 우선한다 — bin/establish_baseline.py
    --remeasure(#262)가 cv_score 재측정 시 같이 갱신하는 값이라 최신 스케일이다.
    없으면(재측정 이력 없음) 연결된 attempt의 fold_scores로 폴백 — forward 경로는
    데이터/스케일이 안 바뀌었으므로 attempt 값 그대로가 맞다.

    확정 파이프라인이 없으면 None — 재측정 없는 attempt 최고값의 fold_scores로
    폴백하지 않는다(decisions.md ADR-025).
    """
    row = conn.execute(
        """
        select coalesce(p.fold_scores, a.fold_scores)
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
    return row[0] if row and row[0] else None


_FOLD1_CACHE_LIMIT = 200  # 최근 N개 attempt만 본다 — 재생산 몰림은 대부분 최근 이력에서 잡히고,
# 대회가 오래될수록 커지는 쿼리/payload 비용을 이 상한으로 막는다.


def _recent_fold1_cache(conn: PgConn, competition_id: str, n_splits: int) -> list[list[float]] | None:
    """최근 attempt의 fold_scores 중 이번 대회의 fold 구조(n_splits)와 맞는 것만, fold-1 값으로 중복
    제거해 돌려준다(#376) — fold-1이 일치하면 나머지 fold를 재계산하지 않고 그대로 재사용한다.
    #339(확정 base와의 tie)와 달리 base가 아닌 임의의 과거 attempt와도 매칭해, 서로 다른 patch가
    같은 params/로직으로 수렴하는 반복 재생산의 fold 계산을 회수한다."""
    rows = conn.execute(
        """
        select fold_scores from raw.attempts
        where competition_id = %s and fold_scores is not null and jsonb_array_length(fold_scores) = %s
        order by run_ts desc limit %s
        """,
        [competition_id, n_splits, _FOLD1_CACHE_LIMIT],
    ).fetchall()
    seen_fold1: set[float] = set()
    cache: list[list[float]] = []
    for (scores,) in rows:
        if not scores or scores[0] in seen_fold1:
            continue
        seen_fold1.add(scores[0])
        cache.append(scores)
    return cache or None


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

    source = strip_code_header(_code_download(code_path) or "")
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
    record_confirm(conn, attempt_id, confirm)
    if not confirm.confirmed:
        reason = "holdout 악화" if confirm.holdout_regressed else "cross-seed 미재현"
        _LOG.info("bootstrap baseline 미확립 — %s (%s)", competition_id, reason)
        return False

    fp_dict = competition_fingerprint(conn, competition_id)

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
        # 트랜잭션 안에서 부른다 — 업로드 실패가 insert를 롤백해야 DB와 blob이 어긋나지 않는다.
        _best_pipeline_upload(competition_id, materialized, strict=True)
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


def _dynamic_eda_context(conn: PgConn, competition_id: str, prev_best_cv: float | None) -> str:
    """매 사이클 DB를 조회해 EDA 카드 하단에 붙일 동적 컨텍스트를 생성한다."""
    window = 10
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


def _recent_failure_summary(conn: PgConn, competition_id: str) -> str:
    window = 5
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
        f"{CODE_HEADER_SEP}\n"
    )
    return _code_upload(competition_id, filename, header + source)


def _load_best_pipeline(competition_id: str) -> str | None:
    """Materialized best pipeline source. None if not yet stored."""
    return _best_pipeline_download(competition_id)


def _retrieval_scores(lessons: list[dict]) -> list[float | None] | None:
    """lessons의 score 목록. failure-lesson 채널은 score 키가 없음(.get 필수)."""
    return [l.get("score") for l in lessons] or None


_FEATURE_ACTIONS = ("feature_engineering", "preprocessing")
_DTYPE_MISMATCH_RE = re.compile(
    r"conversion from `str` to|cannot compare string with numeric|could not convert string to float"
)


def _resource_kill_feedback(error_trace: str, cpu_budget_sec: float, action_type: str = "") -> str:
    """워치독이 리소스 상한 초과로 강제종료한 에러는 원문(rc=-9 등) 대신 실행
    가능한 지시로 바꿔 재생성 피드백에 넘긴다. 원문은 "이유 모르게 죽었다"로만
    읽혀 재시도가 비슷하게 비싼 코드를 다시 쓰는 낭비를 낳았다(2026-08 실측:
    CPU kill attempt 113건이 예외 없이 이 경로로 재시도했고 전부 같은 자리에서
    다시 실패)."""
    if error_trace.startswith("cpu budget exceeded") and action_type == "ensemble":
        # 멤버의 트리 수를 줄이라고 하면 코더가 멤버를 base보다 약하게 만들어 블렌드가 나빠진다(#362, ADR-057).
        return (
            f"이 앙상블은 CPU 예산 {cpu_budget_sec:.0f}초를 초과해 종료됐다(코드 버그 아님). 멤버의 n_estimators/"
            "iterations/max_depth는 줄이지 마라 — 약한 멤버는 블렌드를 더 나쁘게 만든다. 가장 비싼 멤버를 빼 멤버 수를"
            " 줄이거나, method가 stack이면 weighted_average로 바꿔라."
        )
    if error_trace.startswith("cpu budget exceeded") and action_type in _FEATURE_ACTIONS:
        # 이 패치는 모델·n_estimators·후보 수를 못 바꾼다 — 일반 문구는 바꿀 수 없는 값을 가리킨다.
        return (
            f"이 패치는 CPU 예산 {cpu_budget_sec:.0f}초를 초과해 종료됐다(코드 버그 아님). 현재 best는 큰 LightGBM이라 fold 비용이 "
            "컬럼 수에 비례한다. 새로 만드는 컬럼을 8개 안팎으로 줄이고, 연속값 컬럼·기존 컬럼 재인코딩·컬럼 다수의 OOF target "
            "encoding·행 단위 map_elements를 빼라."
        )
    if "during preselect" in error_trace:  # runtime/isolate.py:_watch의 preselect 투영 kill 메시지와 맞물린다
        return (
            "param_candidates의 후보는 전부 CV 전에 80% train으로 한 번씩 재학습된다(preselect). 이 후보 목록은 그 단계만으로 "
            f"CPU 예산 {cpu_budget_sec:.0f}초의 절반을 넘겨 종료됐다(코드 버그 아님). 후보를 3~4개로 줄이고 "
            "n_estimators가 크거나 learning_rate가 작은 후보를 빼라."
        )
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
    if _DTYPE_MISMATCH_RE.search(error_trace):
        return (
            "컬럼 dtype이 가정과 다르다. 이 hook은 현재 best pipeline의 preprocess가 문자열 컬럼을 이미 Int32 ordinal 코드로 바꾼 "
            "뒤에 실행된다 — 그 컬럼에 문자열 키 replace_strict나 문자열 리터럴 비교를 쓰지 말고 train.schema로 dtype을 확인한 뒤 "
            "숫자 코드로 다뤄라. override=[\"preprocess\"]를 선언했다면 문자열 컬럼 인코딩을 직접 해야 한다.\n" + error_trace
        )
    return error_trace


def _noop_tie_feedback(action_type: str) -> str:
    """fold-1 조기 중단(#339)으로 확인된 no-op tie를 재생성 피드백으로 알린다.

    폴드 분할이 결정적이라(_make_folds) fold-1 점수가 confirmed baseline과
    비트 단위로 같다는 것 자체가 patch가 유효 계산을 못 바꿨다는 확정 신호다 —
    나머지 4개 fold를 마저 돌려봐야 결과가 달라지지 않는다."""
    return (
        f"이 {action_type} patch는 fold-1 검증 점수가 confirmed baseline과 완전히"
        " 동일했다(유효 계산 변화 없음 — params가 무시됐거나 기존 로직을 그대로"
        " 재발명했을 가능성). 다른 접근으로 다시 써라."
    )



_MAX_CODE_RETRIES = 2
_NO_EVAL = IsolatedResult(
    cv_score=None, cv_fold_var=None, fold_scores=None, label=None, gain_vs_best=None, error_trace=None,
)


@dataclass(frozen=True, slots=True)
class _EvalOutcome:
    source: str
    retries: int
    error_trace: str | None
    result: IsolatedResult
    peak_rss_bytes: int | None
    peak_cpu_sec: float | None


def _generate_until_valid(
    gen_kwargs: dict, action_type: str, feedback: str | None = None,
) -> tuple[str, int, list[str]]:
    """코드를 생성하고 정적 검사를 통과할 때까지 위반 목록을 피드백으로 다시 생성한다(최대 _MAX_CODE_RETRIES + 1회 생성).
    (소스, 생성 횟수, 마지막 정적 위반 목록 — 통과했으면 빈 리스트)."""
    source, errors, generated = "", [], 0
    for attempt in range(_MAX_CODE_RETRIES + 1):
        source = generate_code(**gen_kwargs, **({} if feedback is None else {"error_feedback": feedback}))
        generated += 1
        errors = validate_patch(source, action_type)
        if not errors:
            break
        feedback = "\n".join(errors)
        if attempt < _MAX_CODE_RETRIES:
            _LOG.info("static error (%d violation(s)) → regenerating (retry %d)", len(errors), attempt + 1)
    return source, generated, errors


def _evaluate_attempt(
    conn: PgConn, config: CycleConfig, source: str, retries: int, prev_best_cv: float | None, prev_code: str | None,
    action_type: str, gen_kwargs: dict, prev_best_fold_scores: list[float] | None,
) -> _EvalOutcome:
    """격리 평가를 최대 2회차까지 돌린다. 실패하면 실행 가능한 피드백으로 코드를 재생성해 재시도하고,
    fold-1 no-op tie(#339)는 남은 예산으로 다른 후보를 1회 더 시도한다."""
    _LOG.info("evaluating (n_splits=%d metric=%s)...", config.n_splits, config.metric)
    t_eval = time.monotonic()
    # #376 — attempt당 1회만 조회하고 두 eval 회차(재시도 포함)가 같은 캐시를 공유한다.
    known_fold1_scores = _recent_fold1_cache(conn, config.competition_id, config.n_splits)
    # CPU 예산은 eval 회차가 아니라 attempt 전체 기준으로 집행한다 — 회차마다 독립적으로 주면 rc=-9 피드백으로 재생성한
    # 2회차가 같은 자리에서 또 예산을 태워 attempt 하나가 예산의 2배까지 갔다(2026-08 실측). 1회차가 다 쓰면 2회차는 돌리지 않는다.
    cpu_budget_total = (
        config.cpu_budget_secs
        if config.cpu_budget_secs is not None
        else float(os.environ.get("EVAL_CPU_BUDGET_SECS", str(DEFAULT_CPU_BUDGET_SECS)))
    )
    cpu_spent = 0.0
    result = _NO_EVAL
    error_trace: str | None = None
    peak_rss_bytes: int | None = None
    peak_cpu_sec: float | None = None
    for eval_i in range(2):
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
            prev_best_fold_scores=prev_best_fold_scores,
            known_fold1_scores=known_fold1_scores,
        )
        peak_rss_bytes = iso.peak_rss_bytes
        peak_cpu_sec = iso.peak_cpu_sec
        cpu_spent += iso.peak_cpu_sec or 0.0
        if not iso.error_trace:
            result = iso
            gain_str = f"{iso.gain_vs_best:+.6f}" if iso.gain_vs_best is not None else "N/A"
            _LOG.info(
                "eval ok in %.1fs cv=%.6f fold_var=%.6f gain=%s label=%s",
                time.monotonic() - t_eval, iso.cv_score, iso.cv_fold_var or 0.0, gain_str, iso.label or "regression",
            )
            if iso.is_noop_tie:
                _LOG.warning(
                    "no-op tie: cv_score matches prev_best within float noise "
                    "(action=%s)%s — patch made no effective change",
                    action_type, " [fold-1 조기 중단]" if iso.noop_early_exit else "",
                )
            if iso.noop_early_exit and eval_i == 0 and cpu_budget_total - cpu_spent > 0:
                # tie 결과는 이미 유효하니 재시도가 실패해도 잃을 게 없다. 재생성이 정적검사를 못 넘겨도 그 tie 결과를 그대로 채택한다.
                _LOG.info("noop tie (fold-1 조기 중단) → 다른 후보로 재시도")
                source, generated, static_errs = _generate_until_valid(
                    gen_kwargs, action_type, _noop_tie_feedback(action_type),
                )
                retries += generated
                if static_errs:
                    break
                continue
            break
        _LOG.warning("eval error (try %d) → regenerating: %s", eval_i + 1, (iso.error_trace or "")[:120])
        if eval_i == 0 and cpu_budget_total - cpu_spent <= 0:
            # 남은 예산 0으로 재시도해봐야 즉시 다시 죽으므로 LLM 호출과 2회차 eval을 통째로 아낀다.
            _LOG.warning(
                "cpu budget exhausted after try 1 (spent %.0fs of %.0fs) — skipping retry", cpu_spent, cpu_budget_total,
            )
            error_trace = iso.error_trace
            break
        if eval_i == 0:
            # 재생성 정적검사도 최초 codegen과 같은 재시도 폭을 준다(#273). 전부 실패해도 원래 kill 사유를 error_trace에
            # 남겨 정적 가드 메시지가 진짜 원인을 가리지 않게 한다.
            source, generated, static_errs = _generate_until_valid(
                gen_kwargs, action_type, _resource_kill_feedback(iso.error_trace, cpu_remaining, action_type),
            )
            retries += generated
            if static_errs:
                error_trace = (
                    f"{iso.error_trace}\n(regeneration also failed static "
                    f"validation after {_MAX_CODE_RETRIES + 1} tries: {'; '.join(static_errs)})"
                )
                break
        else:
            error_trace = iso.error_trace
            # 1회차 tie 결과는 지금 저장될 (에러난) 재생성 코드의 것이 아니다(#405).
            result = _NO_EVAL
    return _EvalOutcome(source, retries, error_trace, result, peak_rss_bytes, peak_cpu_sec)


def _judge_attempt(
    conn: PgConn, config: CycleConfig, res: IsolatedResult, error_trace: str | None,
    prev_best_fold_scores: list[float] | None,
) -> tuple[str, str | None]:
    """(최종 label, error_trace). 리더보드 세계 1위를 넘는 cv는 격리한다(#288).

    jump 판정은 promotion 게이트(is_significant_gain, paired per-fold t-test)와 같은 기준으로 통일한다 — harness의
    절대-마진 jump는 수렴한 대회에서 사실상 도달 불가해 실제 승격 attempt도 전부 neutral로 남았고,
    bandit/stagnation/reflection이 "성공 신호 0"으로 굳어 있었다."""
    if not error_trace and res.cv_score is not None:
        ceiling_reason = leaderboard_ceiling_violation(
            conn, config.competition_id, res.cv_score, fold_scores=res.fold_scores,
        )
        if ceiling_reason is not None:
            error_trace = ceiling_reason
            _LOG.warning("%s — attempt 격리(promotion 이전 단계, #288)", ceiling_reason)
    if error_trace:
        _LOG.warning("failed — %s", error_trace[:200])
        return "error", error_trace
    _, metric_sign, _ = get_metric(config.metric)
    if is_significant_gain(
        res.gain_vs_best, res.cv_fold_var or 0.0,
        candidate_fold_scores=res.fold_scores,
        baseline_fold_scores=prev_best_fold_scores,
        metric_sign=metric_sign,
    ):
        return "jump", None
    label = res.label or "regression"
    return ("neutral" if label == "jump" else label), None  # harness 절대-마진 jump는 paired 유의성 미달이면 강등


def _attempt_row(
    attempt_id: str, config: CycleConfig, decision: StrategyDecision, action_type: str, lessons: list[dict],
    ev: _EvalOutcome, label: str, error_trace: str | None, code_path: str, duration_sec: float,
) -> dict:
    res = ev.result
    return {
        "attempt_id":       attempt_id,
        "competition_id":   config.competition_id,
        "run_ts":           datetime.now(timezone.utc),
        "stage":            config.stage,
        "hypothesis":       decision.hypothesis,
        "action_type":      action_type,
        "reflection_ids":   decision.reflection_ids or None,
        "retrieval_scores": _retrieval_scores(lessons),
        "retrieved_ids":    [l["reflection_id"] for l in lessons] or None,
        "cv_score":         res.cv_score,
        # noop_early_exit(fold-1 tie 조기 중단)일 때 harness는 None을 의도한다(1-fold만 계산해 분산을 못 만듦) — 0.0으로
        # 대체하면 "분산이 실제로 0"과 구분이 안 되므로 None 그대로 저장한다(#356).
        "cv_fold_var":      res.cv_fold_var,
        "label":            label,
        "gain_vs_best":     res.gain_vs_best,
        "gain_vs_best_relative": res.gain_vs_best_relative,
        "error_trace":      error_trace,
        "error_signature":  normalize_error(error_trace) if error_trace else None,
        "duration_sec":     round(duration_sec, 1),
        "peak_rss_bytes":   ev.peak_rss_bytes,
        "peak_cpu_sec":     ev.peak_cpu_sec,
        "code_path":        str(code_path),
        "retries":          ev.retries,
        "fold_scores":      json.dumps(res.fold_scores) if res.fold_scores is not None else None,
        "params":           json.dumps(res.selected_params) if res.selected_params else None,
        "model_type":       res.model_type,
        "noop_early_exit":  res.noop_early_exit,
    }


def _strategize_attempt(
    conn: PgConn, config: CycleConfig, lessons: list[dict], prev_best_cv: float | None, stagnation: StagnationSignal,
    forced_action: str | None,
) -> StrategyDecision:
    enriched_eda = config.eda_card + _dynamic_eda_context(conn, config.competition_id, prev_best_cv)
    action_prior = get_action_prior(conn, config.competition_id)
    t_strategize = time.monotonic()
    decision = strategize(
        eda_card=enriched_eda,
        lessons=lessons,
        stage=config.stage,
        prev_best_cv=prev_best_cv,
        stagnation=stagnation,
        action_prior=action_prior,
        forced_action_type=forced_action,
    )
    _LOG.info("strategize done in %.1fs", time.monotonic() - t_strategize)
    return decision


def _generate_attempt_source(
    conn: PgConn, config: CycleConfig, decision: StrategyDecision, prev_code: str | None, action_type: str,
) -> tuple[dict, str, int, list[str]]:
    """(gen_kwargs, 소스, 생성 횟수, 마지막 정적 위반 목록)."""
    t_codegen = time.monotonic()
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
    source, generated, static_errs = _generate_until_valid(gen_kwargs, action_type)
    _LOG.info("codegen done in %.1fs retries=%d", time.monotonic() - t_codegen, generated - 1)
    return gen_kwargs, source, generated, static_errs


def run_attempt_core(
    conn: PgConn,
    config: CycleConfig,
    lessons: list[dict],
    prev_best_cv: float | None,
    super_cycle_id: str | None = None,
    attempt_index: int | None = None,
    forced_action: str | None = None,
) -> _AttemptData:
    """Strategize → Generate → Evaluate → Persist one attempt. Returns data needed for reflect."""
    if config.holdout is not None:
        _train_fingerprint_guard(conn, config.competition_id, config.train)
        _eval_fingerprint_guard(conn, config.competition_id, config.n_splits, config.seed)
        _baseline_source_guard(conn, config.competition_id)
    attempt_id = str(uuid.uuid4())
    attempt_start = time.monotonic()

    stagnation = detect_stagnation(conn, config.competition_id)
    _LOG.info(
        "start attempt_id=%s super_cycle=%s idx=%s stage=%s prev_best=%s n_lessons=%d stagnant=%s",
        attempt_id[:8],
        super_cycle_id[:8] if super_cycle_id else "-",
        attempt_index if attempt_index is not None else "-",
        config.stage, prev_best_cv, len(lessons),
        stagnation.is_stagnant if stagnation else False,
    )
    decision = _strategize_attempt(conn, config, lessons, prev_best_cv, stagnation, forced_action)
    prev_code = _load_best_pipeline(config.competition_id)
    action_type = "bootstrap" if (config.stage == "bootstrap" and not prev_code) else decision.action_type

    gen_kwargs, source, generated, static_errs = _generate_attempt_source(conn, config, decision, prev_code, action_type)

    # is_significant_gain(아래)과 eval_isolated의 fold-1 조기 중단(#339)이 같은 baseline fold_scores를 쓰므로 attempt당 1회만 조회한다.
    prev_best_fold_scores = _prev_best_fold_scores(conn, config.competition_id)
    if static_errs:
        ev = _EvalOutcome(source, generated - 1, "\n".join(static_errs), _NO_EVAL, None, None)
    else:
        ev = _evaluate_attempt(
            conn, config, source, generated - 1, prev_best_cv, prev_code, action_type, gen_kwargs, prev_best_fold_scores,
        )
    res = ev.result
    label, error_trace = _judge_attempt(conn, config, res, ev.error_trace, prev_best_fold_scores)

    code_path = _save_code(
        ev.source,
        competition_id=config.slug or config.competition_id,
        attempt_id=attempt_id,
        stage=config.stage,
        hypothesis=decision.hypothesis,
        action_type=action_type,
        cv_score=res.cv_score,
        gain_vs_best=res.gain_vs_best,
        error_trace=error_trace,
    )

    duration_sec = time.monotonic() - attempt_start
    row = _attempt_row(attempt_id, config, decision, action_type, lessons, ev, label, error_trace, code_path, duration_sec)
    if super_cycle_id is not None:
        row["super_cycle_id"] = super_cycle_id
        # NULL이면 reflection_impact 뷰(IS NOT FALSE)가 승격된 attempt로 집계한다(#205) — promote가 확정할 때만 True로 바꾼다.
        row["was_promoted"] = False
    insert_attempt(conn, row)

    _LOG.info(
        "persist done — total %.1fs attempt_id=%s action=%s label=%s retries=%d",
        duration_sec, attempt_id[:8], action_type, label, ev.retries,
    )

    if config.stage == "reflexion":
        update_bandit(
            conn,
            competition_id=config.competition_id,
            action_type=action_type,
            label=label,
            gain_vs_best=res.gain_vs_best,
            error_trace=error_trace,
            is_noop_tie=res.is_noop_tie,
        )

    return _AttemptData(
        attempt_id=attempt_id,
        decision=decision,
        cv_score=res.cv_score,
        label=label,
        gain_vs_best=res.gain_vs_best,
        retries=ev.retries,
        is_noop_tie=res.is_noop_tie,
    )
