CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.competitions (
    competition_id  text PRIMARY KEY,
    name            text,
    task_type       text,
    metric          text,
    metric_sign     int,
    start_ts        timestamp,
    fingerprint     jsonb
);
-- cv↔LB 발산 트립와이어가 채운다. NULL이면 정상 — auto_submit이 이 값이 있으면
-- 해당 대회 제출을 건너뛴다. 자동 해제 없음, 사람이 직접 NULL로 되돌려야 재개된다
-- (docs/decisions.md ADR-026, docs/runbook.md §4-3).
ALTER TABLE raw.competitions ADD COLUMN IF NOT EXISTS auto_submit_paused_reason text;

CREATE TABLE IF NOT EXISTS raw.attempts (
    attempt_id       text PRIMARY KEY,
    competition_id   text,
    run_ts           timestamp,
    stage            text,
    hypothesis       text,
    action_type      text,
    model_type       text,
    params           jsonb,
    features         jsonb,
    cv_score         double precision,
    cv_fold_var      double precision,
    lb_score         double precision,
    label            text,
    gain_vs_best     double precision,
    error_trace      text,
    reflection_ids   text[],
    retrieval_scores double precision[],
    duration_sec     double precision,
    code_path        text,
    retries          int DEFAULT 0,
    super_cycle_id   text,
    was_promoted     boolean
);
ALTER TABLE raw.attempts ADD COLUMN IF NOT EXISTS super_cycle_id text;
ALTER TABLE raw.attempts ADD COLUMN IF NOT EXISTS was_promoted boolean;
ALTER TABLE raw.attempts ADD COLUMN IF NOT EXISTS holdout_score double precision;
ALTER TABLE raw.attempts ADD COLUMN IF NOT EXISTS confirm_seed_gains jsonb;

-- attempt별 fold_scores 영속화 — paired per-fold 유의성 검정이
-- 이전 confirmed best의 fold_scores를 참고하려면 승격 시점이 아니라 매 attempt마다
-- 기록돼 있어야 한다(raw.pipelines가 attempt_id로 join해 재사용).
ALTER TABLE raw.attempts ADD COLUMN IF NOT EXISTS fold_scores jsonb;

-- metric마다 gain_vs_best 스케일이 달라(rmse 원시 단위 vs auc 0~1)
-- reflection_impact 전역 z-score가 오염된다 — regression_error는 baseline_cv로 나눈
-- 상대값, 나머지는 gain_vs_best 그대로. reflection_impact가 이 컬럼만 집계한다.
ALTER TABLE raw.attempts ADD COLUMN IF NOT EXISTS gain_vs_best_relative double precision;

-- retrieved_ids: 검색된 전체 교훈(reflection_ids는 실제 채택분만).
-- forward-only(과거 attempt는 원본 검색 결과가 남아있지 않아 backfill 불가).
ALTER TABLE raw.attempts ADD COLUMN IF NOT EXISTS retrieved_ids text[];
-- error_signature: error_pitfalls._normalize_error(error_trace) 결과 영속화 — 매 조회마다
-- on-the-fly 정규화하던 것을 저장. bin/backfill_error_signatures.py로 기존 행 소급 채움 가능.
ALTER TABLE raw.attempts ADD COLUMN IF NOT EXISTS error_signature text;

-- eval_isolated의 RSS 워치독(runtime/isolate.py)이 폴링한 peak RSS.
-- 성공 attempt에도 채워져 EVAL_RSS_LIMIT_BYTES 기본값(4GiB)이 적정한지
-- 분포로 검증하는 데 쓴다 — forward-only, backfill 불가(과거 프로세스는 이미 종료).
ALTER TABLE raw.attempts ADD COLUMN IF NOT EXISTS peak_rss_bytes bigint;

-- eval_isolated의 CPU 워치독(runtime/isolate.py)이 폴링한 peak CPU 시간(초).
-- 성공 attempt에도 채워져 EVAL_CPU_BUDGET_SECS 기본값(900)이 적정한지 분포로
-- 검증하는 데 쓴다 — forward-only, backfill 불가(과거 프로세스는 이미 종료).
-- peak_rss_bytes와 동일한 계약(GH #159).
ALTER TABLE raw.attempts ADD COLUMN IF NOT EXISTS peak_cpu_sec double precision;

-- materialize 시 sha256 기록 — submit.py exec 전 MinIO 다운로드본과 대조해
-- 익명 write 버킷 변조를 탐지한다. code 컬럼(raw.attempts 원본)이 아니라
-- 실제 exec되는 materialized best_pipeline.py 내용의 해시.
ALTER TABLE raw.pipelines ADD COLUMN IF NOT EXISTS pipeline_sha256 text;

-- merge-verify eval(승격 시 1회) 시점에 함께 뽑은 out-of-fold
-- 예측 — bin/blend.py가 파이프라인 밖에서 결정적 blend(Ridge)를 학습할 때 사용.
ALTER TABLE raw.pipelines ADD COLUMN IF NOT EXISTS oof_preds jsonb;

-- 승격 시점의 병합본(materialized best_pipeline.py) 전문 스냅샷 — Postgres 신뢰 사본.
-- replay(patch 히스토리를 현재 materialize 로직으로 재병합)는 materialize 로직이
-- 바뀌면 당시 병합본과 달라져 평가 시점 base를 재현할 수 없다. submit.py attempt
-- 경로는 이 스냅샷을 base로 쓴다.
ALTER TABLE raw.pipelines ADD COLUMN IF NOT EXISTS materialized_code text;

-- 승격 후 사후 발견된 결함(preprocess valid-target 누수 등) 표기 — NULL이면 유효.
-- 삭제하지 않고 조회 경로(_prev_best, blend 후보, replay 등)에서만 제외해 이력을
-- 보존한다. bin/quarantine_leaks.py가 스캔해서 채운다(docs/decisions.md ADR-025).
ALTER TABLE raw.pipelines ADD COLUMN IF NOT EXISTS invalid_reason text;

CREATE TABLE IF NOT EXISTS raw.submission_budget (
    competition_id  text,
    day             date,
    count           int,
    PRIMARY KEY (competition_id, day)
);

CREATE TABLE IF NOT EXISTS raw.reflections (
    reflection_id   text PRIMARY KEY,
    created_at      timestamp,
    attempt_id      text,
    competition_id  text,
    embedded_text   text,
    embedding       vector(1024),
    full_lesson     text,
    generality      text,
    label           text,
    reflector_label text,
    lesson_type     text,
    gain_vs_best    double precision,
    archived        boolean DEFAULT false
);
ALTER TABLE raw.reflections ADD COLUMN IF NOT EXISTS lesson_type text;

CREATE TABLE IF NOT EXISTS raw.pipelines (
    pipeline_id          text PRIMARY KEY,
    attempt_id           text,
    competition_id       text,
    fingerprint_snapshot jsonb,
    code                 text,
    cv_score             double precision,
    gain_vs_best         double precision
);

CREATE TABLE IF NOT EXISTS raw.cycle_queue (
    queue_id     text PRIMARY KEY,
    competition  text NOT NULL,
    stage        text NOT NULL,
    n_cycles     int NOT NULL,
    priority     int DEFAULT 0,
    status       text NOT NULL DEFAULT 'pending',
    created_at   timestamp DEFAULT now(),
    started_at   timestamp,
    ended_at     timestamp,
    cycles_done  int DEFAULT 0,
    latest_score double precision,
    error        text
);
-- bin/run_daemon.py의 리스 기반 라운드로빈(#133) — 한 큐 항목이 daemon을 통째로
-- 붙잡지 않도록 DAEMON_CYCLES_PER_LEASE 사이클마다 pending으로 되돌린다.
-- _pop_pending이 이 컬럼 기준 오름차순으로 다음 항목을 골라, 방금 리스를 마친
-- 항목이 자연히 뒤로 밀리게 한다. NULL이면(리스 이력 없음) created_at으로 폴백.
ALTER TABLE raw.cycle_queue ADD COLUMN IF NOT EXISTS last_leased_at timestamp;

CREATE OR REPLACE VIEW score_progression AS
SELECT
    a.attempt_id,
    a.competition_id,
    row_number() OVER (
        PARTITION BY a.competition_id ORDER BY a.run_ts
    ) AS attempt_no,
    a.run_ts,
    a.stage,
    a.action_type,
    a.cv_score,
    a.lb_score,
    a.label,
    a.gain_vs_best,
    max(c.metric_sign * a.cv_score) OVER (
        PARTITION BY a.competition_id
        ORDER BY a.run_ts
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) * c.metric_sign AS best_so_far
FROM raw.attempts a
JOIN raw.competitions c USING (competition_id);

-- CASCADE 필요: lesson_dead가 reflection_impact를 참조해서 CASCADE 없이는 두 번째
-- apply_schema부터 "cannot drop view ... other objects depend on it"로 실패한다.
-- lesson_dead는 이 파일 뒤쪽에서 CREATE OR REPLACE로 다시 만들어지므로 안전하다.
DROP VIEW IF EXISTS reflection_impact CASCADE;
DROP VIEW IF EXISTS stg_attempts_reflexion_only;
DROP VIEW IF EXISTS stg_attempts;

CREATE VIEW stg_attempts AS
SELECT
    a.*,
    c.metric_sign
FROM raw.attempts a
JOIN raw.competitions c USING (competition_id);

CREATE VIEW stg_attempts_reflexion_only AS
SELECT * FROM stg_attempts
WHERE stage = 'reflexion';

CREATE VIEW reflection_impact AS
WITH scored AS (
    SELECT
        competition_id,
        reflection_ids,
        gain_vs_best_relative AS gain_vs_best   -- baseline 재사용 + metric별 스케일
                       -- 상대화(regression_error는 baseline_cv 비율, 나머지는 gain_vs_best와
                       -- 동일). NULL인 legacy row는 아래서 제외.
    FROM stg_attempts_reflexion_only
    WHERE was_promoted IS NOT FALSE  -- NULL=legacy (promoted), TRUE=winner, FALSE=super-cycle loser excluded
      AND gain_vs_best_relative IS NOT NULL
),
per_reflection AS (
    SELECT
        unnest(reflection_ids) AS reflection_id,
        gain_vs_best / array_length(reflection_ids, 1) AS gain_vs_best
    FROM scored
    WHERE reflection_ids IS NOT NULL
)
SELECT
    reflection_id,
    count(*)                                              AS times_applied,
    round(avg(gain_vs_best)::numeric, 5)                  AS avg_gain,
    sum(CASE WHEN gain_vs_best > 0 THEN 1 ELSE 0 END)    AS jumps,
    round(max(gain_vs_best)::numeric, 5)                  AS best_jump
FROM per_reflection
GROUP BY reflection_id
ORDER BY avg_gain DESC;

-- super-cycle 공유 retrieve 컨텍스트 (retrieve task → attempt/promote tasks)
-- PK는 queue_id가 아닌 run_id(Airflow dag_run_id) — queue_id는 같은 super-cycle의
-- 여러 cycle이 공유해서(max_active_runs=4) 동시 실행 시 서로의 context row를
-- 덮어쓰거나 훔쳐 지우는 레이스가 있었다. run_id는 cycle마다 유일.
CREATE TABLE IF NOT EXISTS raw.super_cycle_context (
    run_id            text PRIMARY KEY,
    queue_id          text NOT NULL,
    super_cycle_id    text NOT NULL,
    competition_id    text NOT NULL,
    prev_best_cv      double precision,
    lessons           jsonb NOT NULL,
    assigned_actions  jsonb,
    created_at        timestamp DEFAULT now()
);
ALTER TABLE raw.super_cycle_context ADD COLUMN IF NOT EXISTS assigned_actions jsonb;
ALTER TABLE raw.super_cycle_context ADD COLUMN IF NOT EXISTS run_id text;

-- action_type별 Beta-Bernoulli 밴딧 (advise용, stagnation 승격)
CREATE TABLE IF NOT EXISTS raw.action_bandit (
    scope       text NOT NULL,
    scope_key   text NOT NULL,
    action_type text NOT NULL,
    alpha       double precision NOT NULL DEFAULT 1.0,
    beta        double precision NOT NULL DEFAULT 1.0,
    updated_at  timestamp DEFAULT now(),
    PRIMARY KEY (scope, scope_key, action_type)
);

CREATE OR REPLACE VIEW action_bandit_posterior AS
SELECT
    scope,
    scope_key,
    action_type,
    alpha / (alpha + beta)                    AS posterior_mean,
    alpha + beta                              AS trials,
    alpha,
    beta,
    updated_at
FROM raw.action_bandit
ORDER BY scope, scope_key, posterior_mean DESC;

CREATE OR REPLACE VIEW cold_start_progression AS
SELECT
    a.competition_id,
    row_number() OVER (PARTITION BY a.competition_id ORDER BY a.run_ts) AS attempt_no,
    a.run_ts,
    a.stage,
    a.cv_score,
    max(c.metric_sign * a.cv_score) OVER (
        PARTITION BY a.competition_id ORDER BY a.run_ts
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) * c.metric_sign AS best_so_far
FROM raw.attempts a
JOIN raw.competitions c USING (competition_id)
WHERE a.cv_score IS NOT NULL;

-- 고정 seed=42 fold에 수백 attempt가 최적화되면서 교훈·밴딧·전략 신호가
-- seed-42 노이즈를 학습할 위험을 감시하는 추세 뷰. overfit_gap 양수 = holdout이
-- CV보다 나쁨(metric_sign으로 방향 통일) = drift 의심.
CREATE OR REPLACE VIEW holdout_cv_gap_trend AS
SELECT
    a.attempt_id,
    a.competition_id,
    a.run_ts,
    a.cv_score,
    a.holdout_score,
    c.metric_sign * (a.cv_score - a.holdout_score) AS overfit_gap
FROM raw.attempts a
JOIN raw.competitions c USING (competition_id)
WHERE a.holdout_score IS NOT NULL;

-- 교훈 파이프라인 어디서 끊기나(작성→검색→인용→양의gain).
-- retrieved/retrieve_rate는 P1(retrieved_ids) 반영 이후 데이터만 채워진다 —
-- retrieved_precise=false면 과거 데이터가 섞여 retrieve_rate를 신뢰하지 말 것.
CREATE OR REPLACE VIEW lesson_funnel AS
WITH written AS (
    SELECT competition_id, count(*) AS written
    FROM raw.reflections
    WHERE archived = false
    GROUP BY competition_id
),
attempt_stats AS (
    SELECT
        competition_id,
        count(*)                                                                    AS total_attempts,
        count(*) FILTER (WHERE retrieved_ids IS NOT NULL
                            AND array_length(retrieved_ids, 1) > 0)                  AS retrieved,
        count(*) FILTER (WHERE reflection_ids IS NOT NULL
                            AND array_length(reflection_ids, 1) > 0)                 AS cited,
        count(*) FILTER (WHERE reflection_ids IS NOT NULL
                            AND array_length(reflection_ids, 1) > 0
                            AND gain_vs_best_relative > 0)                           AS positive_gain,
        bool_or(retrieved_ids IS NOT NULL)                                          AS retrieved_precise
    FROM raw.attempts
    WHERE stage = 'reflexion'
    GROUP BY competition_id
)
SELECT
    coalesce(w.competition_id, s.competition_id)                  AS competition_id,
    coalesce(w.written, 0)                                        AS written,
    coalesce(s.total_attempts, 0)                                 AS total_attempts,
    coalesce(s.retrieved, 0)                                      AS retrieved,
    coalesce(s.cited, 0)                                          AS cited,
    coalesce(s.positive_gain, 0)                                  AS positive_gain,
    round(s.retrieved::numeric / nullif(s.total_attempts, 0), 4)  AS retrieve_rate,
    round(s.cited::numeric / nullif(s.total_attempts, 0), 4)      AS cite_rate,
    round(s.positive_gain::numeric / nullif(s.cited, 0), 4)       AS gain_rate,
    coalesce(s.retrieved_precise, false)                          AS retrieved_precise
FROM written w
FULL OUTER JOIN attempt_stats s USING (competition_id);

-- 죽은 교훈 — 검색 풀 오염원. archive_lessons.py는 times_applied>=3인 것만 정리해
-- 인용 0회(never_cited) 교훈은 영원히 남는다.
CREATE OR REPLACE VIEW lesson_dead AS
SELECT
    r.reflection_id,
    r.competition_id,
    r.lesson_type,
    r.generality,
    r.created_at,
    coalesce(i.times_applied, 0) AS times_cited,
    i.avg_gain,
    CASE
        WHEN coalesce(i.times_applied, 0) = 0 THEN 'never_cited'
        WHEN i.avg_gain <= 0 THEN 'applied_negative'
        ELSE null
    END AS reason
FROM raw.reflections r
LEFT JOIN reflection_impact i USING (reflection_id)
WHERE r.archived = false
  AND (coalesce(i.times_applied, 0) = 0 OR i.avg_gain <= 0);

-- near-duplicate 교훈(Reflector 패러프레이즈 남발 감시). 임계는 넉넉히(0.90) 저장해
-- 두고 api.py가 ?threshold= 로 더 좁힌다. embedding에 ANN 인덱스가 없어 자기조인
-- 비용이 대회당 반영 건수의 제곱에 비례한다 — s4e1처럼 reflection이 수천 건
-- 쌓인 대회에서 12초 이상 걸려 대시보드 로드를 막던 실측 문제(2026-08-03).
-- 최근 것일수록 중복 여부가 운영에 의미 있으므로 대회당 최근 250건으로 자기조인
-- 범위를 캡핑한다(s4e1 실측: cap 없음 12.2s → 500건 1.7s → 250건 0.3s) —
-- 오래된 pair는 이미 archive_lessons 등으로 정리됐을 가능성이 높음.
CREATE OR REPLACE VIEW lesson_duplicates AS
WITH recent AS (
    SELECT reflection_id, competition_id, embedding
    FROM (
        SELECT reflection_id, competition_id, embedding,
               row_number() OVER (PARTITION BY competition_id ORDER BY created_at DESC) AS rn
        FROM raw.reflections
        WHERE archived = false
    ) ranked
    WHERE rn <= 250
)
SELECT
    a.competition_id,
    a.reflection_id AS reflection_id_a,
    b.reflection_id AS reflection_id_b,
    1 - (a.embedding <=> b.embedding) AS cos_sim
FROM recent a
JOIN recent b
    ON a.reflection_id < b.reflection_id
   AND a.competition_id = b.competition_id
WHERE 1 - (a.embedding <=> b.embedding) > 0.90;

-- 밴딧 믿음(posterior_mean)과 실측 가중성공률 괴리. jump_rate가 아니라
-- update_bandit(action_optimizer.py)과 동일한 가중치로 실측해야 구조적 gap을 피한다.
CREATE OR REPLACE VIEW bandit_calibration AS
WITH realized AS (
    SELECT
        competition_id AS scope_key,
        action_type,
        sum(CASE
                WHEN label = 'jump' THEN 1.0
                WHEN gain_vs_best > 0 THEN 0.5
                WHEN label = 'regression' OR error_trace IS NOT NULL THEN 0.0
                ELSE 0.1
            END) AS num,
        sum(CASE
                WHEN label = 'jump' THEN 1.0
                WHEN gain_vs_best > 0 THEN 0.6
                WHEN label = 'regression' OR error_trace IS NOT NULL THEN 1.0
                ELSE 0.2
            END) AS den,
        count(*)    AS picks,
        max(run_ts) AS last_picked_ts
    FROM raw.attempts
    WHERE stage = 'reflexion' AND cv_score IS NOT NULL
    GROUP BY competition_id, action_type
)
SELECT
    b.scope,
    b.scope_key,
    b.action_type,
    b.alpha,
    b.beta,
    b.posterior_mean,
    b.alpha + b.beta - 2                                 AS net_evidence,
    r.picks,
    r.last_picked_ts,
    r.num / nullif(r.den, 0)                             AS weighted_success,
    abs(b.posterior_mean - r.num / nullif(r.den, 0))     AS calibration_gap
FROM action_bandit_posterior b
LEFT JOIN realized r ON r.scope_key = b.scope_key AND r.action_type = b.action_type;

-- 에러 시그니처별 재발(P2 error_signature 기반). pitfall_active는 top_error_pitfalls의
-- min_count=2와 동일 기준(2회째까지는 pitfall 미주입) — occurrences_after_active>0이면
-- 안티패턴 루프가 안 닫힌 것.
CREATE OR REPLACE VIEW error_recurrence AS
WITH sigs AS (
    SELECT
        competition_id,
        action_type,
        error_signature,
        run_ts,
        row_number() OVER (
            PARTITION BY competition_id, action_type, error_signature
            ORDER BY run_ts
        ) AS occurrence_no
    FROM raw.attempts
    WHERE label = 'error' AND error_signature IS NOT NULL
)
SELECT
    s.competition_id,
    s.action_type,
    s.error_signature,
    count(*)                                    AS total,
    min(s.run_ts)                               AS first_seen,
    max(s.run_ts)                               AS last_seen,
    bool_or(s.occurrence_no >= 2)                AS pitfall_active,
    count(*) FILTER (WHERE s.occurrence_no > 2)  AS occurrences_after_active,
    exists (
        SELECT 1 FROM raw.reflections r
        JOIN raw.attempts a2 USING (attempt_id)
        WHERE r.lesson_type = 'failure' AND r.archived = false
          AND a2.competition_id = s.competition_id AND a2.action_type = s.action_type
    ) AS has_avoid_lesson
FROM sigs s
GROUP BY s.competition_id, s.action_type, s.error_signature;

-- 대회 간 교훈 전이 매트릭스(X1) — source_comp != target_comp 행이 실제 전이.
CREATE OR REPLACE VIEW transfer_matrix AS
SELECT
    src.competition_id AS source_comp,
    a.competition_id   AS target_comp,
    count(*)            AS citations
FROM raw.attempts a
CROSS JOIN LATERAL unnest(a.reflection_ids) AS rid
JOIN raw.reflections src ON src.reflection_id = rid
GROUP BY src.competition_id, a.competition_id;

CREATE TABLE IF NOT EXISTS raw.kaggle_submissions (
    submission_id  text PRIMARY KEY,
    competition_id text NOT NULL,
    attempt_id     text,
    submitted_at   timestamp,
    message        text,
    csv_path       text,
    status         text DEFAULT 'queued',
    lb_score       double precision,
    error          text,
    checked_at     timestamp
);

-- cv↔LB 정합성 — 완료된 제출을 시간순으로 이어 delta_cv/delta_lb를 계산한다.
-- 둘 다 metric_sign을 곱해 "개선이면 양수"로 방향을 통일했다. diverged=true는
-- cv는 개선인데 LB는 악화된 제출. 실시간 차단은 이 뷰가 아니라
-- bin/api.py:refresh_submission_row의 트립와이어가 담당(승격 시점에 즉시
-- invalid_reason/auto_submit_paused_reason을 채움, docs/decisions.md ADR-026) —
-- 이 뷰는 그 판정의 사후 관측·검증용.
CREATE OR REPLACE VIEW cv_lb_calibration AS
WITH ordered AS (
    SELECT
        s.submission_id,
        s.competition_id,
        s.submitted_at,
        s.attempt_id,
        a.cv_score,
        s.lb_score,
        c.metric_sign,
        lag(a.cv_score) OVER (PARTITION BY s.competition_id ORDER BY s.submitted_at) AS prev_cv,
        lag(s.lb_score) OVER (PARTITION BY s.competition_id ORDER BY s.submitted_at) AS prev_lb
    FROM raw.kaggle_submissions s
    JOIN raw.competitions c USING (competition_id)
    LEFT JOIN raw.attempts a ON a.attempt_id = s.attempt_id
    WHERE s.status = 'complete' AND s.lb_score IS NOT NULL
)
SELECT
    submission_id,
    competition_id,
    submitted_at,
    attempt_id,
    cv_score,
    lb_score,
    CASE WHEN prev_cv IS NOT NULL THEN metric_sign * (cv_score - prev_cv) END AS delta_cv,
    CASE WHEN prev_lb IS NOT NULL THEN metric_sign * (lb_score - prev_lb) END AS delta_lb,
    (prev_cv IS NOT NULL AND prev_lb IS NOT NULL
        AND metric_sign * (cv_score - prev_cv) > 0
        AND metric_sign * (lb_score - prev_lb) < 0) AS diverged
FROM ordered;

-- bin/blend.py 가중치 — 대회당 최신 1건만 유지(competition_id PK, upsert).
-- 승격 시점마다 cycle/run.py·bin/run_promote_task.py가 자동 재계산해 채운다.
-- 로컬 파일(runs/blend/*.json)은 Airflow task 컨테이너 간 안 보이므로 DB가
-- 유일한 신뢰 사본 — submit.py는 이 값을 소비하지 않는다(계산·저장까지만).
CREATE TABLE IF NOT EXISTS raw.blend_weights (
    competition_id  text PRIMARY KEY,
    pipeline_ids    jsonb,
    weights         jsonb,
    intercept       double precision,
    blend_cv_score  double precision,
    metric          text,
    generated_at    timestamp
);

-- hot-path indexes
CREATE INDEX IF NOT EXISTS idx_attempts_comp_ts     ON raw.attempts (competition_id, run_ts DESC);
CREATE INDEX IF NOT EXISTS idx_attempts_comp_action ON raw.attempts (competition_id, action_type);
CREATE INDEX IF NOT EXISTS idx_reflections_comp_arch ON raw.reflections (competition_id, archived);
