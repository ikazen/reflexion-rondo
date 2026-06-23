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

DROP VIEW IF EXISTS reflection_impact;
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
        metric_sign * cv_score
        - max(metric_sign * cv_score) OVER (
            PARTITION BY competition_id
            ORDER BY run_ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
          ) AS gain_vs_best
    FROM stg_attempts_reflexion_only
    WHERE was_promoted IS NOT FALSE  -- NULL=legacy (promoted), TRUE=winner, FALSE=super-cycle loser excluded
),
per_reflection AS (
    SELECT
        unnest(reflection_ids) AS reflection_id,
        gain_vs_best / array_length(reflection_ids, 1) AS gain_vs_best
    FROM scored
    WHERE reflection_ids IS NOT NULL
      AND gain_vs_best IS NOT NULL
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

-- BON-110: super-cycle 공유 retrieve 컨텍스트 (retrieve task → attempt tasks)
CREATE TABLE IF NOT EXISTS raw.super_cycle_context (
    queue_id          text PRIMARY KEY,
    super_cycle_id    text NOT NULL,
    competition_id    text NOT NULL,
    prev_best_cv      double precision,
    lessons           jsonb NOT NULL,
    assigned_actions  jsonb,
    created_at        timestamp DEFAULT now()
);
ALTER TABLE raw.super_cycle_context ADD COLUMN IF NOT EXISTS assigned_actions jsonb;

-- BON-109: action_type별 Beta-Bernoulli 밴딧 (advise용, stagnation 승격)
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

-- BON-184: hot-path indexes
CREATE INDEX IF NOT EXISTS idx_attempts_comp_ts     ON raw.attempts (competition_id, run_ts DESC);
CREATE INDEX IF NOT EXISTS idx_attempts_comp_action ON raw.attempts (competition_id, action_type);
CREATE INDEX IF NOT EXISTS idx_reflections_comp_arch ON raw.reflections (competition_id, archived);
