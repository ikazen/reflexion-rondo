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
    retries          int DEFAULT 0
);

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
    gain_vs_best    double precision,
    archived        boolean DEFAULT false
);

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

CREATE OR REPLACE VIEW stg_attempts AS
SELECT
    a.*,
    c.metric_sign
FROM raw.attempts a
JOIN raw.competitions c USING (competition_id);

CREATE OR REPLACE VIEW stg_attempts_reflexion_only AS
SELECT * FROM stg_attempts
WHERE stage = 'reflexion';

CREATE OR REPLACE VIEW reflection_impact AS
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
),
per_reflection AS (
    SELECT unnest(reflection_ids) AS reflection_id, gain_vs_best
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
