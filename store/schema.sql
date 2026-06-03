create schema if not exists raw;

create table if not exists raw.competitions (
    competition_id  text primary key,
    name            text,
    task_type       text,
    metric          text,
    metric_sign     int,
    start_ts        timestamp,
    fingerprint     json
);

create table if not exists raw.attempts (
    attempt_id      text primary key,
    competition_id  text,
    run_ts          timestamp,
    stage           text,
    hypothesis      text,
    action_type     text,
    model_type      text,
    params          json,
    features        json,
    cv_score        double,
    cv_fold_var     double,
    lb_score        double,
    label           text,
    gain_vs_best    double,
    error_trace     text
);

create view if not exists score_progression as
select
    a.competition_id,
    row_number() over (
        partition by a.competition_id order by a.run_ts
    ) as attempt_no,
    a.run_ts,
    a.stage,
    a.action_type,
    a.cv_score,
    a.lb_score,
    a.label,
    a.gain_vs_best,
    max(c.metric_sign * a.cv_score) over (
        partition by a.competition_id
        order by a.run_ts
        rows between unbounded preceding and current row
    ) * c.metric_sign as best_so_far
from raw.attempts a
join raw.competitions c using (competition_id);

create table if not exists raw.submission_budget (
    competition_id  text,
    day             date,
    count           int,
    primary key (competition_id, day)
);

create table if not exists raw.reflections (
    reflection_id   text primary key,
    created_at      timestamp,
    attempt_id      text,
    competition_id  text,
    embedded_text   text,
    embedding       float[1024],
    full_lesson     text,
    generality      text,
    label           text,
    reflector_label text,
    gain_vs_best    double,
    archived        boolean default false
);

alter table raw.attempts add column if not exists reflection_ids   text[];
alter table raw.attempts add column if not exists retrieval_scores double[];
alter table raw.attempts add column if not exists duration_sec     double;
alter table raw.attempts add column if not exists code_path        text;
alter table raw.attempts add column if not exists retries          int default 0;

create table if not exists raw.pipelines (
    pipeline_id          text primary key,
    attempt_id           text,
    competition_id       text,
    fingerprint_snapshot json,
    code                 text,
    cv_score             double,
    gain_vs_best         double
);

create view if not exists cold_start_progression as
select
    a.competition_id,
    row_number() over (partition by a.competition_id order by a.run_ts) as attempt_no,
    a.run_ts,
    a.stage,
    a.cv_score,
    max(c.metric_sign * a.cv_score) over (
        partition by a.competition_id order by a.run_ts
        rows between unbounded preceding and current row
    ) * c.metric_sign as best_so_far
from raw.attempts a
join raw.competitions c using (competition_id)
where a.cv_score is not null;

create view if not exists stg_attempts as
select
    a.*,
    c.metric_sign
from raw.attempts a
join raw.competitions c using (competition_id);

create view if not exists stg_attempts_reflexion_only as
select * from stg_attempts
where stage = 'reflexion';

create view if not exists reflection_impact as
with scored as (
    select
        competition_id,
        reflection_ids,
        metric_sign * cv_score
        - max(metric_sign * cv_score) over (
            partition by competition_id
            order by run_ts rows between unbounded preceding and 1 preceding
          ) as gain_vs_best
    from stg_attempts_reflexion_only
),
per_reflection as (
    select unnest(reflection_ids) as reflection_id, gain_vs_best
    from scored
    where reflection_ids is not null
      and gain_vs_best is not null
)
select
    reflection_id,
    count(*)                                          as times_applied,
    round(avg(gain_vs_best), 5)                       as avg_gain,
    sum(case when gain_vs_best > 0 then 1 else 0 end) as jumps,
    round(max(gain_vs_best), 5)                       as best_jump
from per_reflection
group by reflection_id
order by avg_gain desc;
