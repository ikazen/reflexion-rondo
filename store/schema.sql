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
