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

create table if not exists raw.submission_budget (
    competition_id  text,
    day             date,
    count           int,
    primary key (competition_id, day)
);
