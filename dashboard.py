from __future__ import annotations

import streamlit as st
import polars as pl

from store.db import PgConn, connect


def _rows_df(rows: list[tuple], columns: list[str]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame({c: [] for c in columns})
    return pl.DataFrame(rows, schema=columns, orient="row")


def _query_df(conn: PgConn, query: str, columns: list[str], params: list | None = None) -> pl.DataFrame:
    return _rows_df(conn.execute(query, params or []).fetchall(), columns)

st.set_page_config(page_title="Reflexion Monitor", layout="wide")
st.title("Reflexion Monitor")

try:
    conn = connect(apply_schema=False)
except Exception as exc:
    st.error(f"DB connection failed: {exc}")
    st.stop()

competitions = conn.execute(
    "select competition_id, name from raw.competitions order by start_ts desc"
).fetchall()

if not competitions:
    conn.close()
    st.info("No competitions registered yet.")
    st.stop()

comp_options = {f"{c[0]} — {c[1]}": c[0] for c in competitions}
selected_label = st.sidebar.selectbox("Competition", list(comp_options.keys()))
comp_id = comp_options[selected_label]

cols = {r[0] for r in conn.execute("select column_name from information_schema.columns where table_schema='raw' and table_name='attempts'").fetchall()}
has_duration = "duration_sec" in cols
has_holdout = "holdout_score" in cols

_duration_col = "duration_sec" if has_duration else "null as duration_sec"
_retries_col  = "retries"       if "retries"      in cols else "0 as retries"
_holdout_col  = "holdout_score" if has_holdout   else "null as holdout_score"

attempts_df = _query_df(
    conn,
    f"""
    select
        row_number() over (order by run_ts) as attempt_no,
        run_ts,
        stage,
        action_type,
        hypothesis,
        cv_score,
        label,
        gain_vs_best,
        {_duration_col},
        {_retries_col},
        {_holdout_col},
        error_trace is not null as has_error
    from raw.attempts
    where competition_id = %s
    order by run_ts
    """,
    [
        "attempt_no", "run_ts", "stage", "action_type", "hypothesis",
        "cv_score", "label", "gain_vs_best", "duration_sec", "retries",
        "holdout_score", "has_error",
    ],
    [comp_id],
)

if attempts_df.is_empty():
    st.info("No attempts yet for this competition.")
    st.stop()

best_so_far = _query_df(
    conn,
    """
    select attempt_no, best_so_far
    from score_progression
    where competition_id = %s
    order by attempt_no
    """,
    ["attempt_no", "best_so_far"],
    [comp_id],
)

total = len(attempts_df)
errors = attempts_df["has_error"].sum()
jumps = (attempts_df["label"] == "jump").sum()
best_values = best_so_far["best_so_far"].drop_nulls() if not best_so_far.is_empty() else []
best_cv = best_values[-1] if len(best_values) else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total attempts", total)
col2.metric("Jumps", jumps)
col3.metric("Errors", errors)
col4.metric("Best CV", f"{best_cv:.5f}" if best_cv else "-")

st.divider()

st.subheader("CV Score Progression")

has_scores = attempts_df["cv_score"].drop_nulls()
if not has_scores.is_empty():
    chart_df = attempts_df.select(["attempt_no", "cv_score"]).filter(
        pl.col("cv_score").is_not_null()
    )
    if not best_so_far.is_empty():
        chart_df = chart_df.join(best_so_far, on="attempt_no", how="left")

    st.line_chart(
        chart_df.to_pandas().set_index("attempt_no")[
            [c for c in ["cv_score", "best_so_far"] if c in chart_df.columns]
        ]
    )
else:
    st.info("No CV scores recorded yet.")

st.divider()

if has_holdout:
    holdout_rows = attempts_df.filter(pl.col("holdout_score").is_not_null())
    if not holdout_rows.is_empty():
        st.subheader("CV vs Holdout Divergence")
        div_df = holdout_rows.select(["attempt_no", "cv_score", "holdout_score"]).filter(
            pl.col("cv_score").is_not_null()
        ).with_columns(
            (pl.col("cv_score") - pl.col("holdout_score")).alias("cv_minus_holdout")
        )
        st.line_chart(
            div_df.to_pandas().set_index("attempt_no")[["cv_score", "holdout_score"]],
        )
        st.caption(
            f"promoted attempts: {len(holdout_rows)}  |  "
            f"avg(cv - holdout): {div_df['cv_minus_holdout'].mean():.5f}"
        )
        st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Action Type Distribution")
    at_counts = (
        attempts_df.group_by("action_type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    st.bar_chart(at_counts.to_pandas().set_index("action_type"))

with col_right:
    st.subheader("Label Distribution")
    label_counts = (
        attempts_df.group_by("label")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    st.bar_chart(label_counts.to_pandas().set_index("label"))

st.divider()

st.subheader("Top Lessons by Impact")

impact_df = _query_df(
    conn,
    """
    select i.reflection_id, i.times_applied, i.avg_gain, i.jumps, i.best_jump,
           r.embedded_text, r.generality
    from reflection_impact i
    join raw.reflections r using (reflection_id)
    where r.competition_id = %s or r.generality in ('L2_class', 'L3_general')
    order by i.avg_gain desc
    limit 10
    """,
    ["reflection_id", "times_applied", "avg_gain", "jumps", "best_jump", "embedded_text", "generality"],
    [comp_id],
)

if impact_df.is_empty():
    st.info("No reflection impact data yet (need reflexion stage attempts).")
else:
    st.dataframe(
        impact_df.select(["embedded_text", "generality", "times_applied", "avg_gain", "jumps", "best_jump"]),
        use_container_width=True,
    )

st.divider()

st.subheader("Cold-start Progression")

csp_df = _query_df(
    conn,
    """
    select competition_id, attempt_no, stage, cv_score, best_so_far
    from cold_start_progression
    order by competition_id, attempt_no
    """,
    ["competition_id", "attempt_no", "stage", "cv_score", "best_so_far"],
)

if csp_df.is_empty():
    st.info("No cold_start_progression data yet.")
else:
    summary = _query_df(
        conn,
        """
        select
            c.competition_id,
            c.name,
            max(case when p.stage = 'bootstrap' then p.best_so_far else null end) as bootstrap_best,
            max(p.best_so_far) as overall_best
        from cold_start_progression p
        join raw.competitions c using (competition_id)
        group by c.competition_id, c.name
        order by c.competition_id
        """,
        ["competition_id", "name", "bootstrap_best", "overall_best"],
    )
    summary = summary.with_columns(
        pl.col("bootstrap_best").round(5),
        pl.col("overall_best").round(5),
        pl.when(pl.col("overall_best") != 0)
        .then((pl.col("bootstrap_best") / pl.col("overall_best")).round(4))
        .otherwise(None)
        .alias("warm_start_ratio"),
    )
    st.dataframe(summary, use_container_width=True)

    st.caption("warm_start_ratio = bootstrap_best / overall_best (높을수록 cold-start 효과 좋음)")

    pivot = (
        csp_df
        .filter(pl.col("best_so_far").is_not_null())
        .select(["competition_id", "attempt_no", "best_so_far"])
        .pivot(on="competition_id", index="attempt_no", values="best_so_far", aggregate_function="mean")
        .sort("attempt_no")
    )
    st.line_chart(pivot.to_pandas().set_index("attempt_no"))

st.divider()

st.subheader("Recent Attempts")

display_df = (
    attempts_df.sort("attempt_no", descending=True)
    .head(20)
    .select([
        "attempt_no", "run_ts", "stage", "action_type",
        "cv_score", "label", "gain_vs_best", "duration_sec", "retries", "has_error",
        "hypothesis",
    ])
)
st.dataframe(display_df, use_container_width=True)

conn.close()
