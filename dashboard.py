from __future__ import annotations

from pathlib import Path

import streamlit as st
import polars as pl

from store.db import connect

st.set_page_config(page_title="Reflexion Monitor", layout="wide")
st.title("Reflexion Monitor")

conn = connect()

# --- Competition selector ---
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

# --- Attempts ---
attempts_df = conn.execute(
    """
    select
        row_number() over (order by run_ts) as attempt_no,
        run_ts,
        stage,
        action_type,
        hypothesis,
        cv_score,
        label,
        gain_vs_best,
        duration_sec,
        error_trace is not null as has_error
    from raw.attempts
    where competition_id = ?
    order by run_ts
    """,
    [comp_id],
).pl()

if attempts_df.is_empty():
    st.info("No attempts yet for this competition.")
    st.stop()

best_so_far = conn.execute(
    """
    select attempt_no, best_so_far
    from score_progression
    where competition_id = ?
    order by attempt_no
    """,
    [comp_id],
).pl()

# --- KPIs ---
total = len(attempts_df)
errors = attempts_df["has_error"].sum()
jumps = (attempts_df["label"] == "jump").sum()
best_cv = best_so_far["best_so_far"].max() if not best_so_far.is_empty() else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total attempts", total)
col2.metric("Jumps", jumps)
col3.metric("Errors", errors)
col4.metric("Best CV", f"{best_cv:.5f}" if best_cv else "-")

st.divider()

# --- Score progression ---
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

# --- Action type & Label distribution ---
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

# --- reflection_impact ---
st.subheader("Top Lessons by Impact")

impact_df = conn.execute(
    """
    select i.reflection_id, i.times_applied, i.avg_gain, i.jumps, i.best_jump,
           r.embedded_text, r.generality
    from reflection_impact i
    join raw.reflections r using (reflection_id)
    where r.competition_id = ? or r.generality in ('L2_class', 'L3_general')
    order by i.avg_gain desc
    limit 10
    """,
    [comp_id],
).pl()

if impact_df.is_empty():
    st.info("No reflection impact data yet (need reflexion stage attempts).")
else:
    st.dataframe(
        impact_df.select(["embedded_text", "generality", "times_applied", "avg_gain", "jumps", "best_jump"]),
        use_container_width=True,
    )

st.divider()

# --- Recent attempts table ---
st.subheader("Recent Attempts")

display_df = (
    attempts_df.sort("attempt_no", descending=True)
    .head(20)
    .select([
        "attempt_no", "run_ts", "stage", "action_type",
        "cv_score", "label", "gain_vs_best", "duration_sec", "has_error",
        "hypothesis",
    ])
)
st.dataframe(display_df, use_container_width=True)

conn.close()
