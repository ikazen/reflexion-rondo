from __future__ import annotations

from pathlib import Path

import duckdb
import streamlit as st
import polars as pl

DB_PATH = __import__("pathlib").Path(__file__).parent / "runs" / "reflexion.duckdb"

st.set_page_config(page_title="Reflexion Monitor", layout="wide")
st.title("Reflexion Monitor")

if not DB_PATH.exists():
    st.error(f"DB not found: {DB_PATH}")
    st.stop()

try:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
except duckdb.IOException:
    st.warning("사이클 실행 중입니다. 잠시 후 새로고침 하세요.")
    st.stop()

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

# 컬럼 존재 여부 확인
cols = {r[0] for r in conn.execute("select column_name from information_schema.columns where table_schema='raw' and table_name='attempts'").fetchall()}
has_duration = "duration_sec" in cols

_duration_col = "duration_sec" if has_duration else "null as duration_sec"
_retries_col  = "retries"       if "retries"      in cols else "0 as retries"

# --- Attempts ---
attempts_df = conn.execute(
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

# --- Cold-start Progression (전 대회 비교) ---
st.subheader("Cold-start Progression")

csp_df = conn.execute(
    """
    select competition_id, attempt_no, stage, cv_score, best_so_far
    from cold_start_progression
    order by competition_id, attempt_no
    """
).pl()

if csp_df.is_empty():
    st.info("No cold_start_progression data yet.")
else:
    # warm_start_ratio: bootstrap_best / overall_best per competition
    summary = conn.execute(
        """
        select
            c.competition_id,
            c.name,
            round(max(case when p.stage = 'bootstrap' then p.best_so_far else null end), 5) as bootstrap_best,
            round(max(p.best_so_far), 5) as overall_best,
            round(
                max(case when p.stage = 'bootstrap' then p.best_so_far else null end)
                / nullif(max(p.best_so_far), 0)
            , 4) as warm_start_ratio
        from cold_start_progression p
        join raw.competitions c using (competition_id)
        group by c.competition_id, c.name
        order by c.competition_id
        """
    ).pl()
    st.dataframe(summary, use_container_width=True)

    st.caption("warm_start_ratio = bootstrap_best / overall_best (높을수록 cold-start 효과 좋음)")

    # 대회별 best_so_far 추세 비교
    pivot = (
        csp_df
        .filter(pl.col("best_so_far").is_not_null())
        .select(["competition_id", "attempt_no", "best_so_far"])
        .pivot(on="competition_id", index="attempt_no", values="best_so_far", aggregate_function="mean")
        .sort("attempt_no")
    )
    st.line_chart(pivot.to_pandas().set_index("attempt_no"))

st.divider()

# --- Recent attempts table ---
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
