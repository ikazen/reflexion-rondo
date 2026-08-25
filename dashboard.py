"""Streamlit 운영 대시보드 — Postgres 파생 뷰 직접 조회(daemon API 미경유, GH #65).

Fleet Overview + 대회 선택 종속 섹션(Submissions/Quarantine/Blend 등)로 구성.
"""
from __future__ import annotations

import altair as alt
import streamlit as st
import polars as pl

from config.settings import ACTION_TYPES
from config.competitions import active_competition_ids
from cycle.stagnation import detect_stagnation
from store.db import PgConn, connect


def _traffic_light(*, green: bool, red: bool) -> str:
    """bin/api.py:_traffic_light와 동일 판정(red가 green보다 우선) — API를 거치지
    않고 대시보드가 같은 뷰를 직접 재계산하는 GH #65 설계라 여기서도 재현한다."""
    if red:
        return "🔴"
    if green:
        return "🟢"
    return "🟡"


def _rows_df(rows: list[tuple], columns: list[str]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame({c: [] for c in columns})
    # infer_schema_length 기본값(100)이면 앞쪽 100행이 전부 null인 컬럼(예: 대부분의
    # attempt에는 없는 holdout_score)에서 타입을 Null로 오추론해, 뒤에서 실제 float
    # 값이 나오면 "could not append value" ComputeError로 죽는다 — 전체 행을 스캔.
    return pl.DataFrame(rows, schema=columns, orient="row", infer_schema_length=None)


@st.cache_data(ttl=60)
def _query_df(_conn: PgConn, query: str, columns: list[str], params: list | None = None) -> pl.DataFrame:
    """결과를 60초 캐싱한다 — Streamlit은 상호작용마다 스크립트를 처음부터 재실행해
    캐싱 없이는 대회 선택·탭 클릭마다 쿼리 10개+가 매번 다시 나간다. `_conn`은
    언더스코어 프리픽스라 캐시 키 해싱에서 제외(연결 객체 자체가 매 rerun 새로
    생성돼도 캐시가 깨지지 않음) — bin/api.py의 60초 TTL 캐시(_cache.get)와 동일 값.
    """
    return _rows_df(_conn.execute(query, params or []).fetchall(), columns)


@st.cache_data(ttl=60)
def _fetch_stagnation(_conn: PgConn, competition_id: str):
    return detect_stagnation(_conn, competition_id)


_LB_STALE_DAYS = 7  # #233 완료 기준: deep tier 전 대회가 주 1회 이상 LB 갱신


def _fleet_attention(row: dict) -> str:
    """대회 하나의 "지금 봐야 하나" 판정 — _traffic_light와 신호 조합이 달라
    별도 함수. 천장 진단(2026-08-03)에서 대회 26개를 하나씩 SQL로 찔러 찾았던
    "baseline 없음/큐 방치/OOM 과다"를 한 테이블에서 바로 보이게 하는 게 목적."""
    if row["auto_submit_paused_reason"]:
        return "🔴"
    attempts = row["attempts_14d"] or 0
    if attempts > 0 and (row["errors_14d"] or 0) / attempts > 0.7:
        return "🔴"
    if (row["confirmed"] or 0) == 0 and (row["quarantined"] or 0) > 0:
        return "🔴"  # baseline이 전부 격리돼 지금 하나도 없음
    if (row["confirmed"] or 0) == 0:
        return "🟡"  # 아직 baseline 자체가 없음
    if row["is_active"] and (
        row["days_since_lb"] is None or row["days_since_lb"] > _LB_STALE_DAYS
    ):
        return "🟡"  # deep tier인데 외부 검증(LB)을 주 1회도 못 받고 있다(#233 완료 기준)
    if row["queue_status"] == "pending":
        return "🟡"  # 예약만 되고 daemon이 아직 안 돎
    if (row["jumps_14d"] or 0) == 0:
        return "🟡"
    return "🟢"


st.set_page_config(page_title="Reflexion Monitor", layout="wide")
st.title("Reflexion Monitor")

try:
    conn = connect(apply_schema=False)
except Exception as exc:
    st.error(f"DB connection failed: {exc}")
    st.stop()

st.subheader("Fleet Overview")
st.caption(
    "전 대회 한눈에 — 어디부터 볼지 여기서 고른다. attention: 🔴 즉시 확인 / 🟡 관찰 필요 / 🟢 정상. "
    "lb_percentile은 대회 리더보드 분포 기준 백분위(높을수록 좋음) — 대회 간 비교가 가능한 유일한 지표다."
)

# raw.cycle_queue.competition은 슬러그(s4e7), 나머지 테이블은 풀 competition_id
# (playground-series-s4e7) — split_part로 조인 키를 맞춘다.
queue_latest = _query_df(
    conn,
    """
    select distinct on (competition) competition, status, n_cycles, cycles_done, started_at
    from raw.cycle_queue order by competition, created_at desc
    """,
    ["competition", "queue_status", "n_cycles", "cycles_done", "started_at"],
)
pipeline_counts = _query_df(
    conn,
    """
    select competition_id,
           count(*) filter (where invalid_reason is null) as confirmed,
           count(*) filter (where invalid_reason is not null) as quarantined
    from raw.pipelines group by competition_id
    """,
    ["competition_id", "confirmed", "quarantined"],
)
recent_activity = _query_df(
    conn,
    """
    select competition_id, count(*) as attempts_14d,
           count(*) filter (where label='jump') as jumps_14d,
           count(*) filter (where label='error') as errors_14d,
           count(*) filter (where error_trace like '%%rc=-9%%') as oom_14d,
           max(run_ts) as last_attempt
    from raw.attempts where run_ts > now() - interval '14 days'
    group by competition_id
    """,
    ["competition_id", "attempts_14d", "jumps_14d", "errors_14d", "oom_14d", "last_attempt"],
)
paused = _query_df(
    conn,
    "select competition_id, auto_submit_paused_reason from raw.competitions where auto_submit_paused_reason is not null",
    ["competition_id", "auto_submit_paused_reason"],
)
all_comps = _query_df(
    conn,
    "select competition_id, name from raw.competitions order by competition_id",
    ["competition_id", "name"],
)
lb_status = _query_df(
    conn,
    """
    select distinct on (competition_id)
           competition_id, lb_percentile,
           extract(epoch from (now() - submitted_at)) / 86400.0 as days_since_lb
    from raw.kaggle_submissions
    where status = 'complete' and lb_score is not null
    order by competition_id, submitted_at desc
    """,
    ["competition_id", "lb_percentile", "days_since_lb"],
)

# 조인 소스가 비어 있으면(예: 지금은 paused=[]) _rows_df가 컬럼 dtype을 Null로 만들어
# join key 타입이 안 맞아 죽는다 — join key만 명시 캐스팅해 방어.
queue_latest = queue_latest.with_columns(pl.col("competition").cast(pl.Utf8))
pipeline_counts = pipeline_counts.with_columns(pl.col("competition_id").cast(pl.Utf8))
recent_activity = recent_activity.with_columns(pl.col("competition_id").cast(pl.Utf8))
paused = paused.with_columns(pl.col("competition_id").cast(pl.Utf8))
lb_status = lb_status.with_columns(pl.col("competition_id").cast(pl.Utf8))

fleet = (
    all_comps
    .with_columns(pl.col("competition_id").str.split("-").list.get(2).alias("_slug"))
    .join(queue_latest, left_on="_slug", right_on="competition", how="left")
    .join(pipeline_counts, on="competition_id", how="left")
    .join(recent_activity, on="competition_id", how="left")
    .join(paused, on="competition_id", how="left")
    .join(lb_status, on="competition_id", how="left")
    .drop("_slug")
)

active_ids = active_competition_ids()
fleet = fleet.with_columns(
    pl.col("competition_id").is_in(list(active_ids)).alias("is_active")
)
fleet_rows = [_fleet_attention(r) for r in fleet.iter_rows(named=True)]
fleet = fleet.with_columns(pl.Series("attention", fleet_rows)).sort(
    pl.col("attention").replace_strict({"🔴": 0, "🟡": 1, "🟢": 2}, return_dtype=pl.Int32)
)

st.dataframe(
    fleet.select([
        "attention", "competition_id", "name", "is_active", "queue_status", "cycles_done",
        "n_cycles", "confirmed", "quarantined", "attempts_14d", "jumps_14d", "errors_14d",
        "oom_14d", "lb_percentile", "days_since_lb", "auto_submit_paused_reason", "last_attempt",
    ]),
    use_container_width=True,
    height=min(35 * (len(fleet) + 1), 500),
)

st.divider()

competitions = _query_df(
    conn,
    "select competition_id, name from raw.competitions order by start_ts desc",
    ["competition_id", "name"],
).rows()

if not competitions:
    conn.close()
    st.info("No competitions registered yet.")
    st.stop()

comp_options = {f"{c[0]} — {c[1]}": c[0] for c in competitions}
selected_label = st.sidebar.selectbox("Competition", list(comp_options.keys()))
comp_id = comp_options[selected_label]

cols = set(
    _query_df(
        conn,
        "select column_name from information_schema.columns where table_schema='raw' and table_name='attempts'",
        ["column_name"],
    )["column_name"]
)
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

st.subheader("Health Signals")

# lesson_funnel/bandit_calibration/error_recurrence는 아래 상세 섹션에서도 쓰인다 —
# 컬럼 목록만 다르게 두 번 쿼리하면 캐시 키가 갈려 캐싱 혜택이 없다. 여기서 전체
# 컬럼으로 한 번만 쿼리해 상세 섹션에서 그대로 재사용한다(쿼리 왕복 3회 절감).
funnel_df = _query_df(
    conn,
    """
    select written, total_attempts, retrieved, cited, positive_gain,
           retrieve_rate, cite_rate, gain_rate, retrieved_precise
    from lesson_funnel where competition_id = %s
    """,
    ["written", "total_attempts", "retrieved", "cited", "positive_gain",
     "retrieve_rate", "cite_rate", "gain_rate", "retrieved_precise"],
    [comp_id],
)
funnel_row = funnel_df.row(0) if not funnel_df.is_empty() else None
cite_rate = float(funnel_row[6]) if funnel_row and funnel_row[6] is not None else 0.0
retrieved_precise = bool(funnel_row[8]) if funnel_row else False

jumps_last10 = int(
    attempts_df.filter(pl.col("stage") == "reflexion")
    .sort("run_ts", descending=True)
    .head(10)["label"].eq("jump").sum()
)

stagnation = _fetch_stagnation(conn, comp_id)

bandit_df = _query_df(
    conn,
    """
    select action_type, posterior_mean, weighted_success, calibration_gap, picks
    from bandit_calibration where scope_key = %s
    order by calibration_gap desc nulls last
    """,
    ["action_type", "posterior_mean", "weighted_success", "calibration_gap", "picks"],
    [comp_id],
)
posteriors = bandit_df["posterior_mean"].drop_nulls()
gaps = bandit_df["calibration_gap"].drop_nulls()
posterior_spread = float(posteriors.max() - posteriors.min()) if len(posteriors) else 0.0
calibration_gap = float(gaps.max()) if len(gaps) else 0.0

err_df = _query_df(
    conn,
    """
    select action_type, error_signature, total, first_seen, last_seen,
           pitfall_active, occurrences_after_active, has_avoid_lesson
    from error_recurrence where competition_id = %s
    order by pitfall_active desc, occurrences_after_active desc
    limit 30
    """,
    ["action_type", "error_signature", "total", "first_seen", "last_seen",
     "pitfall_active", "occurrences_after_active", "has_avoid_lesson"],
    [comp_id],
)
active_rows = err_df.filter(pl.col("pitfall_active"))
active_total = int(active_rows["total"].sum()) if not active_rows.is_empty() else 0
repeat_after_pitfall_rate = (
    float(active_rows["occurrences_after_active"].sum()) / active_total if active_total else 0.0
)

recent_labels = attempts_df.filter(pl.col("stage") == "reflexion").sort("run_ts", descending=True).head(200)
error_rate_slope = 0
if len(recent_labels) >= 20:
    mid = len(recent_labels) // 2
    recent_rate = recent_labels.head(mid)["label"].eq("error").mean()
    earlier_rate = recent_labels.tail(len(recent_labels) - mid)["label"].eq("error").mean()
    error_rate_slope = -1 if recent_rate < earlier_rate else (1 if recent_rate > earlier_rate else 0)

action_coverage = len(ACTION_TYPES) - len(stagnation.underused_actions)

h1, h2, h3, h4 = st.columns(4)
with h1:
    status = _traffic_light(
        green=(cite_rate >= 0.30 and jumps_last10 >= 1),
        red=(cite_rate < 0.10 or (jumps_last10 == 0 and stagnation.is_stagnant)),
    )
    st.metric(f"{status} Accumulation", f"cite {cite_rate:.2f}")
    st.caption(f"jumps(last10)={jumps_last10}" + ("" if retrieved_precise else " (approx)"))
with h2:
    status = _traffic_light(
        green=(posterior_spread >= 0.15 and calibration_gap <= 0.15),
        red=(posterior_spread < 0.05 or calibration_gap > 0.30),
    )
    st.metric(f"{status} Bandit", f"gap {calibration_gap:.2f}")
    st.caption(f"posterior spread={posterior_spread:.2f}")
with h3:
    status = _traffic_light(
        green=(error_rate_slope < 0 and repeat_after_pitfall_rate <= 0.20),
        red=(repeat_after_pitfall_rate > 0.50),
    )
    st.metric(f"{status} Antipattern", f"repeat {repeat_after_pitfall_rate:.2f}")
    st.caption(f"error slope={error_rate_slope:+d}")
with h4:
    status = _traffic_light(
        green=(action_coverage >= 3),
        red=(stagnation.is_stagnant and action_coverage <= 1),
    )
    st.metric(f"{status} Exploration", f"{action_coverage}/{len(ACTION_TYPES)} types")
    st.caption(f"stagnant_for={stagnation.stagnant_for}")

st.divider()

st.subheader("Submissions · Quarantine · Blend")

paused_reason = conn.execute(
    "select auto_submit_paused_reason from raw.competitions where competition_id = %s",
    [comp_id],
).fetchone()
if paused_reason and paused_reason[0]:
    st.warning(f"auto-submit 일시중단: {paused_reason[0]} — 자동 해제 없음, runbook.md §4-3 절차로 수동 해제 필요")

calib_df = _query_df(
    conn,
    """
    select submitted_at, cv_score, lb_score, delta_cv, delta_lb, diverged
    from cv_lb_calibration where competition_id = %s
    order by submitted_at desc
    limit 20
    """,
    ["submitted_at", "cv_score", "lb_score", "delta_cv", "delta_lb", "diverged"],
    [comp_id],
)
quarantine_df = _query_df(
    conn,
    """
    select pipeline_id, cv_score, invalid_reason
    from raw.pipelines where competition_id = %s and invalid_reason is not null
    order by cv_score desc
    """,
    ["pipeline_id", "cv_score", "invalid_reason"],
    [comp_id],
)
sq_left, sq_right = st.columns(2)
with sq_left:
    st.caption("CV ↔ LB 정합 (제출 이력)")
    if calib_df.is_empty():
        st.info("No submissions yet.")
    else:
        n_diverged = int(calib_df["diverged"].fill_null(False).sum())
        if n_diverged:
            st.warning(f"CV는 개선인데 LB는 악화된 제출 {n_diverged}건 — cv_lb_calibration 발산")
        st.dataframe(calib_df, use_container_width=True)

gap_df = _query_df(
    conn,
    """
    select submitted_at, cv_score, lb_score, lb_percentile, cv_lb_gap
    from cv_lb_gap_trend where competition_id = %s
    order by submitted_at desc
    limit 20
    """,
    ["submitted_at", "cv_score", "lb_score", "lb_percentile", "cv_lb_gap"],
    [comp_id],
)

with sq_right:
    st.caption("cv-LB 갭 (양수 = CV가 LB보다 낙관적) · lb_percentile은 높을수록 좋음")
    if gap_df.is_empty():
        st.info("No LB-scored submissions yet.")
    else:
        st.dataframe(gap_df, use_container_width=True)

st.caption(f"격리된 파이프라인 ({len(quarantine_df)}건)")
if quarantine_df.is_empty():
    st.info("No quarantined pipelines.")
else:
    st.dataframe(quarantine_df, use_container_width=True)

st.divider()

st.subheader("CV Score Progression")

has_scores = attempts_df["cv_score"].drop_nulls()
if not has_scores.is_empty():
    chart_df = attempts_df.select(["attempt_no", "cv_score", "label"]).filter(
        pl.col("cv_score").is_not_null()
    )
    if not best_so_far.is_empty():
        chart_df = chart_df.join(best_so_far, on="attempt_no", how="left")

    pdf = chart_df.to_pandas()
    # 명시 type(:Q) — 자동추론에 맡기면 레이어마다 다르게 추론돼 Vega-Lite가
    # "Scale bindings are currently only supported for..." 경고를 낸다.
    # .interactive()(pan/zoom 바인딩)도 제거 — 정적 모니터링 차트라 불필요하고,
    # 레이어 중 하나라도 값이 전부 null이면 바인딩 스케일 계산이 Infinity로 죽는다.
    line = alt.Chart(pdf).mark_line().encode(x="attempt_no:Q", y="cv_score:Q")
    layers = [line]
    if "best_so_far" in pdf.columns and pdf["best_so_far"].notna().any():
        layers.append(alt.Chart(pdf).mark_line(color="orange").encode(x="attempt_no:Q", y="best_so_far:Q"))
    jump_points = pdf[pdf["label"] == "jump"]
    if not jump_points.empty:
        layers.append(
            alt.Chart(jump_points).mark_point(size=80, color="green", filled=True).encode(
                x="attempt_no:Q", y="cv_score:Q", tooltip=["attempt_no", "cv_score"]
            )
        )
    st.altair_chart(alt.layer(*layers), use_container_width=True)
    if stagnation.is_stagnant:
        st.warning(f"정체 중 — 최근 {stagnation.stagnant_for} attempt 동안 jump 없음")
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

st.subheader("Lesson Funnel")

# funnel_df는 Health Signals에서 이미 전체 컬럼으로 쿼리해 둔 것을 재사용.
if funnel_df.is_empty():
    st.info("No lesson_funnel data yet.")
else:
    f = funnel_df.row(0, named=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Written", f["written"])
    fc2.metric("Retrieved", f["retrieved"], f"rate {f['retrieve_rate']}" if f["retrieve_rate"] is not None else None)
    fc3.metric("Cited", f["cited"], f"rate {f['cite_rate']}" if f["cite_rate"] is not None else None)
    fc4.metric("Positive gain", f["positive_gain"], f"rate {f['gain_rate']}" if f["gain_rate"] is not None else None)
    if not f["retrieved_precise"]:
        st.caption("retrieve_rate는 P1 계측 이전 데이터가 섞여 근사치일 수 있음")

    dead_tab, dup_tab = st.tabs(["Dead lessons", "Near-duplicates"])
    with dead_tab:
        dead_df = _query_df(
            conn,
            """
            select reflection_id, lesson_type, generality, times_cited, avg_gain, reason
            from lesson_dead where competition_id = %s
            order by times_cited desc
            limit 30
            """,
            ["reflection_id", "lesson_type", "generality", "times_cited", "avg_gain", "reason"],
            [comp_id],
        )
        if dead_df.is_empty():
            st.info("No dead lessons.")
        else:
            st.dataframe(dead_df, use_container_width=True)
    with dup_tab:
        dup_df = _query_df(
            conn,
            """
            select reflection_id_a, reflection_id_b, cos_sim
            from lesson_duplicates where competition_id = %s
            order by cos_sim desc
            limit 30
            """,
            ["reflection_id_a", "reflection_id_b", "cos_sim"],
            [comp_id],
        )
        if dup_df.is_empty():
            st.info("No near-duplicate lessons (cos_sim > 0.90).")
        else:
            st.dataframe(dup_df, use_container_width=True)

st.divider()

st.subheader("Bandit Calibration")

# bandit_df는 Health Signals에서 이미 전체 컬럼으로 쿼리해 둔 것을 재사용.
if bandit_df.is_empty():
    st.info("No bandit_calibration data yet.")
else:
    bc_left, bc_right = st.columns(2)
    with bc_left:
        # weighted_success는 LEFT JOIN이라 한 번도 안 뽑힌 action_type에서 null —
        # 차트에 null 열이 섞이면 Vega-Lite가 해당 필드 도메인을 Infinity로 잡고 경고한다.
        chart_rows = bandit_df.select(["action_type", "posterior_mean", "weighted_success"]).drop_nulls(
            "weighted_success"
        )
        if chart_rows.is_empty():
            st.info("No action_type has been picked yet (weighted_success unavailable).")
        else:
            st.bar_chart(chart_rows.to_pandas().set_index("action_type"))
    with bc_right:
        st.dataframe(bandit_df, use_container_width=True)

st.divider()

st.subheader("Error Recurrence")

# err_df는 Health Signals에서 이미 전체 컬럼으로 쿼리해 둔 것을 재사용.
if err_df.is_empty():
    st.info("No error_recurrence data yet.")
else:
    unresolved = err_df.filter(
        (pl.col("pitfall_active")) & (pl.col("occurrences_after_active") > 0)
    )
    if not unresolved.is_empty():
        st.warning(f"pitfall 주입 후에도 재발한 에러 시그니처 {len(unresolved)}건")
    st.dataframe(err_df, use_container_width=True)

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

st.subheader("Transfer Matrix")
st.caption("행=교훈 출처 대회, 열=인용한 대회. 대각선 밖(source != target)이 실제 cross-competition 전이.")

tm_df = _query_df(
    conn,
    "select source_comp, target_comp, citations from transfer_matrix",
    ["source_comp", "target_comp", "citations"],
)

if tm_df.is_empty():
    st.info("No transfer_matrix data yet.")
else:
    tm_pivot = (
        tm_df.pivot(on="target_comp", index="source_comp", values="citations", aggregate_function="sum")
        .fill_null(0)
        .sort("source_comp")
    )
    tm_pandas = tm_pivot.to_pandas().set_index("source_comp")
    st.dataframe(tm_pandas.style.background_gradient(cmap="Blues"), use_container_width=True)

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
