"""auto-submit 유의성 게이트 (#87).

기존 auto-submit은 best attempt_id가 바뀌었는지만 보고 재제출해, fold noise 수준
(실측 0.07~0.49σ)의 CV 차이로도 재제출되어 LB가 사실상 랜덤워크했다(s6e7 07-25:
0.07σ 차이로 재제출했는데 LB는 그 폭의 14배를 반대 방향으로 잃음). 승격 게이트
(cycle/run.py)와 동일한 is_significant_gain을 재사용해 제출 게이트도 같은 기준으로
맞춘다 — _submission_gain_significant가 그 판정, auto_submit이 실제 게이트 삽입 지점.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from bin.api import DaemonState, _best_attempt, _submission_gain_significant, create_app  # noqa: E402
from config.settings import SUBMISSIONS_PER_DAY  # noqa: E402


def _conn_for(metric_sign, candidate_row, baseline_row):
    """metric_sign 조회 1회 → candidate attempt 조회 1회 → baseline attempt 조회 1회
    순서로 conn.execute가 호출된다는 가정 하에 side_effect로 순서대로 값을 흘려보낸다."""
    conn = MagicMock()
    results = [
        MagicMock(fetchone=MagicMock(return_value=(metric_sign,))),
        MagicMock(fetchone=MagicMock(return_value=candidate_row)),
        MagicMock(fetchone=MagicMock(return_value=baseline_row)),
    ]
    conn.execute.side_effect = results
    return conn



def test_noise_level_gain_is_not_significant():
    """s6e7 07-25 재현: cv 차이가 fold_std의 0.07배뿐이면 유의하지 않다."""
    conn = _conn_for(
        metric_sign=1,
        candidate_row=(0.949607, 0.001182 ** 2, None),
        baseline_row=(0.949523, 0.001220 ** 2, None),
    )
    assert not _submission_gain_significant(conn, "playground-series-s6e7", "cand", "base")


def test_large_gain_is_significant():
    """gain이 LABEL_Z*fold_std 임계값을 넘으면(절대-margin 폴백) 유의하다."""
    from config.settings import LABEL_Z

    fold_std = 0.001
    conn = _conn_for(
        metric_sign=1,
        candidate_row=(0.900 + LABEL_Z * fold_std + 0.0005, fold_std ** 2, None),
        baseline_row=(0.900, fold_std ** 2, None),
    )
    assert _submission_gain_significant(conn, "playground-series-s6e6", "cand", "base")


def test_paired_fold_scores_used_when_available():
    """fold_scores가 있으면 절대-margin이 아니라 paired t-test로 판정한다."""
    conn = _conn_for(
        metric_sign=1,
        candidate_row=(0.902, 0.01, [0.901, 0.902, 0.900, 0.903]),
        baseline_row=(0.900, 0.01, [0.899, 0.900, 0.898, 0.901]),
    )
    assert _submission_gain_significant(conn, "comp", "cand", "base")


def test_regression_metric_sign_flips_direction():
    """rmse(sign=-1)처럼 낮을수록 좋은 메트릭에서 candidate가 baseline보다 낮으면 개선."""
    conn = _conn_for(
        metric_sign=-1,
        candidate_row=(8.753040, 0.020774 ** 2, None),
        baseline_row=(8.755803, 0.019499 ** 2, None),
    )
    # ΔCV=0.13σ 수준(s6e1 07-19 재현) — 유의하지 않아야 함
    assert not _submission_gain_significant(conn, "playground-series-s6e1", "cand", "base")


def test_missing_attempt_data_fails_open():
    """attempt 데이터가 없으면(과거 결손 등) 게이트를 막지 않는다 — #73의 데드락 재발 방지."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = [(1,), None, (0.9, 0.01, None)]
    assert _submission_gain_significant(conn, "comp", "cand", "base")



def _client_for_auto_submit(
    monkeypatch, *, active_competitions, best, last, significant,
    unsubmitted=None, submitted_today=0,
):
    """unsubmitted 기본값 []은 "미제출 백로그 없음" — 예전 best/유의성 경로로 떨어지는 케이스."""
    import bin.api as api_mod

    monkeypatch.setattr(api_mod, "_competition_id_to_slug", lambda: {c: c for c in active_competitions})
    monkeypatch.setattr(api_mod, "_active_competition_ids", lambda: set(active_competitions))
    monkeypatch.setattr(api_mod, "_best_attempt", lambda conn, cid: best)
    monkeypatch.setattr(api_mod, "_last_submitted_attempt", lambda conn, cid: last)
    monkeypatch.setattr(api_mod, "_submissions_today", lambda conn, cid: submitted_today)
    monkeypatch.setattr(
        api_mod, "_unsubmitted_confirmed",
        lambda conn, cid, limit: list(unsubmitted or [])[:limit],
    )
    monkeypatch.setattr(
        api_mod, "_submission_gain_significant", lambda conn, cid, cand, base: significant
    )
    submissions: list[str] = []

    def _fake_start(conn, slug, cid, attempt_id, msg):
        submissions.append(attempt_id)
        return f"sub-{len(submissions)}"

    monkeypatch.setattr(api_mod, "_start_submission", _fake_start)

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [(c,) for c in active_competitions]
    conn.execute.return_value.fetchone.return_value = (None,)  # auto_submit_paused_reason 없음(#104)
    app = create_app(conn, DaemonState())
    return TestClient(app)


def test_auto_submit_skips_when_gain_not_significant(monkeypatch):
    """s6e7 07-25 케이스: best attempt는 바뀌었지만 gain이 fold noise 이하 → skip."""
    client = _client_for_auto_submit(
        monkeypatch,
        active_competitions=["playground-series-s6e7"],
        best=("cand-attempt", 0.949607),
        last="base-attempt",
        significant=False,
    )
    resp = client.post("/api/submissions/auto", json={"window_hours": 24})
    body = resp.json()
    assert body["submitted"] == []
    assert body["skipped"] == [
        {"competition": "playground-series-s6e7", "reason": "gain not significant"}
    ]


def test_auto_submit_submits_when_gain_significant(monkeypatch):
    """gain이 유의하면 이전과 동일하게 제출을 진행한다 — s6e6처럼 실제 개선인 경우 회귀 방지."""
    client = _client_for_auto_submit(
        monkeypatch,
        active_competitions=["playground-series-s6e6"],
        best=("cand-attempt", 0.96423),
        last="base-attempt",
        significant=True,
    )
    resp = client.post("/api/submissions/auto", json={"window_hours": 24})
    body = resp.json()
    assert body["skipped"] == []
    assert len(body["submitted"]) == 1
    assert body["submitted"][0]["attempt_id"] == "cand-attempt"


def test_auto_submit_skips_when_no_confirmed_pipeline(monkeypatch):
    """_best_attempt가 None이면(확정 pipeline 없음, #178) 유의성 검정 없이 skip한다."""
    client = _client_for_auto_submit(
        monkeypatch,
        active_competitions=["playground-series-s6e8"],
        best=None,
        last="base-attempt",
        significant=False,  # 호출되면 안 됨
    )
    resp = client.post("/api/submissions/auto", json={"window_hours": 24})
    body = resp.json()
    assert body["submitted"] == []
    assert body["skipped"] == [
        {"competition": "playground-series-s6e8", "reason": "no confirmed pipeline"}
    ]


def test_auto_submit_first_submission_skips_gain_check(monkeypatch):
    """직전 유효 제출이 없으면(콜드스타트) 유의성 검정 없이 그대로 제출한다."""
    client = _client_for_auto_submit(
        monkeypatch,
        active_competitions=["playground-series-s4e1"],
        best=("cand-attempt", 0.89),
        last=None,
        significant=False,  # 호출되면 안 됨 — 콜드스타트는 게이트 자체를 건너뛴다
    )
    resp = client.post("/api/submissions/auto", json={"window_hours": 24})
    body = resp.json()
    assert body["skipped"] == []
    assert len(body["submitted"]) == 1


def test_auto_submit_prefers_unsubmitted_backlog_over_best(monkeypatch):
    """미제출 confirmed 백로그가 있으면 유의성 게이트와 무관하게 예산만큼 내보낸다 —
    #233의 핵심 동작. 각 미제출 pipeline이 새 cv-LB 쌍을 만든다."""
    client = _client_for_auto_submit(
        monkeypatch,
        active_competitions=["playground-series-s4e10"],
        best=("already-submitted", 0.96),
        last="already-submitted",
        significant=False,  # 백로그 경로는 유의성 게이트를 타지 않는다
        unsubmitted=[("cand-a", 0.9598), ("cand-b", 0.9591), ("cand-c", 0.9585)],
    )
    resp = client.post("/api/submissions/auto", json={})
    body = resp.json()
    assert body["skipped"] == []
    assert [s["attempt_id"] for s in body["submitted"]] == ["cand-a", "cand-b"]


def test_unsubmitted_confirmed_orders_by_prior_snapshot_then_cv():
    """bin/submit.py는 그 attempt 직전 승격분의 materialized_code 스냅샷 위에서 patch를
    실행한다(#80/#89). 스냅샷이 없으면 replay 폴백인데, materialize 합성 규칙이 바뀐 뒤
    (ADR-037/#232) 그 replay는 sha 검증에 대부분 걸린다 — 실측 첫 배치 10건 중 직전
    스냅샷 없는 5건 가운데 4건이 실패했다. 그래서 cv보다 제출 가능성을 먼저 본다."""
    import bin.api as api_mod

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [("a", 0.95, True), ("b", 0.96, False)]
    result = api_mod._unsubmitted_confirmed(conn, "playground-series-s4e10", 2)

    sql = conn.execute.call_args.args[0]
    assert "order by prior_snapshot desc" in sql
    assert result == [("a", 0.95), ("b", 0.96)]


def test_unsubmitted_confirmed_hard_excludes_known_unreproducible_prior():
    """#254 백필이 직전 승격분을 재현 불가로 판정하면(materialized_code IS NULL AND
    materialized_origin IS NOT NULL) 그 attempt는 영영 제출 불가 — 후순위가 아니라
    하드 제외. 아직 백필 안 한 경우(둘 다 NULL)는 후순위로만 남는다."""
    import bin.api as api_mod

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    api_mod._unsubmitted_confirmed(conn, "playground-series-s4e10", 2)
    sql = conn.execute.call_args.args[0]
    assert "materialized_origin is null" in sql
    # 후순위 정렬 키(prior_snapshot)는 그대로 유지 — 미시도 후보를 배제하지 않는다.
    assert "order by prior_snapshot desc" in sql


def test_unsubmitted_confirmed_excludes_candidate_with_own_unverifiable_verdict():
    """후보 자신의 승격 행이 train_drift/eval_error verdict면(materialized_code IS NULL AND
    materialized_origin IS NOT NULL) 제외 — 최초 승격분처럼 직전 행이 없어 대칭 체크에
    안 걸리는 경우를 잡는다(s4e10 05365dfa 실측)."""
    import bin.api as api_mod

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    api_mod._unsubmitted_confirmed(conn, "playground-series-s4e10", 2)
    sql = conn.execute.call_args.args[0]
    assert "p.materialized_code is not null or p.materialized_origin is null" in sql


def test_auto_submit_respects_daily_budget(monkeypatch):
    """오늘 이미 쓴 만큼 예산에서 뺀다 — Kaggle 일일 한도를 넘기지 않기 위함."""
    client = _client_for_auto_submit(
        monkeypatch,
        active_competitions=["playground-series-s4e10"],
        best=("already-submitted", 0.96),
        last="already-submitted",
        significant=False,
        unsubmitted=[("cand-a", 0.9598), ("cand-b", 0.9591)],
        submitted_today=1,
    )
    resp = client.post("/api/submissions/auto", json={})
    body = resp.json()
    assert [s["attempt_id"] for s in body["submitted"]] == ["cand-a"]


def test_auto_submit_skips_when_daily_budget_spent(monkeypatch):
    """예산을 다 썼으면 백로그가 남아 있어도 오늘은 더 안 낸다."""
    client = _client_for_auto_submit(
        monkeypatch,
        active_competitions=["playground-series-s4e10"],
        best=("cand-a", 0.9598),
        last=None,
        significant=True,
        unsubmitted=[("cand-a", 0.9598)],
        submitted_today=SUBMISSIONS_PER_DAY,
    )
    resp = client.post("/api/submissions/auto", json={})
    body = resp.json()
    assert body["submitted"] == []
    assert body["skipped"] == [
        {"competition": "playground-series-s4e10", "reason": "daily budget spent"}
    ]


def test_auto_submit_ignores_inactive_competitions(monkeypatch):
    """동결 대회(ACTIVE=False)는 attempt 이력이 있어도 대상이 아니다 — ADR-032 deep tier."""
    import bin.api as api_mod

    monkeypatch.setattr(
        api_mod, "_competition_id_to_slug",
        lambda: {"playground-series-s4e10": "s4e10", "playground-series-s5e3": "s5e3"},
    )
    monkeypatch.setattr(api_mod, "_active_competition_ids", lambda: {"playground-series-s4e10"})
    monkeypatch.setattr(api_mod, "_submissions_today", lambda conn, cid: 0)
    monkeypatch.setattr(api_mod, "_unsubmitted_confirmed", lambda conn, cid, limit: [("cand-a", 0.9)])
    monkeypatch.setattr(api_mod, "_start_submission", lambda *a, **k: "sub-1")

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (None,)
    client = TestClient(create_app(conn, DaemonState()))

    body = client.post("/api/submissions/auto", json={}).json()
    assert [s["competition"] for s in body["submitted"]] == ["playground-series-s4e10"]
    assert body["skipped"] == []


@pytest.mark.skipif(
    not os.getenv("RONDO_DB_URL"),
    reason="RONDO_DB_URL not set — DB-backed _best_attempt SQL test skipped",
)
def test_best_attempt_join_does_not_raise_ambiguous_column():
    """#221 재현: raw.attempts와 raw.pipelines 둘 다 competition_id를 가져
    join raw.competitions ... using (competition_id)가 psycopg2.errors.AmbiguousColumn으로
    500을 낸다. 이전 테스트는 conn을 MagicMock으로 대체해 SQL을 실행하지 않아 이 버그를
    못 잡았다 — 여기는 실제 Postgres에 SQL을 태워 플랜 단계 실패를 검증한다."""
    from store.db import connect

    conn = connect(apply_schema=True)
    competition_id = f"bon221-test-{uuid.uuid4().hex[:8]}"
    attempt_id = f"{competition_id}-a0"
    try:
        conn.execute(
            "INSERT INTO raw.competitions (competition_id, name, task_type, metric, metric_sign)"
            " VALUES (%s, 'ambiguous-join test', 'binary', 'auc', 1)",
            [competition_id],
        )
        conn.execute(
            "INSERT INTO raw.attempts (attempt_id, competition_id, run_ts, stage, cv_score)"
            " VALUES (%s, %s, now(), 'reflexion', 0.9)",
            [attempt_id, competition_id],
        )
        conn.execute(
            "INSERT INTO raw.pipelines (pipeline_id, attempt_id, competition_id)"
            " VALUES (%s, %s, %s)",
            [f"{attempt_id}-p", attempt_id, competition_id],
        )
        assert _best_attempt(conn, competition_id) == (attempt_id, 0.9)
    finally:
        conn.execute("DELETE FROM raw.pipelines WHERE competition_id = %s", [competition_id])
        conn.execute("DELETE FROM raw.attempts WHERE competition_id = %s", [competition_id])
        conn.execute("DELETE FROM raw.competitions WHERE competition_id = %s", [competition_id])
        conn.close()
