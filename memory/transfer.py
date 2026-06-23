"""Cross-competition transfer: 유사 대회 검색 + cold-start 교훈/시드 추출."""
from __future__ import annotations

import json
import math

from store.db import PgConn

_METRIC_CLASS: dict[str, str] = {
    "auc": "binary_proba", "logloss": "binary_proba",
    "rmse": "regression_error", "mae": "regression_error", "rmsle": "regression_error",
    "accuracy": "classification", "f1": "classification",
    "qwk": "classification",
}
_SIZE_RANK: dict[str, int] = {"tiny": 0, "small": 1, "mid": 2, "large": 3}


def _fp_distance(a: dict, b: dict) -> float:
    dist = 0.0

    if a.get("task_type") != b.get("task_type"):
        dist += 100.0

    mc_a = _METRIC_CLASS.get(a.get("metric", ""), "unknown")
    mc_b = _METRIC_CLASS.get(b.get("metric", ""), "unknown")
    if mc_a != mc_b:
        dist += 50.0

    rank_a = _SIZE_RANK.get(a.get("size_class", "mid"), 2)
    rank_b = _SIZE_RANK.get(b.get("size_class", "mid"), 2)
    dist += abs(rank_a - rank_b) * 20.0

    n_a = (a.get("n_numeric", 0) + a.get("n_categorical", 0)) or 1
    n_b = (b.get("n_numeric", 0) + b.get("n_categorical", 0)) or 1
    dist += abs(a.get("n_numeric", 0) / n_a - b.get("n_numeric", 0) / n_b) * 10.0
    dist += abs(a.get("n_categorical", 0) / n_a - b.get("n_categorical", 0) / n_b) * 10.0

    dist += abs(a.get("missing_ratio_overall", 0.0) - b.get("missing_ratio_overall", 0.0)) * 5.0

    log_card_a = math.log(a.get("cardinality_max", 0) + 1)
    log_card_b = math.log(b.get("cardinality_max", 0) + 1)
    dist += abs(log_card_a - log_card_b) * 2.0

    dist += abs(a.get("target_stat", 0.0) - b.get("target_stat", 0.0)) * 2.0

    return dist


def find_similar_competitions(
    conn: PgConn,
    fp_new: dict,
    exclude_id: str,
    k: int = 3,
) -> list[tuple[str, float]]:
    """fingerprint 가중 거리로 유사 대회 top-k 반환. [(competition_id, dist), ...]"""
    rows = conn.execute(
        "select competition_id, fingerprint from raw.competitions where competition_id != %s",
        [exclude_id],
    ).fetchall()

    scored: list[tuple[str, float]] = []
    for comp_id, fp_json in rows:
        fp = json.loads(fp_json) if isinstance(fp_json, str) else (fp_json or {})
        if not fp:
            continue
        dist = _fp_distance(fp_new, fp)
        scored.append((comp_id, dist))

    scored.sort(key=lambda x: x[1])
    return scored[:k]


def cold_start_lessons(
    conn: PgConn,
    similar: list[str],
    k: int = 10,
) -> list[dict]:
    """유사 대회의 L2_class 교훈 + 전체 L3_general 교훈 (reflection_impact 재순위)."""
    rows = conn.execute(
        """
        select
            r.reflection_id,
            r.embedded_text,
            r.full_lesson,
            r.generality,
            r.gain_vs_best,
            coalesce(i.avg_gain, 0.0) as avg_gain,
            coalesce(i.avg_gain, 0.0) as score
        from raw.reflections r
        left join reflection_impact i using (reflection_id)
        where r.archived = false
          and (
              (r.competition_id = any(%s::text[]) and r.generality = 'L2_class')
              or r.generality = 'L3_general'
          )
        order by score desc
        limit %s
        """,
        [similar, k],
    ).fetchall()

    cols = ["reflection_id", "embedded_text", "full_lesson", "generality", "gain_vs_best", "avg_gain", "score"]
    return [dict(zip(cols, row)) for row in rows]


def bootstrap_seeds(
    conn: PgConn,
    similar: list[str],
    n: int = 2,
) -> list[dict]:
    """유사 대회의 gain_vs_best > 0 파이프라인 (cv_score 내림차순). [{pipeline_id, code, cv_score}, ...]"""
    if not similar:
        return []

    rows = conn.execute(
        """
        select pipeline_id, code, cv_score, competition_id
        from raw.pipelines
        where competition_id = any(%s::text[])
          and gain_vs_best > 0
        order by cv_score desc
        limit %s
        """,
        [similar, n],
    ).fetchall()

    return [
        {"pipeline_id": r[0], "code": r[1], "cv_score": r[2], "competition_id": r[3]}
        for r in rows
    ]
