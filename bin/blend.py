"""파이프라인 밖 결정적 blend — 승격된 파이프라인들의 OOF 예측을 Ridge로 조합.

Usage:
    uv run python -m bin.blend --competition s4e1 [--n-top 5]

Reflexion 루프의 귀속 규율은 그대로 유지 — 블렌딩은 파이프라인 밖,
최종 제출용 가중치만 계산한다. submit.py 연결은 범위 밖 — 가중치를
runs/blend/{competition_id}_weights.json에 저장하는 데까지가 이번 범위.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import Ridge

from evaluator.metrics import get as get_metric
from store.db import connect
from store.train_data import load_train

BLEND_DIR = ROOT / "runs" / "blend"
DEFAULT_N_TOP = 5


def fetch_oof_candidates(
    conn, competition_id: str, n_top: int
) -> list[tuple[str, list[float], float]]:
    """cv_score 상위 n_top개의 (pipeline_id, oof_preds, cv_score) 반환.

    "최근 N개"가 아니라 "상위 성능 N개"로 정의한다 — raw.pipelines에 recency
    컬럼(created_at 등)이 없고, 앙상블 다양성 관점에서도 상위 성능이 더 타당하다.
    oof_preds가 없는(OOF 수집 이전에 승격된 것 등) pipeline은 자동 제외된다.
    invalid_reason이 표기된(격리된) pipeline도 제외된다 — 누수 pipeline의
    부풀려진 cv_score가 blend 가중치를 오염시키면 안 된다.
    """
    rows = conn.execute(
        """
        SELECT p.pipeline_id, p.oof_preds, p.cv_score
        FROM raw.pipelines p
        JOIN raw.competitions c USING (competition_id)
        WHERE p.competition_id = %s
          AND p.oof_preds IS NOT NULL
          AND p.invalid_reason IS NULL
        ORDER BY c.metric_sign * p.cv_score DESC
        LIMIT %s
        """,
        [competition_id, n_top],
    ).fetchall()
    result = []
    for pipeline_id, oof_preds, cv_score in rows:
        oof = oof_preds if isinstance(oof_preds, list) else json.loads(oof_preds)
        result.append((pipeline_id, oof, cv_score))
    return result


def build_oof_matrix(
    candidates: list[tuple[str, list[float], float]], n_rows: int
) -> tuple[np.ndarray, list[str]]:
    """OOF 길이가 n_rows와 다른 pipeline(예: 과거 EXTRA_TRAIN_PATHS 변경)은 스킵하고 경고.

    반환: (행렬[n_rows, k], 실제 사용된 pipeline_id 목록).
    """
    cols: list[np.ndarray] = []
    used_ids: list[str] = []
    for pipeline_id, oof, _cv_score in candidates:
        if len(oof) != n_rows:
            print(
                f"[blend] WARNING: pipeline {pipeline_id[:8]} oof 길이({len(oof)}) != "
                f"현재 train 행 수({n_rows}) — 스킵 (EXTRA_TRAIN_PATHS 변경 등 의심)"
            )
            continue
        cols.append(np.asarray(oof, dtype=float))
        used_ids.append(pipeline_id)
    if not cols:
        return np.empty((n_rows, 0)), used_ids
    return np.column_stack(cols), used_ids


def fit_blend(oof_matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """non-negative Ridge로 blend 가중치 학습. (weights, intercept) 반환."""
    model = Ridge(alpha=1.0, positive=True)
    model.fit(oof_matrix, target)
    return model.coef_, float(model.intercept_)


def _store_blend_weights(conn, result: dict) -> None:
    conn.execute(
        """
        INSERT INTO raw.blend_weights
            (competition_id, pipeline_ids, weights, intercept, blend_cv_score, metric, generated_at)
        VALUES (%s, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
        ON CONFLICT (competition_id) DO UPDATE SET
            pipeline_ids   = EXCLUDED.pipeline_ids,
            weights        = EXCLUDED.weights,
            intercept      = EXCLUDED.intercept,
            blend_cv_score = EXCLUDED.blend_cv_score,
            metric         = EXCLUDED.metric,
            generated_at   = EXCLUDED.generated_at
        """,
        [
            result["competition_id"], json.dumps(result["pipeline_ids"]),
            json.dumps(result["weights"]), result["intercept"], result["blend_cv_score"],
            result["metric"], result["generated_at"],
        ],
    )


def compute_and_store_blend(
    conn, competition_id: str, train: "object", target_col: str, metric: str,
    n_top: int = DEFAULT_N_TOP,
) -> dict | None:
    """확정 파이프라인(격리분 제외) 상위 n_top의 OOF로 blend 가중치를 재계산해
    raw.blend_weights에 upsert한다. 승격 시점마다 호출해 자동 최신화한다.

    최소 2개 pipeline이 없거나 OOF 길이가 train과 안 맞는 pipeline뿐이면(과거
    다른 train 크기로 평가된 것 등) 조용히 None을 반환한다 — 승격 흐름 자체를
    막으면 안 되는 best-effort 훅이라 예외를 던지지 않는다. train은 후보들의
    oof_preds가 실제로 어떤 데이터로 생성됐는지에 맞춰 호출부가 골라 넘긴다
    (예: bin/run_promote_task.py는 merge-verify가 쓴 train90).
    """
    candidates = fetch_oof_candidates(conn, competition_id, n_top)
    if len(candidates) < 2:
        return None

    target = train[target_col].to_numpy().astype(float)
    oof_matrix, used_ids = build_oof_matrix(candidates, len(target))
    if oof_matrix.shape[1] < 2:
        return None

    weights, intercept = fit_blend(oof_matrix, target)
    fn, _metric_sign, _metric_class = get_metric(metric)
    blend_preds = oof_matrix @ weights + intercept
    blend_score = float(fn(target, blend_preds))

    result = {
        "competition_id": competition_id,
        "pipeline_ids": used_ids,
        "weights": weights.tolist(),
        "intercept": intercept,
        "blend_cv_score": blend_score,
        "metric": metric,
        "generated_at": datetime.now(timezone.utc),
    }
    _store_blend_weights(conn, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", required=True)
    parser.add_argument("--n-top", type=int, default=DEFAULT_N_TOP)
    args = parser.parse_args()

    comp = importlib.import_module(f"config.competitions.{args.competition}")
    conn = connect(apply_schema=False)
    try:
        train = load_train(comp)
        result = compute_and_store_blend(
            conn, comp.COMPETITION_ID, train, comp.TARGET, comp.METRIC, args.n_top,
        )
    finally:
        conn.close()

    if result is None:
        print(
            f"[blend] {comp.COMPETITION_ID}: 후보 부족(확정 파이프라인 2개 미만 또는 "
            "OOF 길이 불일치) — blend 중단."
        )
        sys.exit(1)

    print(f"[blend] {comp.COMPETITION_ID}: {len(result['pipeline_ids'])}개 pipeline 사용, "
          f"blend_cv_score={result['blend_cv_score']:.6f}")
    for pid, w in zip(result["pipeline_ids"], result["weights"]):
        print(f"  {pid[:8]}: weight={w:.6f}")
    print(f"[blend] weights stored in raw.blend_weights (competition_id={comp.COMPETITION_ID})")

    BLEND_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BLEND_DIR / f"{comp.COMPETITION_ID}_weights.json"
    out_path.write_text(json.dumps({**result, "generated_at": result["generated_at"].isoformat()}, indent=2))
    print(f"[blend] weights also saved locally: {out_path}")


if __name__ == "__main__":
    main()
