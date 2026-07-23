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
    """
    rows = conn.execute(
        """
        SELECT p.pipeline_id, p.oof_preds, p.cv_score
        FROM raw.pipelines p
        JOIN raw.competitions c USING (competition_id)
        WHERE p.competition_id = %s
          AND p.oof_preds IS NOT NULL
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", required=True)
    parser.add_argument("--n-top", type=int, default=DEFAULT_N_TOP)
    args = parser.parse_args()

    comp = importlib.import_module(f"config.competitions.{args.competition}")
    conn = connect(apply_schema=False)

    candidates = fetch_oof_candidates(conn, comp.COMPETITION_ID, args.n_top)
    conn.close()
    if len(candidates) < 2:
        print(
            f"[blend] {comp.COMPETITION_ID}: oof_preds 보유 pipeline이 {len(candidates)}개뿐 "
            "— blend에는 최소 2개 필요. 중단."
        )
        sys.exit(1)

    train = load_train(comp)
    target = train[comp.TARGET].to_numpy().astype(float)

    oof_matrix, used_ids = build_oof_matrix(candidates, len(target))
    if oof_matrix.shape[1] < 2:
        print(f"[blend] 길이 일치하는 pipeline이 {oof_matrix.shape[1]}개뿐 — blend 중단.")
        sys.exit(1)

    weights, intercept = fit_blend(oof_matrix, target)

    fn, metric_sign, metric_class = get_metric(comp.METRIC)
    blend_preds = oof_matrix @ weights + intercept
    blend_score = float(fn(target, blend_preds))

    print(f"[blend] {comp.COMPETITION_ID}: {len(used_ids)}개 pipeline 사용, "
          f"blend_cv_score={blend_score:.6f}")
    for pid, w in zip(used_ids, weights):
        print(f"  {pid[:8]}: weight={w:.6f}")

    BLEND_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BLEND_DIR / f"{comp.COMPETITION_ID}_weights.json"
    out_path.write_text(json.dumps({
        "competition_id": comp.COMPETITION_ID,
        "pipeline_ids": used_ids,
        "weights": weights.tolist(),
        "intercept": intercept,
        "blend_cv_score": blend_score,
        "metric": comp.METRIC,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"[blend] weights saved: {out_path}")


if __name__ == "__main__":
    main()
