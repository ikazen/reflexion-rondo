from __future__ import annotations

import time

import numpy as np
from ollama import Client

from config import settings
from store.db import PgConn

_EMBED_DIM = 1024
_EMBED_RETRY_DELAYS = (1.0, 4.0, 16.0)
# MMR: λ=0.5 — 관련성/다양성 균형. 높일수록 score 우선, 낮출수록 다양성 우선.
_MMR_LAMBDA = 0.5
# impact 가중 z-score 배율: multiplier = clip(1 + z * W, [0.5, 1.5])
_IMPACT_W = 0.25


class EmbeddingUnavailableError(RuntimeError):
    """임베딩 엔드포인트 일시 장애 — 재시도 소진 후 raise."""


def _client() -> Client:
    return Client(host=settings.OLLAMA_BASE_URL)


def embed(text: str) -> list[float]:
    last_exc: Exception | None = None
    for delay in (0.0, *_EMBED_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            resp = _client().embed(model=settings.MODEL_EMBEDDING, input=text)
            raw = resp.embeddings[0]
            if len(raw) != _EMBED_DIM:
                # 모델 native 차원이 스키마 vector(_EMBED_DIM)와 다를 때 슬라이스.
                # 슬라이싱은 방향 정보를 왜곡하므로 L2 재정규화로 보정한다.
                vec = np.array(raw[:_EMBED_DIM], dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                if norm > 1e-9:
                    vec = vec / norm
                return vec.tolist()
            return raw
        except Exception as exc:
            last_exc = exc
    raise EmbeddingUnavailableError(
        f"embedding failed after {1 + len(_EMBED_RETRY_DELAYS)} attempts: {last_exc}"
    ) from last_exc


def insert_reflection(
    conn: PgConn,
    reflection_id: str,
    attempt_id: str,
    competition_id: str,
    embedded_text: str,
    full_lesson: str,
    generality: str,
    label: str,
    gain_vs_best: float,
    reflector_label: str | None = None,
    lesson_type: str | None = None,
) -> None:
    vec = np.array(embed(embedded_text), dtype=np.float32)
    conn.execute(
        """
        INSERT INTO raw.reflections (
            reflection_id, created_at, attempt_id, competition_id,
            embedded_text, embedding, full_lesson, generality,
            label, reflector_label, lesson_type, gain_vs_best, archived
        ) VALUES (%s, now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, false)
        """,
        [
            reflection_id, attempt_id, competition_id,
            embedded_text, vec, full_lesson, generality,
            label, reflector_label, lesson_type, gain_vs_best,
        ],
    )


def search(
    conn: PgConn,
    query_text: str,
    competition_id: str,
    k: int = 5,
) -> list[dict]:
    query_vec = np.array(embed(query_text), dtype=np.float32)
    rows = conn.execute(
        """
        SELECT
            r.reflection_id,
            r.embedded_text,
            r.full_lesson,
            r.generality,
            r.gain_vs_best,
            r.lesson_type,
            r.embedding,
            1 - (r.embedding <=> %s::vector) AS sim,
            coalesce(i.avg_gain, 0.0)::double precision AS avg_gain
        FROM raw.reflections r
        LEFT JOIN reflection_impact i USING (reflection_id)
        WHERE r.archived = false
          AND (
              r.generality IN ('L2_class', 'L3_general')
              OR r.competition_id = %s
          )
        ORDER BY sim DESC
        LIMIT %s
        """,
        [query_vec, competition_id, k * 4],
    ).fetchall()

    cols = ["reflection_id", "embedded_text", "full_lesson", "generality",
            "gain_vs_best", "lesson_type", "embedding", "sim", "avg_gain"]
    candidates = [dict(zip(cols, row)) for row in rows]
    candidates = _apply_impact_score(candidates)
    selected = _mmr_rerank(candidates, k)
    for item in selected:
        del item["embedding"]
    return selected


def search_failure_lessons(
    conn: PgConn,
    competition_id: str,
    k: int = 3,
) -> list[dict]:
    """최근 실패(lesson_type='failure') 교훈을 코사인 무관하게 최대 k개 반환한다.

    error-fix 교훈은 현재 쿼리(hypothesis)와 의미가 달라 search()의 top-k*4 코사인
    후보에 안 들 수 있음(BON-134) — 이 함수는 임베딩 없이 순수 SQL로 별도 채널을 연다.
    """
    rows = conn.execute(
        """
        SELECT
            r.reflection_id,
            r.embedded_text,
            r.full_lesson,
            r.generality,
            r.gain_vs_best,
            r.lesson_type,
            coalesce(i.avg_gain, 0.0)::double precision AS avg_gain
        FROM raw.reflections r
        LEFT JOIN reflection_impact i USING (reflection_id)
        WHERE r.archived = false
          AND r.lesson_type = 'failure'
          AND (
              r.generality IN ('L2_class', 'L3_general')
              OR r.competition_id = %s
          )
        ORDER BY r.created_at DESC
        LIMIT %s
        """,
        [competition_id, k],
    ).fetchall()

    cols = ["reflection_id", "embedded_text", "full_lesson", "generality",
            "gain_vs_best", "lesson_type", "avg_gain"]
    return [dict(zip(cols, row)) for row in rows]


def _apply_impact_score(candidates: list[dict]) -> list[dict]:
    """avg_gain z-score 표준화 후 sim에 승수 적용. no_op 교훈 추가 감쇠.

    avg_gain 실제 스케일(0.001~0.05)과 sim(0~1)이 달라 단순 clamp 승수로는
    impact 차이가 sim에 묻힘. z-score로 정규화하면 스케일 무관하게 상/하위 교훈 구분 가능.
    """
    if not candidates:
        return candidates
    gains = np.array([c["avg_gain"] for c in candidates], dtype=np.float64)
    std = float(gains.std())
    if std > 1e-9:
        z = (gains - float(gains.mean())) / std
    else:
        z = np.zeros(len(gains))
    for i, c in enumerate(candidates):
        damp = 0.5 if c.get("lesson_type") == "no_op" else 1.0
        mult = float(np.clip(1.0 + z[i] * _IMPACT_W, 0.5, 1.5))
        c["score"] = c["sim"] * mult * damp
    return candidates


def _mmr_rerank(candidates: list[dict], k: int) -> list[dict]:
    """Greedy MMR: λ * score_norm - (1-λ) * max_cos_sim_to_selected.

    score를 min-max 정규화해 redundancy(코사인 0~1)와 같은 스케일로 혼합.
    λ=_MMR_LAMBDA(기본 0.5): 관련성·다양성 균형. 높이면 score 우선.
    """
    if len(candidates) <= k:
        return candidates

    vecs = np.array([c["embedding"] for c in candidates], dtype=np.float32)
    raw_scores = np.array([c["score"] for c in candidates], dtype=np.float32)

    s_min, s_max = float(raw_scores.min()), float(raw_scores.max())
    if s_max - s_min > 1e-9:
        scores = (raw_scores - s_min) / (s_max - s_min)
    else:
        scores = np.ones_like(raw_scores)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs_norm = vecs / norms

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    best = int(np.argmax(scores))
    selected.append(best)
    remaining.remove(best)

    while len(selected) < k and remaining:
        sel_mat = vecs_norm[selected]
        rem_vecs = vecs_norm[remaining]
        redundancy = (rem_vecs @ sel_mat.T).max(axis=1)
        rem_scores = scores[remaining]
        mmr = _MMR_LAMBDA * rem_scores - (1 - _MMR_LAMBDA) * redundancy
        pick = remaining[int(np.argmax(mmr))]
        selected.append(pick)
        remaining.remove(pick)

    return [candidates[i] for i in selected]
