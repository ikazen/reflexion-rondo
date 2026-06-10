from __future__ import annotations

import time

import numpy as np
from ollama import Client

from config import settings
from store.db import PgConn

_EMBED_DIM = 1024
_EMBED_RETRY_DELAYS = (1.0, 4.0, 16.0)
_MMR_LAMBDA = 0.3  # BON-96: 다양성 가중치↑ (coverage 손실 보완)


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
            return resp.embeddings[0][:_EMBED_DIM]
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
            (1 - (r.embedding <=> %s::vector))
                * (1 + greatest(-1.0, least(1.0, coalesce(i.avg_gain, 0.0)::double precision)))
                * CASE WHEN r.lesson_type = 'no_op' THEN 0.5 ELSE 1.0 END AS score
        FROM raw.reflections r
        LEFT JOIN reflection_impact i USING (reflection_id)
        WHERE r.archived = false
          AND (
              r.generality IN ('L2_class', 'L3_general')
              OR r.competition_id = %s
          )
        ORDER BY score DESC
        LIMIT %s
        """,
        [query_vec, query_vec, competition_id, k * 4],  # BON-96: 후보 풀 확장
    ).fetchall()

    cols = ["reflection_id", "embedded_text", "full_lesson", "generality",
            "gain_vs_best", "lesson_type", "embedding", "sim", "score"]
    candidates = [dict(zip(cols, row)) for row in rows]
    selected = _mmr_rerank(candidates, k)
    for item in selected:
        del item["embedding"]
    return selected


def _mmr_rerank(candidates: list[dict], k: int) -> list[dict]:
    """Greedy MMR: λ * score - (1-λ) * max_cos_sim_to_selected."""
    if len(candidates) <= k:
        return candidates

    vecs = np.array([c["embedding"] for c in candidates], dtype=np.float32)
    scores = np.array([c["score"] for c in candidates], dtype=np.float32)
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
