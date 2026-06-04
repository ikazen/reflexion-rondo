from __future__ import annotations

import time

import duckdb
import numpy as np
from ollama import Client

from config import settings

_EMBED_DIM = 1024
_EMBED_RETRY_DELAYS = (1.0, 4.0, 16.0)  # 3회, exponential backoff
_MMR_LAMBDA = 0.5  # relevance vs. diversity 균형


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
    conn: duckdb.DuckDBPyConnection,
    reflection_id: str,
    attempt_id: str,
    competition_id: str,
    embedded_text: str,
    full_lesson: str,
    generality: str,
    label: str,
    gain_vs_best: float,
    reflector_label: str | None = None,
) -> None:
    vec = embed(embedded_text)
    conn.execute(
        """
        insert into raw.reflections (
            reflection_id, created_at, attempt_id, competition_id,
            embedded_text, embedding, full_lesson, generality,
            label, reflector_label, gain_vs_best, archived
        ) values (?, now(), ?, ?, ?, ?, ?, ?, ?, ?, ?, false)
        """,
        [
            reflection_id, attempt_id, competition_id,
            embedded_text, vec, full_lesson, generality,
            label, reflector_label, gain_vs_best,
        ],
    )


def search(
    conn: duckdb.DuckDBPyConnection,
    query_text: str,
    competition_id: str,
    k: int = 5,
) -> list[dict]:
    query_vec = embed(query_text)
    # MMR을 위해 2k 후보를 가져온 뒤 Python에서 greedy 재순위
    rows = conn.execute(
        """
        select
            r.reflection_id,
            r.embedded_text,
            r.full_lesson,
            r.generality,
            r.gain_vs_best,
            r.embedding,
            array_cosine_similarity(r.embedding, $query_vec::float[1024]) as sim,
            sim * (1 + greatest(-1.0, least(1.0, coalesce(i.avg_gain, 0.0)))) as score
        from raw.reflections r
        left join reflection_impact i using (reflection_id)
        where r.archived = false
          and (
              r.generality in ('L2_class', 'L3_general')
              or r.competition_id = $competition_id
          )
        order by score desc
        limit $candidate_k
        """,
        {"query_vec": query_vec, "competition_id": competition_id, "candidate_k": k * 2},
    ).fetchall()

    cols = ["reflection_id", "embedded_text", "full_lesson", "generality",
            "gain_vs_best", "embedding", "sim", "score"]
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

    # 첫 번째: score 최고 항목
    best = int(np.argmax(scores))
    selected.append(best)
    remaining.remove(best)

    while len(selected) < k and remaining:
        sel_mat = vecs_norm[selected]  # (n_sel, dim)
        rem_vecs = vecs_norm[remaining]  # (n_rem, dim)
        # 이미 선택된 문서와의 최대 코사인 유사도
        redundancy = (rem_vecs @ sel_mat.T).max(axis=1)  # (n_rem,)
        rem_scores = scores[remaining]
        mmr = _MMR_LAMBDA * rem_scores - (1 - _MMR_LAMBDA) * redundancy
        pick = remaining[int(np.argmax(mmr))]
        selected.append(pick)
        remaining.remove(pick)

    return [candidates[i] for i in selected]
