from __future__ import annotations

import duckdb
from ollama import Client

from config import settings

_EMBED_DIM = 1024


def _client() -> Client:
    kwargs: dict = {"host": settings.OLLAMA_BASE_URL}
    if settings.OLLAMA_API_KEY:
        kwargs["headers"] = {"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"}
    return Client(**kwargs)


def embed(text: str) -> list[float]:
    resp = _client().embed(model=settings.MODEL_EMBEDDING, input=text)
    return resp.embeddings[0]


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
    rows = conn.execute(
        """
        select
            r.reflection_id,
            r.embedded_text,
            r.full_lesson,
            r.generality,
            r.gain_vs_best,
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
        limit $k
        """,
        {"query_vec": query_vec, "competition_id": competition_id, "k": k},
    ).fetchall()

    cols = ["reflection_id", "embedded_text", "full_lesson", "generality", "gain_vs_best", "sim", "score"]
    return [dict(zip(cols, row)) for row in rows]
