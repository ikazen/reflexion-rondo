"""memory/retriever.py — _apply_impact_score + _mmr_rerank 단위 테스트.

embed()/search()는 Ollama 연결이 필요하므로 여기서 테스트하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pytest

import memory.retriever as retriever_mod
from memory.retriever import _apply_impact_score, _mmr_rerank


def _vec(seed: int, dim: int = 8) -> list[float]:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32).tolist()


def _c(sim: float, avg_gain: float, lesson_type: str = "normal", seed: int = 0) -> dict:
    return {
        "reflection_id": f"r{seed}",
        "embedded_text": "t",
        "full_lesson": "t",
        "generality": "L1_comp",
        "gain_vs_best": avg_gain,
        "lesson_type": lesson_type,
        "embedding": _vec(seed),
        "sim": sim,
        "avg_gain": avg_gain,
    }


# --- _apply_impact_score ---

def test_impact_same_sim_high_gain_wins():
    """동일 sim, avg_gain 높은 쪽의 final score가 더 크다."""
    hi = _c(sim=0.8, avg_gain=0.05, seed=0)
    lo = _c(sim=0.8, avg_gain=-0.05, seed=1)
    out = _apply_impact_score([hi, lo])
    assert out[0]["score"] > out[1]["score"]


def test_impact_no_op_dampened():
    """no_op 교훈은 동일 sim/avg_gain 대비 score가 절반 이하."""
    normal = _c(sim=0.7, avg_gain=0.0, lesson_type="normal", seed=0)
    noop = _c(sim=0.7, avg_gain=0.0, lesson_type="no_op", seed=1)
    out = _apply_impact_score([normal, noop])
    assert out[0]["score"] > out[1]["score"]
    assert out[1]["score"] == pytest.approx(out[0]["score"] * 0.5, rel=0.01)


def test_impact_zero_std_preserves_sim_order():
    """모든 avg_gain 동일(std=0) → z=0 → multiplier=1 → score 순서=sim 순서."""
    a = _c(sim=0.9, avg_gain=0.01, seed=0)
    b = _c(sim=0.5, avg_gain=0.01, seed=1)
    out = _apply_impact_score([a, b])
    assert out[0]["score"] > out[1]["score"]
    assert abs(out[0]["score"] - 0.9) < 0.01


def test_impact_empty_candidates():
    assert _apply_impact_score([]) == []


def test_impact_score_bounded():
    """multiplier 범위 [0.5, 1.5] 내: score ∈ [sim*0.5, sim*1.5]."""
    candidates = [_c(sim=0.8, avg_gain=float(v), seed=i)
                  for i, v in enumerate([-1.0, -0.1, 0.0, 0.1, 1.0])]
    out = _apply_impact_score(candidates)
    for c in out:
        # no_op 없으므로 damp=1
        assert 0.8 * 0.5 <= c["score"] <= 0.8 * 1.5 + 1e-9


# --- _mmr_rerank ---

def test_mmr_first_is_top_score():
    """첫 번째 선택은 항상 score 최고 후보."""
    rng = np.random.default_rng(0)
    candidates = [
        {"embedding": rng.standard_normal(8).astype(np.float32).tolist(), "score": s}
        for s in [0.9, 0.3, 0.5, 0.7, 0.2]
    ]
    selected = _mmr_rerank(candidates, k=2)
    assert selected[0]["score"] == pytest.approx(0.9)


def test_mmr_returns_k_items():
    rng = np.random.default_rng(1)
    candidates = [
        {"embedding": rng.standard_normal(8).astype(np.float32).tolist(), "score": float(i)}
        for i in range(10)
    ]
    assert len(_mmr_rerank(candidates, k=4)) == 4


def test_mmr_passthrough_when_few():
    candidates = [
        {"embedding": _vec(i), "score": float(i)}
        for i in range(3)
    ]
    result = _mmr_rerank(candidates, k=5)
    assert result == candidates


def test_mmr_diversity_selected_over_redundant():
    """거의 동일한 두 벡터 중 하나보다 다른 방향 벡터가 2번째로 선택된다."""
    v_a = np.ones(8, dtype=np.float32) / np.sqrt(8)
    v_b = (np.ones(8, dtype=np.float32) + np.array([0.02, 0, 0, 0, 0, 0, 0, 0])) / np.sqrt(8)
    v_c = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32) / np.sqrt(8)
    candidates = [
        {"embedding": v_a.tolist(), "score": 0.90},
        {"embedding": v_b.tolist(), "score": 0.85},  # v_a와 거의 동일
        {"embedding": v_c.tolist(), "score": 0.70},  # 다른 방향
    ]
    orig = retriever_mod._MMR_LAMBDA
    try:
        retriever_mod._MMR_LAMBDA = 0.5
        selected = _mmr_rerank([c.copy() for c in candidates], k=2)
    finally:
        retriever_mod._MMR_LAMBDA = orig
    assert selected[1]["score"] == pytest.approx(0.70)


def test_mmr_high_lambda_prefers_score():
    """λ=0.95일 때는 다양성보다 score를 우선해 redundant 고점수 후보를 선택한다."""
    v_a = np.ones(8, dtype=np.float32) / np.sqrt(8)
    v_b = np.ones(8, dtype=np.float32) / np.sqrt(8)  # 완전 동일 벡터
    v_c = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32) / np.sqrt(8)
    candidates = [
        {"embedding": v_a.tolist(), "score": 0.90},
        {"embedding": v_b.tolist(), "score": 0.85},
        {"embedding": v_c.tolist(), "score": 0.70},
    ]
    orig = retriever_mod._MMR_LAMBDA
    try:
        retriever_mod._MMR_LAMBDA = 0.95
        selected = _mmr_rerank([c.copy() for c in candidates], k=2)
    finally:
        retriever_mod._MMR_LAMBDA = orig
    assert selected[1]["score"] == pytest.approx(0.85)


def test_mmr_lambda_monotonicity():
    """λ↑일수록 두 번째 선택의 score 합산이 높다(다양성 희생 → score 우선)."""
    v_a = np.ones(8, dtype=np.float32) / np.sqrt(8)
    v_b = np.ones(8, dtype=np.float32) / np.sqrt(8)
    v_c = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32) / np.sqrt(8)
    candidates = [
        {"embedding": v_a.tolist(), "score": 0.90},
        {"embedding": v_b.tolist(), "score": 0.85},
        {"embedding": v_c.tolist(), "score": 0.70},
    ]
    orig = retriever_mod._MMR_LAMBDA
    results = {}
    try:
        for lam in (0.1, 0.5, 0.9):
            retriever_mod._MMR_LAMBDA = lam
            selected = _mmr_rerank([c.copy() for c in candidates], k=2)
            results[lam] = selected[1]["score"]
    finally:
        retriever_mod._MMR_LAMBDA = orig
    # λ=0.1은 다양성 중시 → 두 번째 score 낮음(0.70), λ=0.9는 score 중시 → 두 번째 score 높음(0.85)
    assert results[0.1] <= results[0.9]
