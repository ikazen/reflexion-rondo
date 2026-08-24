"""evaluator.search_spaces — MODEL_REGISTRY와의 커버리지 및 실제 생성자 호환성 검증."""
from __future__ import annotations

import optuna
import pytest

from evaluator.harness import PipelineContext
from evaluator.models import MODEL_REGISTRY, build_registry_model
from evaluator.search_spaces import SEARCH_SPACES, get_search_space


def _ctx(is_classification: bool) -> PipelineContext:
    return PipelineContext(
        target_col="y", metric="auc" if is_classification else "mae",
        n_splits=3, seed=42, is_classification=is_classification,
    )


def test_search_spaces_cover_full_model_registry():
    assert set(SEARCH_SPACES) == set(MODEL_REGISTRY)


def test_get_search_space_unknown_raises():
    with pytest.raises(ValueError, match="no search space"):
        get_search_space("not_a_model")


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY))
@pytest.mark.parametrize("is_classification", [True, False])
def test_search_space_params_construct_registry_model(model_name, is_classification):
    """suggest_*의 키가 실제 생성자 kwarg와 어긋나면 build_registry_model이 TypeError로
    죽는다 — 어긋남을 조용히 넘기지 않고 여기서 바로 잡는다."""
    study = optuna.create_study()
    trial = study.ask()
    params = get_search_space(model_name)(trial, is_classification)
    model = build_registry_model(model_name, params, _ctx(is_classification))
    assert model is not None
