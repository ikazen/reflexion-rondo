"""evaluator.search_spaces — MODEL_REGISTRY와의 커버리지 및 실제 생성자 호환성 검증."""
from __future__ import annotations

import numpy as np
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


def test_lgbm_search_space_bagging_freq_makes_subsample_effective():
    """#393: LightGBM은 bagging_freq=0(기본값)이면 subsample 값과 무관하게 bagging이
    비활성화돼 이 탐색 차원이 죽어있었다. 탐색공간 출력에 bagging_freq가 포함돼야
    subsample을 바꿨을 때 실제 예측도 달라진다(수정 전엔 동일했음)."""
    study = optuna.create_study()
    trial = study.ask()
    params = get_search_space("lgbm")(trial, is_classification=False)
    assert params.get("bagging_freq") == 1

    rng = np.random.default_rng(0)
    n = 300
    X = rng.standard_normal((n, 5))
    y = X[:, 0] * 2.0 + X[:, 1] - X[:, 2] + rng.standard_normal(n) * 0.5
    ctx = _ctx(is_classification=False)

    low = build_registry_model("lgbm", {**params, "subsample": 0.5, "n_estimators": 50}, ctx)
    high = build_registry_model("lgbm", {**params, "subsample": 1.0, "n_estimators": 50}, ctx)
    low.fit(X, y)
    high.fit(X, y)
    assert not np.allclose(low.predict(X), high.predict(X))


def test_lgbm_search_space_covers_the_dimensions_the_s5e8_base_uses():
    """#488: 확정 pipeline이 쓰는 max_bin/scale_pos_weight가 공간에 없어 23회 튜닝이 같은 점으로 수렴했다. 분류에서만 scale_pos_weight를 낸다."""
    trial = optuna.create_study().ask()
    clf = get_search_space("lgbm")(trial, is_classification=True)
    reg = get_search_space("lgbm")(optuna.create_study().ask(), is_classification=False)

    assert 255 <= clf["max_bin"] <= 4095
    assert 0.5 <= clf["scale_pos_weight"] <= 10.0
    assert "max_bin" in reg
    assert "scale_pos_weight" not in reg

