"""모델군별 Optuna 검색공간 — evaluator/models.py:MODEL_REGISTRY 키와 1:1 대응한다.

각 공간 함수는 trial.suggest_*의 키를 해당 모델 생성자 kwarg 이름과 그대로 맞춘다 —
evaluator/tuner.py가 optuna.Trial.params(자동 기록)를 그대로 build_registry_model에
넘기므로, 키가 어긋나면 튜닝 결과가 조용히 무시된다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import optuna

SearchSpaceFn = Callable[["optuna.Trial", bool], dict]


def _lgbm_space(trial: "optuna.Trial", is_classification: bool) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # catboost의 verbose=False와 동일한 이유 — 수백 trial 동안 매 iteration 로그가
        # 쌓이면 Airflow task 로그가 압도적으로 커진다.
        "verbosity": -1,
    }


def _xgboost_space(trial: "optuna.Trial", is_classification: bool) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _catboost_space(trial: "optuna.Trial", is_classification: bool) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        # verbose=False가 없으면 iteration마다 학습 로그를 stdout에 쏟아 대량 trial에서
        # 로그가 압도적으로 커진다 — 다른 레지스트리 모델엔 없는 catboost 전용 필요.
        "verbose": False,
    }


def _hgb_space(trial: "optuna.Trial", is_classification: bool) -> dict:
    return {
        "max_iter": trial.suggest_int("max_iter", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
    }


def _tree_ensemble_space(trial: "optuna.Trial", is_classification: bool) -> dict:
    """random_forest/extra_trees 공유 — sklearn 생성자 kwarg가 동일하다."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _ridge_space(trial: "optuna.Trial", is_classification: bool) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}


def _elastic_net_space(trial: "optuna.Trial", is_classification: bool) -> dict:
    # 분류(LogisticRegression)와 회귀(ElasticNet)가 정규화 강도를 반대 방향 파라미터로
    # 받는다 — C(클수록 약한 정규화) vs alpha(클수록 강한 정규화). evaluator/models.py의
    # MODEL_REGISTRY 주석과 동일한 이유로 penalty/solver를 여기서도 명시해야 한다.
    if is_classification:
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "penalty": "elasticnet",
            "solver": "saga",
            "max_iter": 2000,
        }
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


SEARCH_SPACES: dict[str, SearchSpaceFn] = {
    "lgbm": _lgbm_space,
    "xgboost": _xgboost_space,
    "catboost": _catboost_space,
    "hgb": _hgb_space,
    "random_forest": _tree_ensemble_space,
    "extra_trees": _tree_ensemble_space,
    "ridge": _ridge_space,
    "elastic_net": _elastic_net_space,
}


def get_search_space(model_name: str) -> SearchSpaceFn:
    fn = SEARCH_SPACES.get(model_name)
    if fn is None:
        raise ValueError(
            f"tuner: no search space for model {model_name!r} — allowed: {sorted(SEARCH_SPACES)}"
        )
    return fn
