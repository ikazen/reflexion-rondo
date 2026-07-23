import numpy as np
import pytest
from sklearn.metrics import root_mean_squared_error

from evaluator.metrics import get, _REGISTRY

_Y_CLS = np.array([0, 0, 1, 1, 1])
_P_PROBA = np.array([0.1, 0.3, 0.6, 0.7, 0.9])
_P_CLS = np.array([0, 0, 1, 1, 1])

_Y_REG = np.array([1.0, 2.0, 3.0, 4.0])
_P_REG = np.array([1.1, 1.9, 3.1, 3.8])

_Y_QWK = np.array([0, 1, 2, 3, 2])
_P_QWK = np.array([0, 1, 2, 2, 3])

_CASES = [
    ("auc",      _Y_CLS, _P_PROBA),
    ("roc_auc",  _Y_CLS, _P_PROBA),
    ("logloss",  _Y_CLS, _P_PROBA),
    ("accuracy", _Y_CLS, _P_CLS),
    ("f1",       _Y_CLS, _P_CLS),
    ("qwk",      _Y_QWK, _P_QWK),
    ("balanced_accuracy", _Y_QWK, _P_QWK),  # multiclass 대상, _Y_QWK가 3-class 이상 라벨 포함
    ("rmse",     _Y_REG, _P_REG),
    ("mae",      _Y_REG, _P_REG),
    ("rmsle",    _Y_REG, _P_REG),
]


@pytest.mark.parametrize("name,y,p", _CASES)
def test_metric_scores_without_error(name, y, p):
    fn, sign, _ = get(name)
    score = fn(y, p)
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_rmse_matches_sklearn():
    fn, _, _ = get("rmse")
    assert abs(fn(_Y_REG, _P_REG) - root_mean_squared_error(_Y_REG, _P_REG)) < 1e-9


def test_get_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        get("nonexistent_metric")


def test_qwk_sign_and_class():
    _, sign, metric_class = get("qwk")
    assert sign == +1
    assert metric_class == "classification"


def test_balanced_accuracy_sign_and_class():
    _, sign, metric_class = get("balanced_accuracy")
    assert sign == +1
    assert metric_class == "classification"


def test_balanced_accuracy_matches_sklearn():
    from sklearn.metrics import balanced_accuracy_score
    fn, _, _ = get("balanced_accuracy")
    assert abs(fn(_Y_QWK, _P_QWK) - balanced_accuracy_score(_Y_QWK, _P_QWK)) < 1e-9


def test_transfer_metric_class_keys_covered():
    from memory.transfer import _METRIC_CLASS
    missing = set(_METRIC_CLASS) - set(_REGISTRY)
    assert not missing, f"transfer._METRIC_CLASS has keys missing from metrics._REGISTRY: {missing}"
