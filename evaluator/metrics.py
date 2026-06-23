from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error
import numpy as np

_REGISTRY: dict[str, tuple] = {
    "auc":       (roc_auc_score,  +1, "binary_proba"),
    "roc_auc":   (roc_auc_score,  +1, "binary_proba"),
    "logloss":   (log_loss,       -1, "binary_proba"),
    "accuracy":  (accuracy_score, +1, "classification"),
    "f1":        (f1_score,       +1, "classification"),
    "qwk":       (lambda y, p: cohen_kappa_score(y, p, weights="quadratic"), +1, "classification"),
    "rmse":      (root_mean_squared_error, -1, "regression_error"),
    "mae":       (mean_absolute_error, -1, "regression_error"),
    "rmsle":     (lambda y, p: root_mean_squared_log_error(y, np.clip(p, 0, None)), -1, "regression_error"),
}


def get(metric: str) -> tuple:
    """Returns (callable, metric_sign, metric_class). Raises if unknown."""
    key = metric.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown metric '{metric}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[key]
