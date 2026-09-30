"""메트릭 레지스트리(get) — auc/rmse/rmsle/accuracy 등, metric_sign 포함.

metric_sign은 gain 계산 시 "개선이면 양수"로 방향을 통일하는 데 쓰인다. float_noise_tolerance는 cv 재현 비교의 허용폭이다.
"""
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error, cohen_kappa_score, balanced_accuracy_score
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error
import numpy as np

_REGISTRY: dict[str, tuple] = {
    "auc":       (roc_auc_score,  +1, "binary_proba"),
    "roc_auc":   (roc_auc_score,  +1, "binary_proba"),
    "logloss":   (log_loss,       -1, "binary_proba"),
    "accuracy":  (accuracy_score, +1, "classification"),
    "f1":        (f1_score,       +1, "classification"),
    "qwk":       (lambda y, p: cohen_kappa_score(y, p, weights="quadratic"), +1, "classification"),
    # balanced_accuracy는 sklearn 표준 함수라 multiclass/binary 둘 다 native 지원
    # (average 파라미터 불필요).
    "balanced_accuracy": (balanced_accuracy_score, +1, "classification"),
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


# 같은 seed·fold를 다시 계산해도 LightGBM 멀티스레드 축약이 프로세스 간 비트 재현이 안 돼 rmse 12.8에서 2e-6이 관측됐다(ADR-062).
# 절대 하한만 두면 큰 스케일 metric에서 정당한 재현을 기각한다.
_FLOAT_NOISE_ABS = 1e-6
_FLOAT_NOISE_REL = 1e-6


def float_noise_tolerance(cv_score: float) -> float:
    return max(_FLOAT_NOISE_ABS, _FLOAT_NOISE_REL * abs(cv_score))
