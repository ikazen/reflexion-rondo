"""모델 레지스트리 — harness가 직접 생성자를 호출해 모델을 만드는 유일한 곳.

ensemble_spec(ADR-023)과 model_spec(ADR-034, #229)이 공유한다. Patch(LLM 생성 코드)가
아니라 이 모듈이 생성자를 호출하므로 super() 오용·stale kwarg 문제가 구조적으로
발생하지 않는다 — 그 문제가 애초에 이 모듈이 존재하는 이유다.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluator.harness import PipelineContext

_MAX_BUILD_MODEL_RETRIES = 3
_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument '(\w+)'")
_MULTIPLE_KWARG_RE = re.compile(r"got multiple values for keyword argument '(\w+)'")

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "lgbm":          {"module": "lightgbm",           "classifier": "LGBMClassifier",                "regressor": "LGBMRegressor"},
    "xgboost":       {"module": "xgboost",             "classifier": "XGBClassifier",                  "regressor": "XGBRegressor"},
    "catboost":      {"module": "catboost",             "classifier": "CatBoostClassifier",             "regressor": "CatBoostRegressor"},
    "hgb":           {"module": "sklearn.ensemble",     "classifier": "HistGradientBoostingClassifier", "regressor": "HistGradientBoostingRegressor"},
    "random_forest": {"module": "sklearn.ensemble",     "classifier": "RandomForestClassifier",         "regressor": "RandomForestRegressor"},
    "extra_trees":   {"module": "sklearn.ensemble",     "classifier": "ExtraTreesClassifier",           "regressor": "ExtraTreesRegressor"},
    "ridge":         {"module": "sklearn.linear_model", "classifier": "RidgeClassifier",                "regressor": "Ridge"},
    # elastic_net 분류기는 sklearn에 전용 클래스가 없다 — LogisticRegression을 쓰되
    # 기본 penalty가 l2라 params에 penalty="elasticnet", solver="saga", l1_ratio=<0~1>을
    # 명시해야 실제로 elastic-net 정규화가 걸린다(안 주면 그냥 일반 로지스틱회귀).
    "elastic_net":   {"module": "sklearn.linear_model", "classifier": "LogisticRegression",             "regressor": "ElasticNet"},
}


def construct_with_kwarg_retry(build_fn, params: dict):
    """params로 build_fn(params)를 호출하되, 제거된/불명 kwarg로 인한 TypeError면
    그 키 하나만 벗기고 재시도한다.

    evaluator.harness._build_model_safe(Patch.build_model 경유)와
    build_registry_model(ensemble_spec/model_spec 멤버 생성)이 이 재시도 로직을
    공유한다 — 둘 다 LLM이 stale constructor kwarg(예: LogisticRegression의 구
    `multi_class`)를 params dict에 넣는 동일한 실패 양상을 겪는다.
    """
    current = dict(params)
    last_exc: TypeError | None = None
    for _ in range(_MAX_BUILD_MODEL_RETRIES):
        try:
            return build_fn(current)
        except TypeError as exc:
            msg = str(exc)
            m = _UNEXPECTED_KWARG_RE.search(msg) or _MULTIPLE_KWARG_RE.search(msg)
            if not m or m.group(1) not in current:
                raise
            current = {k: v for k, v in current.items() if k != m.group(1)}
            last_exc = exc
    raise last_exc


_CLASS_NAME_TO_KEY: dict[str, str] = {
    cls_name: key
    for key, entry in MODEL_REGISTRY.items()
    for cls_name in (entry["classifier"], entry["regressor"])
}


def registry_key_for_class(class_name: str) -> str | None:
    """생성자 클래스명(예: "LGBMRegressor")을 MODEL_REGISTRY 키("lgbm")로 되돌린다.
    미등록 클래스명은 None."""
    return _CLASS_NAME_TO_KEY.get(class_name)


def resolve_model_class(model_name: str, is_classification: bool) -> type:
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise ValueError(
            f"unknown registry model {model_name!r} — allowed: {sorted(MODEL_REGISTRY)}"
        )
    import importlib
    module = importlib.import_module(entry["module"])
    cls_name = entry["classifier"] if is_classification else entry["regressor"]
    return getattr(module, cls_name)


def build_registry_model(model_name: str, params: dict, ctx: "PipelineContext") -> object:
    """레지스트리 기반으로 모델을 직접 생성한다 — LLM이 작성한 코드가 아니라
    harness 자신이 생성자를 호출하므로 super() 오용·stale kwarg 문제가 구조적으로
    발생하지 않는다. random_state는 params가 명시하지 않으면 ctx.seed로 채운다.
    ensemble_spec 멤버와 model_spec 단일 모델 양쪽이 이 함수를 공유한다.

    catboost는 random_state/random_seed를 동의어로 취급해 둘 다 세팅되면
    "only one of the parameters random_seed, random_state should be initialized"로
    fit() 시점에 죽는다(construct_with_kwarg_retry가 잡는 __init__ TypeError가 아니라
    별개의 CatBoostError라 재시도 대상도 아님, 실측 확인) — params가 random_seed를
    이미 명시했으면 random_state를 추가로 채우지 않는다."""
    cls = resolve_model_class(model_name, ctx.is_classification)
    base_params = dict(params or {})
    if "random_state" not in base_params and "random_seed" not in base_params:
        base_params["random_state"] = ctx.seed
    return construct_with_kwarg_retry(lambda p: cls(**p), base_params)
