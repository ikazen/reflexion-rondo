"""Optuna 기반 하이퍼파라미터 튜닝 — 확정 pipeline(raw.pipelines)의 model_spec/ensemble_spec
멤버 params를 900s attempt 예산 밖에서 오래(수십~수백 trial) 탐색한다(ADR-035, #230).

preprocess/feature_transform/postprocess_predictions은 확정 pipeline 그대로 두고 모델
생성만 trial마다 바꾼다 — evaluate_pipeline(evaluator/harness.py)을 그대로 재사용해
attempt와 정확히 같은 CV 방법론(is_original 인지 분할, leak 가드 등)으로 측정한다.
"""
from __future__ import annotations

import ast
import copy
import logging
from dataclasses import dataclass

import optuna

from evaluator.harness import PipelineContext, evaluate_pipeline
from evaluator.metrics import get as get_metric
from evaluator.models import registry_key_for_class
from evaluator.search_spaces import get_search_space

_LOG = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_DEFAULT_N_TRIALS = 100
_SEED = 42


@dataclass
class TunerResult:
    model_name: str
    member_index: int | None  # None=model_spec(단일모델), int=ensemble_spec 멤버 인덱스
    best_params: dict
    best_cv_score: float
    baseline_cv_score: float
    n_trials: int  # 성공적으로 완료된 trial 수 — 예외로 실패한 trial은 제외.
    improved: bool


class _SingleModelTrialPipeline:
    """model_spec 훅만 trial params로 오버라이드하고 나머지는 confirmed pipeline에 위임."""

    def __init__(self, base: object, model_name: str, params: dict) -> None:
        self._base = base
        self._model_name = model_name
        self._params = params

    def preprocess(self, train, valid, target, ctx):
        return self._base.preprocess(train, valid, target, ctx)

    def feature_transform(self, train, valid, target, ctx):
        return self._base.feature_transform(train, valid, target, ctx)

    def postprocess_predictions(self, preds, ctx):
        return self._base.postprocess_predictions(preds, ctx)

    def ensemble_spec(self, ctx):
        return None

    def model_spec(self, ctx):
        return {"model": self._model_name, "params": self._params}


class _EnsembleMemberTrialPipeline:
    """ensemble_spec의 한 멤버만 trial params로 오버라이드, 나머지 멤버는 confirmed
    pipeline이 이미 확정한 값으로 고정."""

    def __init__(self, base: object, base_spec: dict, member_index: int, params: dict) -> None:
        self._base = base
        self._base_spec = base_spec
        self._member_index = member_index
        self._params = params

    def preprocess(self, train, valid, target, ctx):
        return self._base.preprocess(train, valid, target, ctx)

    def feature_transform(self, train, valid, target, ctx):
        return self._base.feature_transform(train, valid, target, ctx)

    def postprocess_predictions(self, preds, ctx):
        return self._base.postprocess_predictions(preds, ctx)

    def ensemble_spec(self, ctx):
        spec = copy.deepcopy(self._base_spec)
        spec["members"][self._member_index]["params"] = self._params
        return spec

    def model_spec(self, ctx):
        return None


def _optimize(objective, n_trials: int, timeout_sec: int | None, direction: str) -> optuna.Study:
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, catch=(Exception,))
    return study


def _direction(ctx: PipelineContext) -> str:
    _, metric_sign, _ = get_metric(ctx.metric)
    return "maximize" if metric_sign > 0 else "minimize"


def tune_single_model(
    pipeline: object,
    train: object,
    ctx: PipelineContext,
    model_name: str,
    n_trials: int = _DEFAULT_N_TRIALS,
    timeout_sec: int | None = None,
) -> TunerResult:
    """pipeline.model_spec(ctx)이 선언한 단일 모델의 params를 탐색한다."""
    # get_search_space를 trial 루프 밖(여기)에서 미리 조회한다 — 등록 안 된 모델명은
    # 설정 오류지 trial 하나하나의 우연한 실패가 아니다. 안에서 조회하면 study.optimize의
    # catch=(Exception,)가 매 trial을 조용히 흡수해 "n_trials개 다 실패, 개선 없음"으로만
    # 보고되고 진짜 원인(오탈자 모델명 등)이 로그에 묻힌다.
    space_fn = get_search_space(model_name)
    baseline_cv = evaluate_pipeline(pipeline, train, ctx).cv_score

    def objective(trial: "optuna.Trial") -> float:
        params = space_fn(trial, ctx.is_classification)
        trial_pipeline = _SingleModelTrialPipeline(pipeline, model_name, params)
        return evaluate_pipeline(trial_pipeline, train, ctx).cv_score

    study = _optimize(objective, n_trials, timeout_sec, _direction(ctx))
    return _to_result(study, model_name, None, baseline_cv, ctx)


def tune_ensemble_member(
    pipeline: object,
    train: object,
    ctx: PipelineContext,
    member_index: int,
    n_trials: int = _DEFAULT_N_TRIALS,
    timeout_sec: int | None = None,
) -> TunerResult:
    """pipeline.ensemble_spec(ctx)의 member_index번째 멤버 params를 탐색한다(다른
    멤버는 confirmed 값에 고정)."""
    base_spec = pipeline.ensemble_spec(ctx)
    if base_spec is None:
        raise ValueError("tune_ensemble_member: pipeline has no ensemble_spec")
    members = base_spec.get("members") or []
    if not (0 <= member_index < len(members)):
        raise ValueError(f"tune_ensemble_member: member_index={member_index} out of range (0..{len(members) - 1})")
    model_name = members[member_index]["model"]
    space_fn = get_search_space(model_name)
    baseline_cv = evaluate_pipeline(pipeline, train, ctx).cv_score

    def objective(trial: "optuna.Trial") -> float:
        params = space_fn(trial, ctx.is_classification)
        trial_pipeline = _EnsembleMemberTrialPipeline(pipeline, base_spec, member_index, params)
        return evaluate_pipeline(trial_pipeline, train, ctx).cv_score

    study = _optimize(objective, n_trials, timeout_sec, _direction(ctx))
    return _to_result(study, model_name, member_index, baseline_cv, ctx)


def _to_result(
    study: "optuna.Study", model_name: str, member_index: int | None, baseline_cv: float, ctx: PipelineContext,
) -> TunerResult:
    _, metric_sign, _ = get_metric(ctx.metric)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        # 전 trial이 예외로 실패(catch=(Exception,))했으면 study.best_value가 없다 —
        # 개선 없음으로 안전하게 보고한다.
        return TunerResult(
            model_name=model_name, member_index=member_index, best_params={},
            best_cv_score=baseline_cv, baseline_cv_score=baseline_cv,
            n_trials=len(completed), improved=False,
        )
    return TunerResult(
        model_name=model_name,
        member_index=member_index,
        best_params=study.best_params,
        best_cv_score=study.best_value,
        baseline_cv_score=baseline_cv,
        n_trials=len(completed),
        improved=metric_sign * (study.best_value - baseline_cv) > 0,
    )


# elastic_net은 추론에서 제외한다 — 자유형 build_model이 LogisticRegression을 쓰면 대개
# 일반 로지스틱회귀지만 elastic_net search space는 penalty="elasticnet"/solver="saga"를
# 강제하므로 confirmed pipeline과 다른 모델을 튜닝하게 된다. 명시적 model_spec으로만 허용.
_INFERABLE_KEYS = frozenset({"lgbm", "xgboost", "catboost", "hgb", "random_forest", "extra_trees", "ridge"})

# 레지스트리에 없지만 "모델처럼 보이는" 생성자 — 있으면 자유형 wrapper로 보고 추론을 포기한다
# (_EnsembleRegressor, EnsembleModel, StackingClassifier, VotingRegressor 등).
_MODEL_LIKE_SUFFIXES = ("Classifier", "Regressor", "Model")


def _call_class_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def infer_registry_model(source: str) -> str | None:
    """자유형 build_model이 "레지스트리 단일 모델 생성자 하나 + params 전달"의 정적으로
    검증 가능한 형태이면 그 레지스트리 키를 반환한다(ADR-035 개정, #252). 아니면 None.

    보수적 통과 조건 — 하나라도 어긋나면 None:
      1. class Patch의 build_model 본문에 레지스트리 생성자 호출이 정확히 1종
      2. "모델처럼 보이는" 미등록 생성자(커스텀 wrapper)가 없음
      3. build_model 2번째 인자(params)가 본문에서 참조됨 — 무시하는 pipeline은 params 치환 무의미
      4. 그 생성자 결과가 return 값(직접 호출 or 그 호출을 담은 지역변수)
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    patch = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Patch"), None
    )
    if patch is None:
        return None
    build_model = next(
        (n for n in patch.body if isinstance(n, ast.FunctionDef) and n.name == "build_model"), None
    )
    if build_model is None:
        return None

    pos_args = build_model.args.args
    params_name = pos_args[1].arg if len(pos_args) >= 2 else None
    if params_name is None:
        return None

    registry_keys: set[str] = set()
    ctor_var_names: set[str] = set()  # 레지스트리 생성자 호출을 담은 지역변수
    params_referenced = False

    for node in ast.walk(build_model):
        if isinstance(node, ast.Name) and node.id == params_name and isinstance(node.ctx, ast.Load):
            params_referenced = True
        if isinstance(node, ast.Call):
            cls_name = _call_class_name(node)
            if cls_name is None:
                continue
            key = registry_key_for_class(cls_name)
            if key is not None:
                registry_keys.add(key)
            elif cls_name.endswith(_MODEL_LIKE_SUFFIXES):
                return None  # 미등록 model-like 생성자 = 커스텀 wrapper

    if len(registry_keys) != 1:
        return None
    key = next(iter(registry_keys))
    if key not in _INFERABLE_KEYS or not params_referenced:
        return None

    for node in ast.walk(build_model):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if registry_key_for_class(_call_class_name(node.value) or "") == key:
                ctor_var_names.update(t.id for t in node.targets if isinstance(t, ast.Name))

    returns = [n for n in ast.walk(build_model) if isinstance(n, ast.Return)]
    if not returns:
        return None
    for ret in returns:
        val = ret.value
        if isinstance(val, ast.Name) and val.id in ctor_var_names:
            continue
        if isinstance(val, ast.Call) and registry_key_for_class(_call_class_name(val) or "") == key:
            continue
        return None
    return key


def tune_confirmed_pipeline(
    pipeline: object,
    train: object,
    ctx: PipelineContext,
    n_trials: int = _DEFAULT_N_TRIALS,
    timeout_sec: int | None = None,
    pipeline_source: str | None = None,
) -> list[TunerResult]:
    """pipeline이 model_spec이면 단일 결과, ensemble_spec이면 멤버별 독립 튜닝 결과
    목록을 반환한다. 둘 다 없으면 pipeline_source에서 레지스트리 모델을 정적 추론하고
    (infer_registry_model), 추론도 실패하면 튜닝 대상이 없다는 에러."""
    ensemble_spec = pipeline.ensemble_spec(ctx)
    if ensemble_spec is not None:
        members = ensemble_spec.get("members") or []
        return [
            tune_ensemble_member(pipeline, train, ctx, i, n_trials=n_trials, timeout_sec=timeout_sec)
            for i in range(len(members))
        ]
    model_spec = pipeline.model_spec(ctx)
    if model_spec is not None:
        return [tune_single_model(pipeline, train, ctx, model_spec["model"], n_trials=n_trials, timeout_sec=timeout_sec)]

    inferred = infer_registry_model(pipeline_source) if pipeline_source else None
    if inferred is not None:
        _LOG.info("tune_confirmed_pipeline: freeform build_model → inferred registry model %r", inferred)
        return [tune_single_model(pipeline, train, ctx, inferred, n_trials=n_trials, timeout_sec=timeout_sec)]

    raise ValueError(
        "tune_confirmed_pipeline: pipeline declares neither ensemble_spec nor model_spec, and "
        "its build_model is not a statically-inferable single registry model — not tunable "
        "(ADR-034/ADR-035, #229/#252)"
    )
