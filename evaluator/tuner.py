"""Optuna 기반 하이퍼파라미터 튜닝 — 확정 pipeline(raw.pipelines)의 model_spec/ensemble_spec
멤버 params를 900s attempt 예산 밖에서 오래(수십~수백 trial) 탐색한다(ADR-035, #230).

preprocess/feature_transform/postprocess_predictions은 확정 pipeline 그대로 두고 모델
생성만 trial마다 바꾼다 — evaluate_pipeline(evaluator/harness.py)을 그대로 재사용해
attempt와 정확히 같은 CV 방법론(is_original 인지 분할, leak 가드 등)으로 측정한다.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import optuna

from evaluator.harness import PipelineContext, evaluate_pipeline
from evaluator.metrics import get as get_metric
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


def tune_confirmed_pipeline(
    pipeline: object,
    train: object,
    ctx: PipelineContext,
    n_trials: int = _DEFAULT_N_TRIALS,
    timeout_sec: int | None = None,
) -> list[TunerResult]:
    """pipeline이 model_spec이면 단일 결과, ensemble_spec이면 멤버별 독립 튜닝 결과
    목록을 반환한다. 둘 다 없으면(자유형 build_model) 튜닝 대상이 없다는 에러."""
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
    raise ValueError(
        "tune_confirmed_pipeline: pipeline declares neither ensemble_spec nor model_spec — "
        "freeform build_model pipelines are not tunable (registry-only, ADR-034/#229와 동일 범위)"
    )
