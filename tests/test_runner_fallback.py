"""runtime.runner._load_best_pipeline의 소스 로드 및 실패 시 fail-hard 동작 단위 테스트."""
from __future__ import annotations

import pytest

from runtime.runner import _load_best_pipeline
from evaluator.harness import BasePipeline, PatchedPipeline


_VALID_SOURCE = """
class Patch:
    action_type = "feature_engineering"
    changed_stages = []
    rationale = "test"

    def feature_transform(self, train, valid, target, ctx):
        return train, valid
"""

_ENSEMBLE_SOURCE = """
class Patch:
    action_type = "ensemble"
    changed_stages = []
    rationale = "test"

    def ensemble_spec(self, ctx):
        return {"members": [{"model": "ridge"}, {"model": "random_forest"}]}
"""

_BROKEN_SOURCE = "class Patch:\n    def broken(self\n"

_NO_PATCH_SOURCE = "x = 1\ndef helper(): pass\n"


def test_valid_source_returns_patched_pipeline():
    pipeline = _load_best_pipeline(_VALID_SOURCE.strip(), BasePipeline, PatchedPipeline)
    assert isinstance(pipeline, PatchedPipeline)
    assert pipeline.feature_transform("train", "valid", "target", None) == ("train", "valid")


def test_ensemble_spec_survives_base_reconstruction():
    """#226 회귀 테스트: 이전엔 훅 이름 고정 목록(_HOOK_NAMES)으로 type(...)을 만들어
    ensemble_spec이 base 재구성 시 조용히 빠졌다 — 승격된 ensemble이 그 다음 사이클부터
    비-ensemble로 퇴화해 prev_best_cv와 실제 재평가가 어긋나는 근본원인이었다.
    PatchedPipeline(BasePipeline(), patch_cls())로 통일한 뒤엔 ensemble_spec을 포함한
    Patch의 모든 속성이 그대로 보존돼야 한다."""
    pipeline = _load_best_pipeline(_ENSEMBLE_SOURCE.strip(), BasePipeline, PatchedPipeline)
    spec = pipeline.ensemble_spec(None)
    assert spec is not None
    assert len(spec["members"]) == 2


def test_broken_source_raises_not_silent_fallback():
    """exec 실패 시 BasePipeline으로 조용히 복귀하지 않고 예외를 전파한다."""
    with pytest.raises(Exception):
        _load_best_pipeline(_BROKEN_SOURCE, BasePipeline, PatchedPipeline)


def test_no_patch_class_raises():
    """Patch 클래스 없는 소스 → RuntimeError (조용한 폴백 없음)."""
    with pytest.raises(RuntimeError, match="no Patch class"):
        _load_best_pipeline(_NO_PATCH_SOURCE, BasePipeline, PatchedPipeline)
