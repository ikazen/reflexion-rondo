from __future__ import annotations

import pytest

from runtime.runner import _load_best_pipeline_class
from evaluator.harness import BasePipeline


_VALID_SOURCE = """
class Patch:
    action_type = "feature_engineering"
    changed_stages = []
    rationale = "test"

    def feature_transform(self, train, valid, target, ctx):
        return train, valid
"""

_BROKEN_SOURCE = "class Patch:\n    def broken(self\n"

_NO_PATCH_SOURCE = "x = 1\ndef helper(): pass\n"


def test_valid_source_returns_merged_class():
    cls = _load_best_pipeline_class(_VALID_SOURCE.strip(), BasePipeline)
    assert cls is not BasePipeline
    assert issubclass(cls, BasePipeline)
    assert hasattr(cls, "feature_transform")


def test_broken_source_raises_not_silent_fallback():
    """exec 실패 시 BasePipeline으로 조용히 복귀하지 않고 예외를 전파한다."""
    with pytest.raises(Exception):
        _load_best_pipeline_class(_BROKEN_SOURCE, BasePipeline)


def test_no_patch_class_raises():
    """Patch 클래스 없는 소스 → RuntimeError (조용한 폴백 없음)."""
    with pytest.raises(RuntimeError, match="no Patch class"):
        _load_best_pipeline_class(_NO_PATCH_SOURCE, BasePipeline)
