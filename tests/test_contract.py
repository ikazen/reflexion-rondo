"""evaluator.contract.validate_patch 정적 가드(pandas-only API, 무제한 병렬성, undefined-name 등) 단위 테스트."""
from __future__ import annotations

import pytest

from evaluator.contract import validate_patch

_VALID_FEATURE_ENG = """
import polars as pl

class Patch:
    action_type = "feature_engineering"
    changed_stages = ["feature_transform"]
    rationale = "Add interaction feature."

    def feature_transform(self, train, valid, target, ctx):
        cols = [c for c in train.columns if c != target]
        return train.select(cols), valid.select(cols)
""".strip()


def test_valid_patch_returns_no_errors():
    assert validate_patch(_VALID_FEATURE_ENG, "feature_engineering") == []


def test_parse_level_syntax_error_caught():
    source = "def f(:\n    pass"
    errs = validate_patch(source, "feature_engineering")
    assert len(errs) == 1
    assert errs[0].startswith("SyntaxError")


def test_return_outside_function_caught():
    """ast.parse passes this; compile() must catch it."""
    source = "x = 1\nreturn x"
    errs = validate_patch(source, "feature_engineering")
    assert len(errs) == 1
    assert errs[0].startswith("SyntaxError")


def test_break_outside_loop_caught():
    """ast.parse passes this; compile() must catch it."""
    source = _VALID_FEATURE_ENG + "\nbreak\n"
    errs = validate_patch(source, "feature_engineering")
    assert len(errs) == 1
    assert errs[0].startswith("SyntaxError")


def test_duplicate_arg_caught():
    """ast.parse passes duplicate args; compile() catches them."""
    source = "class Patch:\n    action_type = 'feature_engineering'\n    def feature_transform(self, a, a, b, c):\n        pass\n"
    errs = validate_patch(source, "feature_engineering")
    assert any(e.startswith("SyntaxError") for e in errs)


def test_forbidden_import_still_caught():
    source = "import os\n" + _VALID_FEATURE_ENG
    errs = validate_patch(source, "feature_engineering")
    assert any("forbidden import" in e for e in errs)


def test_action_type_mismatch_still_caught():
    errs = validate_patch(_VALID_FEATURE_ENG, "model_swap")
    assert any("action_type" in e for e in errs)


def test_pandas_import_forbidden():
    source = "import pandas as pd\n" + _VALID_FEATURE_ENG
    errs = validate_patch(source, "feature_engineering")
    assert any("forbidden import" in e and "pandas" in e for e in errs)


def test_importlib_import_now_forbidden():
    """importlib.import_module('os')로 _FORBIDDEN_IMPORTS 우회하던 경로를 막는다."""
    source = "import importlib\n" + _VALID_FEATURE_ENG
    errs = validate_patch(source, "feature_engineering")
    assert any("forbidden import" in e and "importlib" in e for e in errs)


def test_getattr_string_concat_bypass_not_caught():
    """soft guard 한계 문서화: _collect_calls는 ast.Name/Attribute만 보므로
    getattr(...)(...) 호출 형태는 forbidden call로 잡히지 않는다. 진짜 경계는 실행
    샌드박스이며, 이 테스트는 lint의 한계를 회귀 기준으로 고정한다."""
    source = (
        'class Patch:\n'
        '    action_type = "feature_engineering"\n'
        '    changed_stages = ["feature_transform"]\n'
        '    rationale = "bypass demo"\n'
        '    def feature_transform(self, train, valid, target, ctx):\n'
        '        getattr(__builtins__, "ope" + "n")("/etc/passwd")\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "feature_engineering")
    assert not any("forbidden call" in e for e in errs)


def test_dunder_subclass_chain_bypass_not_caught():
    """soft guard 한계 문서화: dunder 체인으로 임의 클래스 접근은
    이름 기반 검사로 탐지되지 않는다. eval/exec/open 등 직접 호출만 잡힘."""
    source = (
        'class Patch:\n'
        '    action_type = "feature_engineering"\n'
        '    changed_stages = ["feature_transform"]\n'
        '    rationale = "bypass demo"\n'
        '    def feature_transform(self, train, valid, target, ctx):\n'
        '        subclasses = ().__class__.__bases__[0].__subclasses__()\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "feature_engineering")
    assert not any("forbidden call" in e or "forbidden import" in e for e in errs)


def test_allowed_hooks_covers_all_action_types():
    from config.settings import ACTION_TYPES
    from evaluator.contract import _ALLOWED_HOOKS
    # Every emittable action_type must have a contract entry.
    # bootstrap is valid but not bandit-emitted (special stage).
    non_bandit = {"bootstrap"}
    bandit_keys = set(_ALLOWED_HOOKS) - non_bandit
    assert bandit_keys == set(ACTION_TYPES), (
        f"_ALLOWED_HOOKS bandit keys {bandit_keys} != ACTION_TYPES {set(ACTION_TYPES)}"
    )



@pytest.mark.parametrize("attr", [
    "groupby", "map_dict", "take", "apply", "iterrows", "applymap", "get_dummies",
])
def test_pandas_only_attr_forbidden(attr: str):
    """polars 1.41.2에 hasattr로 직접 확인한 순수 pandas 관용구 — 실제 최근 3일 AttributeError
    상위 원인. DataFrame/Series 어디에도 없어 오탐 없이 금지 가능하다."""
    source = (
        'class Patch:\n'
        '    action_type = "feature_engineering"\n'
        '    changed_stages = ["feature_transform"]\n'
        '    rationale = "pandas confusion demo"\n'
        '    def feature_transform(self, train, valid, target, ctx):\n'
        f'        train.{attr}("x")\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "feature_engineering")
    assert any("pandas-only API" in e and attr in e for e in errs)


def test_value_counts_not_forbidden():
    """value_counts는 polars Series에 실존한다(DataFrame에는 없음) — 금지 목록에서
    의도적으로 제외했으므로 오탐이 없어야 한다(회귀 고정)."""
    source = (
        'class Patch:\n'
        '    action_type = "feature_engineering"\n'
        '    changed_stages = ["feature_transform"]\n'
        '    rationale = "value_counts is legit on Series"\n'
        '    def feature_transform(self, train, valid, target, ctx):\n'
        '        counts = train["target"].value_counts()\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "feature_engineering")
    assert not any("pandas-only API" in e for e in errs)



def _build_model_patch(model_call: str) -> str:
    return (
        'class Patch:\n'
        '    action_type = "model_swap"\n'
        '    changed_stages = ["build_model"]\n'
        '    rationale = "unbounded parallelism demo"\n'
        '    def build_model(self, params, ctx):\n'
        f'        return {model_call}\n'
    )


@pytest.mark.parametrize("call,hit", [
    ('LGBMClassifier(n_jobs=-1)', "n_jobs=-1"),
    ('CatBoostClassifier(thread_count=-1)', "thread_count=-1"),
    ('XGBClassifier(n_jobs=-1)', "n_jobs=-1"),
    ('RandomForestClassifier(n_jobs=-1)', "n_jobs=-1"),
    ('LGBMClassifier(num_threads=0)', "num_threads=0"),
    ('XGBClassifier(nthread=-1)', "nthread=-1"),
    ('SomeModel(n_threads=-2)', "n_threads=-2"),
])
def test_unbounded_parallelism_literal_rejected(call: str, hit: str):
    """실측(#162): OMP_NUM_THREADS=2를 걸어도 n_jobs=-1/thread_count=-1류는
    무시하고 자체 스레드풀로 15배 이상 코어를 점유한다(LightGBM 20 threads,
    CatBoost 21 threads, sklearn RandomForest 43 threads 실측). 0 이하 리터럴은
    전부 "가용 코어 전부/대부분" 요청으로 간주해 거부한다."""
    errs = validate_patch(_build_model_patch(call), "model_swap")
    assert any("unbounded parallelism" in e and hit in e for e in errs)


@pytest.mark.parametrize("call", [
    "LGBMClassifier(n_jobs=2)",
    "CatBoostClassifier(thread_count=1)",
    "RandomForestClassifier()",
    "Ridge(alpha=1.0)",
])
def test_bounded_or_unrelated_params_not_forbidden(call: str):
    """양의 고정값(스레드 제한 준수)이나 threading과 무관한 파라미터는 거부하지
    않는다 — 정당한 모델 구성까지 막으면 재생성 왕복만 늘어난다."""
    errs = validate_patch(_build_model_patch(call), "model_swap")
    assert not any("unbounded parallelism" in e for e in errs)


def test_unbounded_parallelism_via_variable_not_flagged():
    """값이 변수/표현식이면 정적으로 판정 불가 — 과소탐지를 오탐보다 우선한다
    (파일 상단 docstring: name 기반 soft guard, 실제 경계는 실행 샌드박스)."""
    source = _build_model_patch("LGBMClassifier(n_jobs=params.get('n_jobs', 2))")
    errs = validate_patch(source, "model_swap")
    assert not any("unbounded parallelism" in e for e in errs)



def test_undefined_name_in_hook_caught():
    """hook 안에서 자기 소스 어디에도 정의되지 않은 이름을 참조하면 에러.

    실제 사고(WeightedEnsemble)와 같은 클래스의 버그를 candidate patch 자신이
    저지른 경우(예: ensemble action이 helper 클래스를 참조만 하고 정의를 빼먹음)를
    재현한다."""
    source = (
        'class Patch:\n'
        '    action_type = "model_swap"\n'
        '    changed_stages = ["build_model"]\n'
        '    rationale = "forgot to define helper"\n'
        '    def build_model(self, params, ctx):\n'
        '        return WeightedEnsemble(params)\n'
    )
    errs = validate_patch(source, "model_swap")
    assert any("undefined name" in e and "WeightedEnsemble" in e for e in errs)


def test_undefined_name_resolved_via_import_or_toplevel_helper_ok():
    """import된 이름이나 같은 소스의 top-level helper로 해석되면 오탐이 없어야 한다."""
    source = (
        'from sklearn.ensemble import RandomForestClassifier\n'
        '\n'
        'def _make_params():\n'
        '    return {"n_estimators": 100}\n'
        '\n'
        'class Patch:\n'
        '    action_type = "model_swap"\n'
        '    changed_stages = ["build_model"]\n'
        '    rationale = "uses import + top-level helper"\n'
        '    def build_model(self, params, ctx):\n'
        '        return RandomForestClassifier(**_make_params())\n'
    )
    errs = validate_patch(source, "model_swap")
    assert not any("undefined name" in e for e in errs)


# preprocess valid-target 직접 참조 정적 가드 (#97, GH #96)
# s5e10 승격 패턴(valid[target]로 quantile bin 생성)을 재생성 왕복 전에 값싸게
# 미리 걸러낸다. 본체는 evaluator.harness._check_preprocess_target_leak(런타임
# 동등성 검사) — 이건 흔한 패턴에 대한 보조 lint.

def test_preprocess_subscript_valid_target_read_caught():
    source = (
        'class Patch:\n'
        '    action_type = "preprocessing"\n'
        '    changed_stages = ["preprocess"]\n'
        '    rationale = "quantile bin from valid target"\n'
        '    def preprocess(self, train, valid, target, ctx):\n'
        '        y_valid = valid[target]\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "preprocessing")
    assert any("target leakage" in e for e in errs)


def test_preprocess_get_column_valid_target_read_caught():
    source = (
        'class Patch:\n'
        '    action_type = "preprocessing"\n'
        '    changed_stages = ["preprocess"]\n'
        '    rationale = "quantile bin from valid target via get_column"\n'
        '    def preprocess(self, train, valid, target, ctx):\n'
        '        y_valid = valid.get_column(target)\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "preprocessing")
    assert any("target leakage" in e for e in errs)


def test_preprocess_train_target_read_not_flagged():
    """train[target] 참조는 정상 — preprocess가 train 통계를 쓰는 것은 흔한 정당한 용도."""
    source = (
        'class Patch:\n'
        '    action_type = "preprocessing"\n'
        '    changed_stages = ["preprocess"]\n'
        '    rationale = "bin edges from train target only"\n'
        '    def preprocess(self, train, valid, target, ctx):\n'
        '        y_train = train[target]\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "preprocessing")
    assert not any("target leakage" in e for e in errs)


def test_preprocess_valid_index_by_other_column_not_flagged():
    """valid[some_other_col] 참조는 target과 무관하므로 오탐이 없어야 한다."""
    source = (
        'class Patch:\n'
        '    action_type = "preprocessing"\n'
        '    changed_stages = ["preprocess"]\n'
        '    rationale = "unrelated column access"\n'
        '    def preprocess(self, train, valid, target, ctx):\n'
        '        geo = valid["geo"]\n'
        '        return train, valid\n'
    )
    errs = validate_patch(source, "preprocessing")
    assert not any("target leakage" in e for e in errs)


def test_star_import_skips_undefined_name_check():
    """star import가 있으면 무엇이 바인딩되는지 알 수 없어 미탐지를 택한다
    (cycle/materialize.py의 동일 원칙과 일치, 오탐 방지 우선)."""
    source = (
        'from sklearn.ensemble import *\n'
        '\n'
        'class Patch:\n'
        '    action_type = "model_swap"\n'
        '    changed_stages = ["build_model"]\n'
        '    rationale = "star import hides bindings"\n'
        '    def build_model(self, params, ctx):\n'
        '        return SomeUnknownEstimator(params)\n'
    )
    errs = validate_patch(source, "model_swap")
    assert not any("undefined name" in e for e in errs)


# ensemble_spec 훅 (#74)

_VALID_ENSEMBLE_SPEC = """
class Patch:
    action_type = "ensemble"
    changed_stages = ["ensemble_spec"]
    rationale = "declarative ensemble of lgbm + xgboost"

    def ensemble_spec(self, ctx):
        return {"members": [{"model": "lgbm"}, {"model": "xgboost"}]}
""".strip()


def test_ensemble_spec_allowed_for_ensemble_action_type():
    assert validate_patch(_VALID_ENSEMBLE_SPEC, "ensemble") == []


def test_ensemble_spec_allowed_for_bootstrap_action_type():
    source = _VALID_ENSEMBLE_SPEC.replace('action_type = "ensemble"', 'action_type = "bootstrap"')
    assert validate_patch(source, "bootstrap") == []


def test_ensemble_spec_allowed_for_model_swap_after_hook_restriction_removed():
    """ADR-006 뒤집기(ADR-037, #232) — action_type별 훅 1개 제한이 하드 리젝트가 아니라
    프롬프트 가이드로만 남아, model_swap도 정적 검증 레벨에서는 ensemble_spec을 구현할 수
    있다(PatchedPipeline.ensemble_spec의 런타임 상속 억제 로직은 그대로 유지됨, #239)."""
    source = _VALID_ENSEMBLE_SPEC.replace('action_type = "ensemble"', 'action_type = "model_swap"')
    assert validate_patch(source, "model_swap") == []


def test_ensemble_spec_wrong_arity_caught():
    source = (
        'class Patch:\n'
        '    action_type = "ensemble"\n'
        '    changed_stages = ["ensemble_spec"]\n'
        '    rationale = "wrong arity"\n'
        '    def ensemble_spec(self, ctx, extra_arg):\n'
        '        return {"members": []}\n'
    )
    errs = validate_patch(source, "ensemble")
    assert any("ensemble_spec" in e and "expected 2 args" in e for e in errs)


# model_spec (ADR-034, #229) — ensemble_spec의 단일 모델 버전. ensemble_spec 회귀
# 세트를 그대로 미러링해 두 훅이 계약 레벨에서 대칭으로 취급되는지 확인한다.

_VALID_MODEL_SPEC = """
class Patch:
    action_type = "model_swap"
    changed_stages = ["model_spec"]
    rationale = "declarative single model swap to catboost"

    def model_spec(self, ctx):
        return {"model": "catboost", "params": {"iterations": 500}}
""".strip()


def test_model_spec_allowed_for_model_swap_action_type():
    assert validate_patch(_VALID_MODEL_SPEC, "model_swap") == []


def test_model_spec_allowed_for_ensemble_action_type():
    source = _VALID_MODEL_SPEC.replace('action_type = "model_swap"', 'action_type = "ensemble"')
    assert validate_patch(source, "ensemble") == []


def test_model_spec_allowed_for_hyperparam_search_after_hook_restriction_removed():
    """ADR-006 뒤집기(ADR-037, #232) — 위 ensemble_spec 케이스와 대칭."""
    source = _VALID_MODEL_SPEC.replace('action_type = "model_swap"', 'action_type = "hyperparam_search"')
    assert validate_patch(source, "hyperparam_search") == []


def test_model_spec_wrong_arity_caught():
    source = (
        'class Patch:\n'
        '    action_type = "model_swap"\n'
        '    changed_stages = ["model_spec"]\n'
        '    rationale = "wrong arity"\n'
        '    def model_spec(self, ctx, extra_arg):\n'
        '        return {"model": "lgbm"}\n'
    )
    errs = validate_patch(source, "model_swap")
    assert any("model_spec" in e and "expected 2 args" in e for e in errs)
