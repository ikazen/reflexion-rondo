from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.coder import generate_code, _extract_code


def _mock_resp(content: str) -> MagicMock:
    m = MagicMock()
    m.message.content = content
    return m


_VALID_PATCH = """
import polars as pl

class Patch:
    action_type = "feature_engineering"
    changed_stages = ["feature"]
    rationale = "Drop low-variance columns."

    def feature_transform(self, train, valid, target, ctx):
        cols = [c for c in train.columns if c != target]
        return train.select(cols), valid.select(cols)
""".strip()


def test_extract_code_from_markdown() -> None:
    text = f"Here is the code:\n```python\n{_VALID_PATCH}\n```\nDone."
    assert _extract_code(text) == _VALID_PATCH


def test_extract_code_plain() -> None:
    assert _extract_code(_VALID_PATCH) == _VALID_PATCH


def test_extract_code_picks_class_patch_block() -> None:
    example_block = "x = 1\ny = 2"
    text = (
        f"Example:\n```python\n{example_block}\n```\n"
        f"Real code:\n```python\n{_VALID_PATCH}\n```"
    )
    assert _extract_code(text) == _VALID_PATCH


def test_extract_code_falls_back_to_first_block_when_no_patch() -> None:
    first = "x = 1"
    second = "y = 2"
    text = f"```python\n{first}\n```\n```python\n{second}\n```"
    assert _extract_code(text) == first


def test_generate_code_returns_string() -> None:
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_PATCH)
        result = generate_code(
            hypothesis="Drop low-variance columns",
            action_type="feature_engineering",
            eda_card="n_rows=165034, task=binary",
        )
    assert "Patch" in result
    assert "feature_transform" in result


def test_generate_code_with_error_feedback() -> None:
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_PATCH)
        generate_code(
            hypothesis="Fix missing class",
            action_type="feature_engineering",
            eda_card="x",
            error_feedback="missing class definition: Patch",
        )
        messages = mock_client.return_value.chat.call_args.kwargs["messages"]
    # system message = static contract; user message = dynamic context + error_feedback
    system_msg = next(m for m in messages if m["role"] == "system")
    user_msg   = next(m for m in messages if m["role"] == "user")
    assert "Patch" in system_msg["content"]
    assert "missing class definition" in user_msg["content"]


def test_generate_code_system_message_is_contract() -> None:
    """정적 contract가 system 메시지, 동적 컨텍스트는 user 메시지여야 한다."""
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_PATCH)
        generate_code(
            hypothesis="Drop low-variance columns",
            action_type="feature_engineering",
            eda_card="n_rows=100",
        )
        messages = mock_client.return_value.chat.call_args.kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    # EDA Card는 user 메시지에 있어야 함
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "n_rows=100" in user_content


def test_generate_code_reflexion_contract_declares_available_libs() -> None:
    """reflexion contract가 실제 설치된 라이브러리를 명시해야 한다."""
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_PATCH)
        generate_code(
            hypothesis="Swap estimator",
            action_type="model_swap",
            eda_card="n_rows=100",
        )
        messages = mock_client.return_value.chat.call_args.kwargs["messages"]
    system_msg = next(m["content"] for m in messages if m["role"] == "system")
    assert "lightgbm" in system_msg
    assert "NOT available: tabpfn" in system_msg
    assert "pandas" in system_msg


def test_generate_code_bootstrap_contract_declares_available_libs() -> None:
    """bootstrap contract도 동일하게 라이브러리 가용 목록을 명시해야 한다."""
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_PATCH)
        generate_code(
            hypothesis="Bootstrap baseline",
            action_type="bootstrap",
            eda_card="n_rows=100",
        )
        messages = mock_client.return_value.chat.call_args.kwargs["messages"]
    system_msg = next(m["content"] for m in messages if m["role"] == "system")
    assert "lightgbm" in system_msg
    assert "NOT available: tabpfn" in system_msg


# --- action_type별 허용 hook 동적 강조 (생성 이전 가드) ---

def test_generate_code_injects_action_type_specific_hook_directive() -> None:
    """user 메시지에 이번 호출의 action_type이 허용하는 hook만 명시돼야 한다.

    s6e7 실측: model_swap이 feature_transform까지 구현하려는 컨트랙트 위반이 47건 —
    정적 검증(evaluator/contract.py)은 생성 *이후*에만 잡아 재시도해도 반복됐다.
    """
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_PATCH)
        generate_code(
            hypothesis="Swap estimator",
            action_type="model_swap",
            eda_card="n_rows=100",
        )
        messages = mock_client.return_value.chat.call_args.kwargs["messages"]
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "model_swap" in user_msg
    assert "build_model" in user_msg
    # 다른 action_type의 hook(feature_transform)이 "허용" 문구에 나타나면 안 된다 —
    # 정확히 evaluator/contract.py._ALLOWED_HOOKS['model_swap']과 일치해야 함.
    allowed_line = next(line for line in user_msg.splitlines() if "You may implement ONLY" in line)
    assert "feature_transform" not in allowed_line


def test_generate_code_hook_directive_matches_contract_source_of_truth() -> None:
    """agents/coder.py가 evaluator/contract.py._ALLOWED_HOOKS를 직접 import해 쓰는지 —
    두 곳에 같은 매핑을 중복 정의하면 드리프트가 생긴다."""
    from evaluator.contract import _ALLOWED_HOOKS as contract_allowed_hooks
    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = _mock_resp(_VALID_PATCH)
        generate_code(
            hypothesis="Hyperparameter search",
            action_type="hyperparam_search",
            eda_card="n_rows=100",
        )
        messages = mock_client.return_value.chat.call_args.kwargs["messages"]
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    expected = sorted(contract_allowed_hooks["hyperparam_search"])
    assert str(expected) in user_msg


# --- multiclass 라벨 왕복(round-trip) 가드 문구 ---

def test_reflexion_contract_warns_about_multiclass_label_roundtrip() -> None:
    """s6e7 실측(45건): 타깃을 정수로 인코딩해놓고 postprocess에서 원복 안 해
    `ValueError: Mix of label input types`로 크래시하는 패턴 — contract에 명시돼야 한다."""
    from agents.coder import _REFLEXION_CONTRACT, _BOOTSTRAP_CONTRACT
    assert "Mix of label input types" in _REFLEXION_CONTRACT
    assert "Mix of label input types" in _BOOTSTRAP_CONTRACT


# --- #74: s6e7 ensemble action_type 70% 크래시 가드 문구 ---

def test_contracts_replace_strict_example_has_default() -> None:
    """replace_strict 예시 자체에 default= 가 없으면 Coder가 그대로 베껴 unseen
    카테고리/null 있는 fold에서 `incomplete mapping` 크래시가 난다(#74 실측
    11건). 예시 코드 자체를 고쳐야 재발이 막힌다."""
    from agents.coder import _REFLEXION_CONTRACT, _BOOTSTRAP_CONTRACT
    assert "replace_strict(mapping, default=" in _REFLEXION_CONTRACT
    assert "replace_strict(mapping, default=" in _BOOTSTRAP_CONTRACT


def test_contracts_warn_xgboost_needs_encoded_labels() -> None:
    """XGBoost의 sklearn wrapper는 문자열 라벨을 못 받는데(#74 실측 12건,
    `ValueError: Invalid classes inferred`) 기존 contract는 "모든 라이브러리가
    문자열을 받는다"고 잘못 안내하고 있었다 — XGBoost 예외를 명시해야 한다.
    기존 라벨 왕복 경고 문구는 회귀 없이 유지돼야 한다."""
    from agents.coder import _REFLEXION_CONTRACT, _BOOTSTRAP_CONTRACT
    for contract in (_REFLEXION_CONTRACT, _BOOTSTRAP_CONTRACT):
        assert "XGBoost" in contract
        assert "Invalid classes inferred" in contract
        assert "Mix of label input types" in contract


def test_contracts_warn_against_explicit_super_call() -> None:
    """Coder가 생성하는 wrapper 클래스가 명시적 super(ClassName, self)를 쓰면
    harness의 동적 클래스 로딩과 충돌해 크래시한다(#74 실측 3건,
    `TypeError: super(type, obj): obj must be an instance or subtype of type`)."""
    from agents.coder import _REFLEXION_CONTRACT, _BOOTSTRAP_CONTRACT
    assert "super()" in _REFLEXION_CONTRACT
    assert "super()" in _BOOTSTRAP_CONTRACT


def test_contracts_warn_estimator_receives_numpy_not_dataframe() -> None:
    """harness는 model.fit()/predict()를 항상 numpy 배열로 호출하는데 Coder가
    생성한 ensemble wrapper가 DataFrame을 가정하고 .to_numpy()를 호출해
    크래시하는 패턴(#74 실측 15건, `AttributeError: 'numpy.ndarray' object has
    no attribute 'to_numpy'`)."""
    from agents.coder import _REFLEXION_CONTRACT, _BOOTSTRAP_CONTRACT
    assert "to_numpy" in _REFLEXION_CONTRACT
    assert "to_numpy" in _BOOTSTRAP_CONTRACT


def test_contracts_ordinal_encoding_example_drops_nulls_before_sorting() -> None:
    """v1.4.3 배포 검증 중 실측: replace_strict에 default=를 추가했지만 그 앞의
    sorted(train[col].unique().to_list())는 여전히 null-unsafe했다 — null이
    섞인 unique() 결과를 sorted()에 넘기면 `TypeError: '<' not supported
    between instances of 'NoneType' and 'str'`로 크래시한다(#74 후속). 정석
    예시 자체가 drop_nulls()를 거쳐야 재발이 막힌다."""
    from agents.coder import _REFLEXION_CONTRACT, _BOOTSTRAP_CONTRACT
    assert "drop_nulls().unique()" in _REFLEXION_CONTRACT
    assert "drop_nulls().unique()" in _BOOTSTRAP_CONTRACT


def test_contract_prefers_ensemble_spec_over_wrapper_class() -> None:
    """#74 — 자유형 ensemble wrapper 클래스가 반복적으로 겪은 버그(super() 오용,
    stale kwarg, to_numpy 등)는 전부 harness가 볼 수 없는 exec된 클래스 몸체
    안에서 발생한다. ensemble_spec 선언형 대안을 prompt가 강하게 권고해야
    Coder가 애초에 그 클래스를 덜 쓰게 유도된다."""
    from agents.coder import _REFLEXION_CONTRACT
    assert "ensemble_spec" in _REFLEXION_CONTRACT
    assert "STRONGLY PREFERRED" in _REFLEXION_CONTRACT or "strongly preferred" in _REFLEXION_CONTRACT.lower()
    for model_name in ("lgbm", "xgboost", "catboost", "hgb", "random_forest", "ridge"):
        assert model_name in _REFLEXION_CONTRACT
    assert "weighted_average" in _REFLEXION_CONTRACT
    assert "majority_vote" in _REFLEXION_CONTRACT
