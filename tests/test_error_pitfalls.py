"""cycle.error_pitfalls의 에러 시그니처 정규화(normalize_error)와 pitfall 조회 단위 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from cycle.error_pitfalls import normalize_error, top_error_pitfalls


_CATEGORICAL_TRACE = """\
Traceback (most recent call last):
  File "runner.py", line 42, in run
    model.fit(X, y)
  File "sklearn/base.py", line 123
    return self._fit(X, y)
ValueError: could not convert string to float: 'France'
"""

_CATBOOST_TRACE = """\
Traceback (most recent call last):
  File "runner.py", line 42, in run
    model.fit(X, y)
catboost.CatBoostError: Cat features must be integer or string
"""

_SYNTAX_TRACE = """\
  File "patch.py", line 7
    def feature_transform(self, train valid, target, ctx):
                                       ^
SyntaxError: invalid syntax
"""

_INDENTATION_TRACE = """\
  File "patch.py", line 3
    def foo():
IndentationError: unexpected indent
"""

_CONTRACT_TRACE = "action_type mismatch: expected feature_engineering, got preprocessing"

# #147 실측 — 파이썬 예외 형식(Class: message)이 아닌 실패 3종. 전부 error_trace는
# 있는데 error_signature가 None으로 빠져 error_recurrence가 못 보고 있었다.
_PANDAS_GUARD_TRACE = (
    "pandas-only API (not on polars — use group_by/replace_strict/map_elements/gather/etc): groupby()"
)
_RUNNER_OOM_TRACE = "runner exited without output.json (rc=-9)"
_UNDERSCORE_MODULE_TRACE = """\
Traceback (most recent call last):
  File "runner.py", line 42, in run
    model.fit(X, y)
_catboost.CatBoostError: catboost/private/libs/options/plain_options_helper.cpp:512: Unknown option {random_state} with value "42"
"""


def test_normalize_categorical_to_signature() -> None:
    sig = normalize_error(_CATEGORICAL_TRACE)
    assert sig is not None
    assert "France" not in sig
    assert sig.startswith("ValueError:")
    assert "<val>" in sig


def test_normalize_same_signature_across_values() -> None:
    trace_spain = _CATEGORICAL_TRACE.replace("'France'", "'Spain'")
    trace_germany = _CATEGORICAL_TRACE.replace("'France'", "'Germany'")
    assert normalize_error(_CATEGORICAL_TRACE) == normalize_error(trace_spain)
    assert normalize_error(_CATEGORICAL_TRACE) == normalize_error(trace_germany)


def test_normalize_catboost_error() -> None:
    sig = normalize_error(_CATBOOST_TRACE)
    assert sig is not None
    assert "CatBoostError" in sig


def test_normalize_syntax_error_excluded() -> None:
    assert normalize_error(_SYNTAX_TRACE) is None


def test_normalize_indentation_error_excluded() -> None:
    assert normalize_error(_INDENTATION_TRACE) is None


def test_normalize_non_exception_message_falls_back_to_last_line() -> None:
    """예외 형식(Class: message)이 아닌 한 줄짜리 메시지도(#147) None으로 버리지
    않고 그대로 시그니처로 쓴다 — error_recurrence가 이런 반복 실패도 추적해야
    한다."""
    sig = normalize_error(_CONTRACT_TRACE)
    assert sig == _CONTRACT_TRACE


def test_normalize_pandas_guard_rejection() -> None:
    """실측(#147, s6e6): 정적 가드 거부 메시지가 'Class: message' 형식이 아니라
    시그니처 없이 누락되던 문제 — #136 집계가 이 케이스에 의존한다. API 이름
    (groupby 등)은 그대로 남아야 어떤 API가 재발하는지 구분 가능하다 — 나머지
    file-path 형태 부분만 <val>로 마스킹된다."""
    sig = normalize_error(_PANDAS_GUARD_TRACE)
    assert sig is not None
    assert sig.startswith("pandas-only API")
    assert "groupby()" in sig


def test_normalize_runner_oom_message() -> None:
    """실측(#147, s6e6): runner 비정상 종료(rc=-9)도 예외 형식이 아니라 누락되던
    문제 — #135 OOM 집계가 이 케이스에 의존한다."""
    sig = normalize_error(_RUNNER_OOM_TRACE)
    assert sig is not None
    assert sig.startswith("runner exited without output.json")


def test_normalize_runner_exit_preserves_signal_number() -> None:
    """#159: rc 숫자를 volatile로 마스킹하면 SIGKILL(rc=-9, 과거엔 OOM/CPU kill이
    구분 없이 여기 섞였다)/segfault(rc=-11)/SIGXCPU 백스톱(rc=-24)이 전부 같은
    시그니처로 뭉개져 원인별 집계가 불가능했다. 신호 번호는 그대로 남아야 한다."""
    assert normalize_error("runner exited without output.json (rc=-9)") == \
        "runner exited without output.json (rc=-9)"
    assert normalize_error("runner exited without output.json (rc=-11)") == \
        "runner exited without output.json (rc=-11)"
    assert normalize_error("runner exited without output.json (rc=-24)") == \
        "runner exited without output.json (rc=-24)"
    sig_9 = normalize_error("runner exited without output.json (rc=-9)")
    sig_24 = normalize_error("runner exited without output.json (rc=-24)")
    assert sig_9 != sig_24


def test_normalize_underscore_prefixed_module_matched() -> None:
    """실측(#147, s6e6): `_catboost.CatBoostError`처럼 밑줄로 시작하는 내부
    모듈명이 기존 정규식([A-Za-z]만 허용)에 안 걸려 시그니처 없이 누락되던
    문제."""
    sig = normalize_error(_UNDERSCORE_MODULE_TRACE)
    assert sig is not None
    assert sig.startswith("CatBoostError:")
    assert "512" not in sig
    assert "42" not in sig


def test_normalize_volatile_numbers_removed() -> None:
    trace = "Traceback:\nValueError: expected 3 got 7"
    sig = normalize_error(trace)
    assert sig is not None
    assert "3" not in sig
    assert "7" not in sig


def _make_conn(rows: list[tuple[str]]) -> MagicMock:
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = rows
    return mock_conn


def test_top_error_pitfalls_frequency_and_order() -> None:
    rows = [
        (_CATEGORICAL_TRACE,),
        (_CATEGORICAL_TRACE.replace("France", "Spain"),),
        (_CATEGORICAL_TRACE.replace("France", "Germany"),),
        (_CATBOOST_TRACE,),
        (_CATBOOST_TRACE,),
    ]
    conn = _make_conn(rows)
    pitfalls = top_error_pitfalls(conn, "s4e1", "feature_engineering", min_count=1)
    assert len(pitfalls) >= 2
    sigs = [sig for sig, _ in pitfalls]
    counts = [cnt for _, cnt in pitfalls]
    assert counts == sorted(counts, reverse=True)
    assert any("ValueError" in s for s in sigs)
    assert any("CatBoostError" in s for s in sigs)


def test_top_error_pitfalls_min_count_filter() -> None:
    rows = [(_CATBOOST_TRACE,)]  # count=1 only
    conn = _make_conn(rows)
    pitfalls = top_error_pitfalls(conn, "s4e1", "feature_engineering", min_count=2)
    assert pitfalls == []


def test_top_error_pitfalls_syntax_excluded() -> None:
    rows = [(_SYNTAX_TRACE,), (_SYNTAX_TRACE,), (_SYNTAX_TRACE,)]
    conn = _make_conn(rows)
    pitfalls = top_error_pitfalls(conn, "s4e1", "feature_engineering", min_count=1)
    assert pitfalls == []


def test_top_error_pitfalls_top_k_limit() -> None:
    traces = []
    for i in range(10):
        traces.append((f"Traceback:\nValueError: unique error number {i * 100}\n",) * 1)
    rows = [(f"Traceback:\nValueError: error variant {i}\n",) for i in range(10) for _ in range(3)]
    conn = _make_conn(rows)
    pitfalls = top_error_pitfalls(conn, "s4e1", "feature_engineering", k=3, min_count=1)
    assert len(pitfalls) <= 3


def test_generate_code_known_errors_in_prompt() -> None:
    from unittest.mock import MagicMock, patch
    from agents.coder import generate_code

    mock_resp = MagicMock()
    mock_resp.message.content = "class Patch:\n    action_type = 'feature_engineering'\n"

    with patch("agents.coder._client") as mock_client:
        mock_client.return_value.chat.return_value = mock_resp
        generate_code(
            hypothesis="Add features",
            action_type="feature_engineering",
            eda_card="x",
            known_errors=["ValueError: could not convert string to float: <val> (seen 3x)"],
        )
        messages = mock_client.return_value.chat.call_args.kwargs["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")

    assert "Known failure modes" in user_content
    assert "ValueError" in user_content
    assert "seen 3x" in user_content
