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


def test_normalize_contract_violation_excluded() -> None:
    assert normalize_error(_CONTRACT_TRACE) is None


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
