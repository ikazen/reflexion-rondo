"""에러 시그니처 정규화(normalize_error) + 과거 pitfall 조회(top_error_pitfalls).

동일 에러의 값만 다른 변형(예: 다른 컬럼명)을 같은 signature로 묶어 재발 빈도를 센다.
"""
from __future__ import annotations

import re
from collections import Counter

from store.db import PgConn

_SYNTAX_ERRORS = {"SyntaxError", "IndentationError", "TabError"}

_VOLATILE = [
    re.compile(r"'[^']*'"),          # 'France', 'foo/bar', any string literal
    re.compile(r'"[^"]*"'),          # "double-quoted literals"
    re.compile(r"\b0x[0-9a-fA-F]+\b"),  # hex addresses
    re.compile(r"\b\d+\.\d+\b"),     # floats
    re.compile(r"\b\d+\b"),          # ints
    re.compile(r"(/[\w./\\-]+)"),     # file paths
]


# 모듈명이 밑줄로 시작하는 내부 확장 모듈(_catboost, _pickle 등)도 실제
# traceback에 나온다 — [A-Za-z]만 허용하면 이런 예외 라인이 통째로 안 걸린다
# (#147, s6e6 실측: _catboost.CatBoostError 라인이 매칭 실패해 시그니처 없이
# 누락됨).
_EXCEPTION_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):\s*(.+)$")

# _VOLATILE의 \d+ 스크럽이 "rc=-9"(SIGKILL)/"rc=-11"(segfault)/"rc=-24"(SIGXCPU
# 백스톱)를 전부 "rc=-<val>"로 합쳐버려, 커널이 어떤 이유로 죽였는지가 시그니처
# 단계에서 사라졌다(#159 — CPU 예산 초과 kill이 이 시그니처로 뭉개져 OOM으로
# 오판된 근본원인 중 하나). eval_isolated의 자체 워치독(RSS/CPU)이 먼저 걸러낸
# 뒤라 이 경로는 이제 드물어야 하지만, 신호 번호는 그대로 살려야 향후 재발 시
# 원인별로 집계·구분할 수 있다.
_RUNNER_EXIT_LINE = re.compile(r"^runner exited without output\.json \(rc=(-?\d+)\)$")


def _normalize_message(text: str) -> str:
    for pattern in _VOLATILE:
        text = pattern.sub("<val>", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_error(trace: str) -> str | None:
    """실패 attempt를 error_recurrence가 추적할 수 있는 시그니처로 정규화한다.

    'ExceptionClass: message' 형식 라인을 아래에서 위로 찾아 우선 매칭한다.
    그런 라인이 하나도 없으면(#147 실측: 정적 가드 거부 메시지 —
    "pandas-only API (...): groupby()", runner 비정상 종료 —
    "runner exited without output.json (rc=-9)" 등, 파이썬 예외가 아니라
    harness/runtime이 직접 만든 상태 메시지) 마지막 비어있지 않은 줄을 그대로
    정규화해 시그니처로 쓴다 — 이런 것도 실제로는 반복되는 실패 패턴이라
    추적 안 하면 #135/#136류 집계가 과소 추정된다.
    """
    last_nonblank: str | None = None
    for line in reversed(trace.splitlines()):
        line = line.strip()
        if not line:
            continue
        if last_nonblank is None:
            last_nonblank = line
        m = _EXCEPTION_LINE.match(line)
        if not m:
            continue
        exc_class = m.group(1).split(".")[-1]
        if exc_class in _SYNTAX_ERRORS:
            return None
        return f"{exc_class}: {_normalize_message(m.group(2))}"

    if last_nonblank is None:
        return None
    m = _RUNNER_EXIT_LINE.match(last_nonblank)
    if m:
        return f"runner exited without output.json (rc={m.group(1)})"
    return _normalize_message(last_nonblank) or None


def top_error_pitfalls(
    conn: PgConn,
    competition_id: str,
    action_type: str,
    *,
    k: int = 5,
    window: int = 100,
    min_count: int = 2,
) -> list[tuple[str, int]]:
    rows = conn.execute(
        """
        SELECT error_trace FROM raw.attempts
        WHERE competition_id = %s
          AND action_type = %s
          AND error_trace IS NOT NULL
        ORDER BY run_ts DESC
        LIMIT %s
        """,
        [competition_id, action_type, window],
    ).fetchall()

    counter: Counter[str] = Counter()
    for (trace,) in rows:
        sig = normalize_error(trace)
        if sig:
            counter[sig] += 1

    return [(sig, cnt) for sig, cnt in counter.most_common(k) if cnt >= min_count]
