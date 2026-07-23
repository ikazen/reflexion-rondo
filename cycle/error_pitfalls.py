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

_EXCEPTION_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*):\s*(.+)$")


def normalize_error(trace: str) -> str | None:
    for line in reversed(trace.splitlines()):
        line = line.strip()
        if not line:
            continue
        m = _EXCEPTION_LINE.match(line)
        if not m:
            continue
        exc_class = m.group(1).split(".")[-1]
        if exc_class in _SYNTAX_ERRORS:
            return None
        message = m.group(2)
        for pattern in _VOLATILE:
            message = pattern.sub("<val>", message)
        message = re.sub(r"\s+", " ", message).strip()
        return f"{exc_class}: {message}"
    return None


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
