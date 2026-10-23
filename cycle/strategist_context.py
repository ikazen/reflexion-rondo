"""Strategist EDA 카드에 붙는 base 요약과 "이 base에서 이미 시도한 것" 블록(#532).

Strategist는 base 코드도 시도 이력도 못 봐서 같은 가설이 반복된다. 동일 gain = 동일 계산이므로 (action_type, gain)으로 묶어 반복을 드러낸다.
"""
from __future__ import annotations

import ast
import re

from store.db import PgConn

_TRIED_LIMIT = 24
_HYPOTHESIS_CHARS = 120
_COLUMNS_LIMIT = 60
_PARAMS_CHARS = 400
_WEIGHTING = re.compile(r"\b(sample_weight|class_weight|auto_class_weights|scale_pos_weight)\b")
_ALIAS = re.compile(r"""\.alias\(\s*['"]([^'"]+)['"]\s*\)""")
_MODEL = re.compile(
    r"\b(?:XGB|LGBM|CatBoost|HistGradientBoosting|RandomForest|ExtraTrees)\w*?(?:Classifier|Regressor)\b|\bLogisticRegression\b"
)


def tried_on_base_lines(conn: PgConn, competition_id: str) -> list[str]:
    """마지막 유효 확정 pipeline 이후 에러 아닌 attempt를 (action_type, gain)으로 묶어 반복이 많은 순으로 돌려준다."""
    # 확정 pipeline이 없으면 서브쿼리가 NULL이라 결과가 비고 블록이 안 붙는다(부트스트랩). gain 7자리 반올림은 비트 동일한 재평가를
    # 묶는다 — 연속 지표의 재현 노이즈(1e-7~2e-6, ADR-062)로 갈라진 같은 계산은 못 묶는다.
    rows = conn.execute(
        """
        select action_type, round(gain_vs_best::numeric, 7), count(*),
               (array_agg(hypothesis order by run_ts desc))[1]
        from raw.attempts
        where competition_id = %s
          and error_trace is null
          and gain_vs_best is not null
          and run_ts > (
              select max(a.run_ts)
              from raw.pipelines p
              join raw.attempts a using (attempt_id)
              where p.competition_id = %s and p.cv_score is not null and p.invalid_reason is null
          )
        group by 1, 2
        order by 3 desc, max(run_ts) desc
        limit %s
        """,
        [competition_id, competition_id, _TRIED_LIMIT],
    ).fetchall()
    if not rows:
        return []
    header = (
        "\n## Already tried on this base (one line per distinct outcome — "
        "do not propose these again, pick something materially different)"
    )
    lines = [header]
    for action_type, gain, cnt, hypothesis in rows:
        text = " ".join((hypothesis or "").split())[:_HYPOTHESIS_CHARS]
        lines.append(f"- {action_type} x{cnt} gain={float(gain):+.6f}: {text}")
    return lines


def _param_candidates_literal(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "param_candidates":
            for node in ast.walk(fn):
                if isinstance(node, ast.Return) and node.value is not None:
                    try:
                        return repr(ast.literal_eval(node.value))[:_PARAMS_CHARS]
                    except ValueError:
                        return None
    return None


def base_summary_lines(prev_code: str | None) -> list[str]:
    """확정 base 코드에서 모델 클래스, `.alias(...)` 컬럼명, param_candidates 리터럴만 뽑는다. 코드 전문은 코더만 본다."""
    if not prev_code:
        return []
    models = list(dict.fromkeys(_MODEL.findall(prev_code)))
    columns = list(dict.fromkeys(_ALIAS.findall(prev_code)))[:_COLUMNS_LIMIT]
    params = _param_candidates_literal(prev_code)
    weighting = list(dict.fromkeys(_WEIGHTING.findall(prev_code)))
    parts = [
        ("model classes", ", ".join(models)),
        ("engineered columns already created", ", ".join(columns)),
        ("param_candidates", params),
        ("class/sample weighting already applied, do not propose it again", ", ".join(weighting)),
    ]
    lines = [f"- {name}: {value}" for name, value in parts if value]
    return ["\n## Current base pipeline (already in place — build on it, do not re-add these)", *lines] if lines else []
