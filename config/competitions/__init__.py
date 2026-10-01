"""대회 config 모듈 스캔 — competition_id ↔ slug 매핑과 ACTIVE(deep tier) 필터.

daemon 큐 리필·auto-submit·대시보드가 같은 판정을 써야 해서 한 곳에 모은다.
"""
from __future__ import annotations

import importlib
from pathlib import Path

_COMP_DIR = Path(__file__).parent


def competition_id_to_slug() -> dict[str, str]:
    """{competition_id: module_slug}. import에 실패한 모듈은 조용히 건너뛴다 —
    config 파일 하나가 깨져도 daemon 메인 루프 전체가 죽으면 안 된다(#223 계열)."""
    result: dict[str, str] = {}
    for path in sorted(_COMP_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        try:
            result[importlib.import_module(f"config.competitions.{path.stem}").COMPETITION_ID] = path.stem
        except Exception:
            continue
    return result


def active_competition_ids() -> set[str]:
    """ACTIVE=True인 대회 id 집합 — ADR-032의 deep tier."""
    return {
        cid for cid, slug in competition_id_to_slug().items()
        if importlib.import_module(f"config.competitions.{slug}").ACTIVE
    }


def attempt_cpu_budget_secs(comp: object) -> float | None:
    """confirm/holdout/merge-verify는 승격 후보를 재평가하므로 이 예산으로 자르지 않고 CPU_BUDGET_SECS를 그대로 쓴다(ADR-063)."""
    return getattr(comp, "ATTEMPT_CPU_BUDGET_SECS", getattr(comp, "CPU_BUDGET_SECS", None))
