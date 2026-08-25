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
            mod = importlib.import_module(f"config.competitions.{path.stem}")
        except Exception:
            continue
        cid = getattr(mod, "COMPETITION_ID", None)
        if cid:
            result[cid] = path.stem
    return result


def active_competition_ids() -> set[str]:
    """ACTIVE=True인 대회 id 집합 — ADR-032의 deep tier. ACTIVE 미선언은 True로 본다."""
    result: set[str] = set()
    for cid, slug in competition_id_to_slug().items():
        try:
            mod = importlib.import_module(f"config.competitions.{slug}")
        except Exception:
            continue
        if getattr(mod, "ACTIVE", True):
            result.add(cid)
    return result
