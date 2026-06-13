"""의존성 health 점검 스크립트.

Usage:
    python -m bin.healthcheck                    # DB/MinIO/Ollama/Airflow 연결 확인
    python -m bin.healthcheck --cycle s4e1       # + 1사이클 실행 (LLM 토큰 소모)
"""
from __future__ import annotations

import argparse
import os
import sys

import requests

import config.settings as settings

_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# 개별 체크 — 반환값: (True|False|None, detail_str)
#   None → SKIP (환경변수 미설정)
# ---------------------------------------------------------------------------

def check_postgres() -> tuple[bool | None, str]:
    try:
        from store.db import connect
        conn = connect(apply_schema=False)
        conn.execute("select 1")
        conn.close()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def check_minio() -> tuple[bool | None, str]:
    endpoint = os.getenv("MINIO_ENDPOINT", "").rstrip("/")
    if not endpoint:
        return None, "MINIO_ENDPOINT not set"
    try:
        r = requests.get(f"{endpoint}/minio/health/live", timeout=5)
        if r.status_code in (200, 204):
            return True, ""
        return False, f"HTTP {r.status_code}"
    except Exception as exc:
        return False, str(exc)


def check_ollama_cloud() -> tuple[bool | None, str]:
    base = settings.OLLAMA_CLOUD_BASE_URL
    key = settings.OLLAMA_API_KEY
    if not key:
        return None, "OLLAMA_API_KEY not set"
    try:
        r = requests.get(
            f"{base}/api/tags",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        r.raise_for_status()
        names = {m["name"] for m in r.json().get("models", [])}
        chat_models = [settings.MODEL_STRATEGIST, settings.MODEL_REFLECTOR, settings.MODEL_CODER]
        missing = [m for m in chat_models if m not in names]
        if missing:
            return False, f"missing: {', '.join(missing)}"
        return True, f"{len(chat_models)} chat models ok"
    except Exception as exc:
        return False, str(exc)


def check_ollama_local() -> tuple[bool | None, str]:
    base = settings.OLLAMA_BASE_URL
    try:
        r = requests.get(f"{base}/api/tags", timeout=5)
        r.raise_for_status()
        names = {m["name"] for m in r.json().get("models", [])}
        if settings.MODEL_EMBEDDING not in names:
            return False, f"missing: {settings.MODEL_EMBEDDING}"
        return True, settings.MODEL_EMBEDDING
    except Exception as exc:
        return False, str(exc)


def check_airflow() -> tuple[bool | None, str]:
    from bin import airflow_client
    if not airflow_client.available():
        return None, "AIRFLOW_URL not set"
    try:
        airflow_client._bearer_token()
        return True, ""
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# run_checks — api.py /api/health 에서도 재사용
# ---------------------------------------------------------------------------

def run_checks() -> dict[str, dict]:
    """반환: {name: {"status": "pass"|"fail"|"skip", "detail": str}}"""
    checks = [
        ("postgres",     check_postgres),
        ("minio",        check_minio),
        ("ollama_cloud", check_ollama_cloud),
        ("ollama_local", check_ollama_local),
        ("airflow",      check_airflow),
    ]
    results: dict[str, dict] = {}
    for name, fn in checks:
        ok, detail = fn()
        status = "skip" if ok is None else ("pass" if ok else "fail")
        results[name] = {"status": status, "detail": detail}
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _row(label: str, ok: bool | None, detail: str) -> None:
    if ok is None:
        tag = f"{_YELLOW}SKIP{_RESET}"
    elif ok:
        tag = f"{_GREEN}PASS{_RESET}"
    else:
        tag = f"{_RED}FAIL{_RESET}"
    suffix = f"  {detail}" if detail else ""
    print(f"  {label:<20} {tag}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="reflexion-rondo dependency health check")
    parser.add_argument("--cycle", metavar="COMPETITION",
                        help="health 통과 후 1사이클 실행 (LLM 토큰 소모)")
    args = parser.parse_args()

    print("\nreflexion-rondo healthcheck")
    print("-" * 44)

    results = run_checks()
    failed = []

    for name, r in results.items():
        ok = {"pass": True, "fail": False, "skip": None}[r["status"]]
        if r["status"] == "fail":
            failed.append(name)
        _row(name, ok, r["detail"])

    print("-" * 44)

    if failed:
        print(f"{_RED}FAIL{_RESET}: {', '.join(failed)}\n")
        sys.exit(1)

    print(f"{_GREEN}OK{_RESET}\n")

    if args.cycle:
        _run_cycle(args.cycle)


def _run_cycle(competition: str) -> None:
    import importlib
    import time

    import polars as pl

    from cycle.run import CycleConfig, run_cycle
    from store.db import connect

    comp = importlib.import_module(f"config.competitions.{competition}")
    train_path = comp.DATA_DIR / "train.csv"
    if not train_path.exists():
        print(f"SKIP cycle: {train_path} not found")
        return

    print(f"running 1 cycle for {competition} ...")
    train = pl.read_csv(train_path)
    if hasattr(comp, "DROP_COLS"):
        train = train.drop([c for c in comp.DROP_COLS if c in train.columns])

    conn = connect(apply_schema=False)
    config = CycleConfig(
        competition_id=comp.COMPETITION_ID,
        train=train,
        target_col=comp.TARGET,
        metric=comp.METRIC,
        stage="bootstrap",
        eda_card=comp.EDA_CARD,
        n_splits=getattr(comp, "N_SPLITS", 5),
        seed=42,
        k_retrieve=5,
        is_classification=comp.IS_CLASSIFICATION,
    )
    t0 = time.time()
    result = run_cycle(conn, config)
    conn.close()
    print(f"cycle done  attempt={result.attempt_id[:8]} cv={result.cv_score} [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
