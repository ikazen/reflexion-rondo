"""대회 train 데이터 로딩 — S3(MinIO) 또는 로컬 우선순위, DROP_COLS 적용.

이전엔 bin/run_attempt_task.py, bin/run_promote_task.py, bin/run_cycle_task.py,
bin/run_daemon.py, bin/run_reflexion.py, bin/submit.py 6곳에 동일한 로딩 로직이
중복돼 있었다. attempt 생성 경로와 promote(cross-seed confirm·merge-verify) 경로가
반드시 같은 데이터를 로드해야 하므로(BON-256 merge-verify가 같은 seed·fold에서
같은 cv_score 재현을 전제) 한 곳으로 통합한다.

BON-250: comp.EXTRA_TRAIN_PATHS(기본 빈 리스트)가 설정된 대회는 Playground Series
합성 원본 데이터셋을 train에 병합한다. Patch(LLM 생성 코드)의 no-IO 원칙은 유지 —
병합은 이 로더(config/task 레벨)에서만 일어난다.
"""
from __future__ import annotations

import os

import polars as pl

_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")


def _load_csv(comp: object, filename: str) -> pl.DataFrame:
    s3_path = getattr(comp, "S3_DATA_PATH", None)
    if s3_path and _MINIO_ENDPOINT:
        return pl.read_csv(f"{_MINIO_ENDPOINT}/kaggle/{s3_path}{filename}")
    return pl.read_csv(comp.DATA_DIR / filename)


def load_train(comp: object) -> pl.DataFrame:
    """comp(config.competitions.<slug> 모듈)의 train.csv를 로드하고 DROP_COLS를 적용한다.

    S3_DATA_PATH가 설정돼 있고 MINIO_ENDPOINT가 있으면 MinIO에서, 아니면
    comp.DATA_DIR 로컬 경로에서 읽는다(기존 6곳의 인라인 로직과 동일).

    comp.EXTRA_TRAIN_PATHS(기본 없음/빈 리스트)가 있으면 각 경로를 같은 S3-or-local
    규칙으로 추가 로드해 concat한다 — 원본이 합성 대회 train과 완전히 같은 스키마가
    아닐 수 있어 base와 공통되는 컬럼만 취하고(교집합), 없는 컬럼은 null로 채운다
    (`is_original` 플래그로 구분). target 컬럼명이 base와 동일하다고 가정한다 —
    다르면 해당 원본 행의 target은 null이 되어 사실상 버려진다(opt-in 기능이라
    현재는 이름 매핑을 지원하지 않음, 필요해지면 후속 확장).
    """
    train = _load_csv(comp, "train.csv").drop(comp.DROP_COLS)

    extra_paths = getattr(comp, "EXTRA_TRAIN_PATHS", None) or []
    if not extra_paths:
        # EXTRA_TRAIN_PATHS 미설정 시 is_original 컬럼조차 추가하지 않는다 —
        # 기존 대회의 스키마·동작을 완전히 불변으로 유지(opt-in 원칙).
        return train

    train = train.with_columns(pl.lit(False).alias("is_original"))
    base_cols = set(train.columns) - {"is_original"}
    frames = [train]
    for path in extra_paths:
        extra = _load_csv(comp, path)
        common_cols = [c for c in extra.columns if c in base_cols]
        extra = extra.select(common_cols).with_columns(pl.lit(True).alias("is_original"))
        frames.append(extra)
    return pl.concat(frames, how="diagonal_relaxed")
