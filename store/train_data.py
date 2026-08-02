"""대회 train 데이터 로딩 — S3(MinIO) 또는 로컬 우선순위, DROP_COLS 적용.

attempt 생성 경로와 promote(cross-seed confirm·merge-verify) 경로가 반드시 같은
데이터를 로드해야 cv_score가 재현되므로 로딩 로직을 이 한 곳으로 통합한다.

comp.EXTRA_TRAIN_PATHS(기본 빈 리스트)가 설정된 대회는 Playground Series
합성 원본 데이터셋을 train에 병합한다. Patch(LLM 생성 코드)의 no-IO 원칙은 유지 —
병합은 이 로더(config/task 레벨)에서만 일어난다.
"""
from __future__ import annotations

import os

import polars as pl

_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")


_MAX_TRAIN_ROWS_SEED = 42  # 고정 — attempt 평가와 promote(cross-seed confirm·merge-verify)가
# 반드시 같은 표본을 봐야 cv_score가 재현된다(materialize.py의 replay 전제와 동일 이유).


def _load_csv(comp: object, filename: str) -> pl.DataFrame:
    s3_path = getattr(comp, "S3_DATA_PATH", None)
    if s3_path and _MINIO_ENDPOINT:
        return pl.read_csv(f"{_MINIO_ENDPOINT}/kaggle/{s3_path}{filename}")
    return pl.read_csv(comp.DATA_DIR / filename)


def _stratified_sample(df: pl.DataFrame, target_col: str, n: int, seed: int) -> pl.DataFrame:
    """target_col 클래스 비율을 유지한 채 대략 n행으로 축소한다(분류 전용).

    회귀(연속형) 타깃에는 부적합 — 호출부가 comp.IS_CLASSIFICATION으로 분기해서만
    쓴다. group별 fraction 샘플이라 반올림으로 정확히 n행은 아닐 수 있다(무시할
    수준 오차, s4e7처럼 수백만 행 규모에서 문제되지 않음).
    """
    frac = n / df.height
    parts = [group.sample(fraction=frac, seed=seed) for _, group in df.group_by(target_col, maintain_order=True)]
    return pl.concat(parts)


def load_train(comp: object) -> pl.DataFrame:
    """comp(config.competitions.<slug> 모듈)의 train.csv를 로드하고 DROP_COLS를 적용한다.

    S3_DATA_PATH가 설정돼 있고 MINIO_ENDPOINT가 있으면 MinIO에서, 아니면
    comp.DATA_DIR 로컬 경로에서 읽는다.

    comp.EXTRA_TRAIN_PATHS(기본 없음/빈 리스트)가 있으면 각 경로를 같은 S3-or-local
    규칙으로 추가 로드해 concat한다 — 원본이 합성 대회 train과 완전히 같은 스키마가
    아닐 수 있어 base와 공통되는 컬럼만 취하고(교집합), 없는 컬럼은 null로 채운다
    (`is_original` 플래그로 구분). target 컬럼명이 base와 동일하다고 가정한다 —
    다르면 해당 원본 행의 target은 null이 되어 사실상 버려진다(현재는 이름 매핑을
    지원하지 않음).

    comp.MAX_TRAIN_ROWS(opt-in, 기본 없음)가 설정돼 있고 로드된 행 수가 그보다 크면
    고정 seed로 축소한다. 분류(IS_CLASSIFICATION=True)면 클래스 비율을 보존하는
    층화 샘플링, 아니면 단순 랜덤 샘플. 미설정 대회는 동작 완전 불변.
    """
    train = _load_csv(comp, "train.csv").drop(comp.DROP_COLS)

    extra_paths = getattr(comp, "EXTRA_TRAIN_PATHS", None) or []
    if extra_paths:
        train = train.with_columns(pl.lit(False).alias("is_original"))
        base_cols = set(train.columns) - {"is_original"}
        frames = [train]
        for path in extra_paths:
            extra = _load_csv(comp, path)
            common_cols = [c for c in extra.columns if c in base_cols]
            extra = extra.select(common_cols).with_columns(pl.lit(True).alias("is_original"))
            frames.append(extra)
        train = pl.concat(frames, how="diagonal_relaxed")

    max_rows = getattr(comp, "MAX_TRAIN_ROWS", None)
    if max_rows and train.height > max_rows:
        if getattr(comp, "IS_CLASSIFICATION", False):
            train = _stratified_sample(train, comp.TARGET, max_rows, seed=_MAX_TRAIN_ROWS_SEED)
        else:
            train = train.sample(n=max_rows, seed=_MAX_TRAIN_ROWS_SEED)

    return train
