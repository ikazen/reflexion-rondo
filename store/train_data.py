"""대회 train 데이터 로딩 — S3(MinIO) 또는 로컬 우선순위, DROP_COLS 적용.

attempt 생성 경로와 promote(cross-seed confirm·merge-verify) 경로가 반드시 같은
데이터를 로드해야 cv_score가 재현되므로 로딩 로직을 이 한 곳으로 통합한다.

comp.EXTRA_TRAIN_PATHS(기본 빈 리스트)가 설정된 대회는 Playground Series
합성 원본 데이터셋을 train에 병합한다. Patch(LLM 생성 코드)의 no-IO 원칙은 유지 —
병합은 이 로더(config/task 레벨)에서만 일어난다.
"""
from __future__ import annotations

import logging
import os

import polars as pl

_LOG = logging.getLogger(__name__)

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


def _target_present(df: pl.DataFrame, target: str) -> pl.Expr:
    expr = pl.col(target).is_not_null()
    if df.schema[target].is_float():
        expr = expr & pl.col(target).is_not_nan()
    return expr


_TWIN_ABORT_FRAC = 0.5  # EXTRA_TRAIN_PATHS 소스가 base와 이 비율 넘게 중복이면 설정
# 실수로 보고 즉시 실패시킨다 — s4e11(#228)에서 Kaggle이 이미 train.csv에 원본 27,901행을
# 통째로 포함시켜놨는데도 EXTRA_TRAIN_PATHS로 다시 병합해 twin 99.72%가 조용히 CV를
# 오염시킨 사고(evaluator/harness.py의 validation twin이 학습 fold에 정답 사본으로 존재,
# cv_score가 세계 1위 LB를 넘어서야 발견됨)의 재발 방지.


def _dedup_key_expr(cols: list[str]) -> pl.Expr:
    """cols 전체(타깃 포함)를 값 기준으로 정규화한 문자열 키. 숫자는 dtype 무관(문자열로
    읽힌 "24"와 float 24.0)하게 Float64 경유로 통일하고, 그 외는 공백 trim한 문자열로
    비교한다 — base/extra가 서로 다른 소스라 dtype이나 포맷팅이 어긋날 수 있어서다."""
    parts = [
        pl.when(pl.col(c).cast(pl.Float64, strict=False).is_not_null())
        .then(pl.col(c).cast(pl.Float64, strict=False).cast(pl.Utf8))
        .otherwise(pl.col(c).cast(pl.Utf8).str.strip_chars())
        .fill_null("\x00NULL\x00")
        for c in cols
    ]
    return pl.concat_str(parts, separator="|")


def load_train(comp: object) -> pl.DataFrame:
    """comp(config.competitions.<slug> 모듈)의 train.csv를 로드하고 DROP_COLS를 적용한다.

    S3_DATA_PATH가 설정돼 있고 MINIO_ENDPOINT가 있으면 MinIO에서, 아니면
    comp.DATA_DIR 로컬 경로에서 읽는다.

    comp.EXTRA_TRAIN_PATHS(기본 없음/빈 리스트)가 있으면 각 경로를 같은 S3-or-local
    규칙으로 추가 로드해 concat한다 — 원본이 합성 대회 train과 완전히 같은 스키마가
    아닐 수 있어 base와 공통되는 컬럼만 취하고(교집합), 없는 컬럼은 null로 채운다
    (`is_original` 플래그로 구분). target 컬럼명 매핑은 지원하지 않으므로 원본에
    comp.TARGET이 없으면 예외로 실패하고, 있어도 값이 비어 있는 행은 버린다 —
    타깃 결측 행은 그대로 두면 harness의 model.fit에서 "Input y contains NaN"으로
    크래시한다(#245, s5e4 original.csv 52500행 중 5395행). 원본 소스가 base(train.csv)와
    common_cols 전체 기준 완전 일치하는 twin 행은 base 쪽만 남기고 버린다(#228 s4e11
    사고 — Kaggle이 이미 train.csv에 원본을 포함시켜놔 재병합이 CV를 암기로 오염시킴).
    twin 비율이 소스의 50% 넘으면 설정 실수로 보고 예외로 실패한다.

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
            if comp.TARGET not in common_cols:
                raise ValueError(
                    f"EXTRA_TRAIN_PATHS {path!r}: target column {comp.TARGET!r} missing "
                    f"(columns: {sorted(extra.columns)})"
                )
            extra = extra.select(common_cols)
            loaded = extra.height
            extra = extra.filter(_target_present(extra, comp.TARGET))
            if extra.height < loaded:
                _LOG.warning(
                    "load_train: %s dropped %d/%d rows with missing target %r",
                    path, loaded - extra.height, loaded, comp.TARGET,
                )

            # Kaggle 합성 대회는 원본 실데이터를 train.csv에 이미 일부/전부 포함시켜
            # 놓는 경우가 있다(twin 행) — EXTRA_TRAIN_PATHS로 그걸 또 병합하면 같은 행이
            # 학습 fold와 validation 양쪽에 들어가 CV가 암기로 부풀려진다(#228 s4e11 사고).
            # base(원 train)와 common_cols 전체(타깃 포함) 기준 완전 일치 행은 버린다.
            key_col = "_twin_dedup_key"
            base_keys = train.select(common_cols).with_columns(
                _dedup_key_expr(common_cols).alias(key_col)
            ).select(key_col)
            before_dedup = extra.height
            extra = (
                extra.with_columns(_dedup_key_expr(common_cols).alias(key_col))
                .join(base_keys, on=key_col, how="anti")
                .drop(key_col)
            )
            removed = before_dedup - extra.height
            if removed:
                twin_frac = removed / loaded
                _LOG.warning(
                    "load_train: %s dedup twin 행 %d/%d(%.1f%%) base와 중복 제거",
                    path, removed, loaded, 100 * twin_frac,
                )
                if twin_frac > _TWIN_ABORT_FRAC:
                    raise ValueError(
                        f"EXTRA_TRAIN_PATHS {path!r}: base와 {twin_frac:.1%} 중복(twin) — "
                        f"Kaggle이 이미 원본을 train.csv에 포함시켜놨을 가능성이 높다. "
                        f"이 대회는 EXTRA_TRAIN_PATHS를 빈 리스트로 되돌려야 할 수 있다."
                    )

            extra = extra.with_columns(pl.lit(True).alias("is_original"))
            frames.append(extra)
        train = pl.concat(frames, how="diagonal_relaxed")

    max_rows = getattr(comp, "MAX_TRAIN_ROWS", None)
    if max_rows and train.height > max_rows:
        if getattr(comp, "IS_CLASSIFICATION", False):
            train = _stratified_sample(train, comp.TARGET, max_rows, seed=_MAX_TRAIN_ROWS_SEED)
        else:
            train = train.sample(n=max_rows, seed=_MAX_TRAIN_ROWS_SEED)

    return train
