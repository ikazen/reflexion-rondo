"""대회 train 데이터 로딩 — S3(MinIO) 또는 로컬 우선순위, DROP_COLS 적용.

이전엔 bin/run_attempt_task.py, bin/run_promote_task.py, bin/run_cycle_task.py,
bin/run_daemon.py, bin/run_reflexion.py, bin/submit.py 6곳에 동일한 로딩 로직이
중복돼 있었다. attempt 생성 경로와 promote(cross-seed confirm·merge-verify) 경로가
반드시 같은 데이터를 로드해야 하므로(BON-256 merge-verify가 같은 seed·fold에서
같은 cv_score 재현을 전제) 한 곳으로 통합한다.
"""
from __future__ import annotations

import os

import polars as pl

_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").rstrip("/")


def load_train(comp: object) -> pl.DataFrame:
    """comp(config.competitions.<slug> 모듈)의 train.csv를 로드하고 DROP_COLS를 적용한다.

    S3_DATA_PATH가 설정돼 있고 MINIO_ENDPOINT가 있으면 MinIO에서, 아니면
    comp.DATA_DIR 로컬 경로에서 읽는다(기존 6곳의 인라인 로직과 동일).
    """
    s3_path = getattr(comp, "S3_DATA_PATH", None)
    if s3_path and _MINIO_ENDPOINT:
        train = pl.read_csv(f"{_MINIO_ENDPOINT}/kaggle/{s3_path}train.csv")
    else:
        train = pl.read_csv(comp.DATA_DIR / "train.csv")
    return train.drop(comp.DROP_COLS)
