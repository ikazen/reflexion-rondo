"""Kaggle Playground s5e9 — Predicting the Beats-per-Minute of Songs 대회 config (metric=rmse).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s5e9"
NAME              = "Predicting the Beats-per-Minute of Songs"
TARGET            = "BeatsPerMinute"
METRIC            = "rmse"
TASK_TYPE         = "regression"
METRIC_SIGN       = -1
IS_CLASSIFICATION = False
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s5e9/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변
ACTIVE            = False  # False면 daemon 큐 리필(_sweep_queue_refill) 대상 제외 (#227, Milestone v1.6.0)

EDA_CARD = """competition: playground-series-s5e9 (Predicting the Beats-per-Minute of Songs)
task: regression  metric: RMSE  target: BeatsPerMinute
rows: 524164  features: 9
target range: 46.718 - 206.037  mean: 119.03
no missing values, all features numeric (no string encoding needed)

feature dtypes (as seen by feature_fn):
  RhythmScore                Float64
  AudioLoudness               Float64
  VocalContent                Float64
  AcousticQuality              Float64
  InstrumentalScore            Float64
  LivePerformanceLikelihood    Float64
  MoodScore                    Float64
  TrackDurationMs               Float64
  Energy                       Float64

encoding note: 문자열 컬럼 없음 — feature_transform은 target 제거만 하면 충분, 인코딩 로직 불필요.

domain note: 오디오 특징 조합(RhythmScore x Energy, AcousticQuality x InstrumentalScore 등)이
BPM 인지와 직관적으로 연관 — feature_engineering 후보. 타깃 범위(46.7~206)에 강한 skew가
없어 log 변환 없이 raw scale RMSE 학습이 기본."""
