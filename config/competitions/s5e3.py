"""Kaggle Playground s5e3 — Rainfall Prediction in Australia 대회 config (metric=auc).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s5e3"
NAME              = "Rainfall Prediction in Australia"
TARGET            = "rainfall"
METRIC            = "auc"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
N_SPLITS          = 10
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s5e3/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변
ACTIVE            = False  # False면 daemon 큐 리필(_sweep_queue_refill) 대상 제외 (#227, Milestone v1.6.0)

EDA_CARD = """competition: playground-series-s5e3 (Rainfall Prediction in Australia)
task: binary classification  metric: AUC  target: rainfall
rows: 2190  features: 11 (all numeric)
target rate: 75.3% rain / 24.7% no-rain (minority=no-rain, imbalanced)
no missing values

feature dtypes (as seen by feature_fn):
  day            Int64
  pressure       Float64
  maxtemp        Float64
  temparature    Float64
  mintemp        Float64
  dewpoint       Float64
  humidity       Float64
  cloud          Float64
  sunshine       Float64
  winddirection  Float64
  windspeed      Float64

note: all features are numeric — no encoding needed.
note: dataset is very small (2190 rows) — overfitting risk high, use regularization."""
