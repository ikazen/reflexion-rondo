"""Kaggle Playground s6e5 — F1 Pit Stops Prediction 대회 config (metric=auc).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s6e5"
NAME              = "F1 Pit Stops Prediction"
TARGET            = "PitNextLap"
METRIC            = "auc"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s6e5/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변
ACTIVE            = False  # False면 daemon 큐 리필(_sweep_queue_refill) 대상 제외 (#227, Milestone v1.6.0)

EDA_CARD = """competition: playground-series-s6e5 (F1 Pit Stops Prediction)
task: binary classification  metric: AUC  target: PitNextLap (다음 랩에 피트스톱 여부)
rows: 439140  features: 14
target rate: 19.9% PitNextLap=1 / 80.1% =0 (minority=1, imbalanced)
no missing values

feature dtypes (as seen by feature_fn):
  Driver                    String   (887개 카테고리 — 매우 높은 cardinality, target/frequency
                                          encoding 권장, one-hot 비권장)
  Compound                  String   (5: MEDIUM / SOFT / HARD / INTERMEDIATE / WET — 타이어 컴파운드)
  Race                      String   (26개 그랑프리명)
  Year                      Int64
  PitStop                   Int64    (누적 피트스톱 횟수로 추정)
  LapNumber                 Int64
  Stint                     Int64
  TyreLife                   Float64 (타이어 사용 랩 수 — PitNextLap과 강상관 예상)
  Position                  Int64
  LapTime (s)               Float64  (컬럼명에 공백+괄호 포함 — pl.col("LapTime (s)")로 접근 필요)
  LapTime_Delta             Float64
  Cumulative_Degradation    Float64
  RaceProgress              Float64
  Position_Change           Float64

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
Driver(887)/Race(26)는 고cardinality — target encoding 또는 frequency encoding 우선 고려, 알파벳
순 정수 매핑은 신호 손실. Compound는 소프트→하드 마모도 순서로 ordinal 근사 가능.

domain note: TyreLife x Cumulative_Degradation, Stint x LapNumber 조합이 피트스톱 타이밍의 핵심
레이싱 도메인 신호 — feature_engineering 후보. Driver/Race는 시계열적 반복 구조(같은 드라이버/
레이스 내 랩 시퀀스)이므로 그룹 단위 집계(driver별 평균 TyreLife 등)가 유효할 수 있음."""
