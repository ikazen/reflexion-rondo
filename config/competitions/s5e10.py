"""Kaggle Playground s5e10 — Road Accident Risk Prediction 대회 config (metric=rmse).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s5e10"
NAME              = "Road Accident Risk Prediction"
TARGET            = "accident_risk"
METRIC            = "rmse"
TASK_TYPE         = "regression"
METRIC_SIGN       = -1
IS_CLASSIFICATION = False
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s5e10/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s5e10 (Road Accident Risk Prediction)
task: regression  metric: RMSE  target: accident_risk (0~1 범위 확률성 스코어)
rows: 517754  features: 13
target range: 0.0 - 1.0  mean: 0.352  nunique: 98 (사실상 이산적인 값 집합 — 순수 연속형이 아닐
  가능성, 구간별 분포 확인 권장)
no missing values

feature dtypes (as seen by feature_fn):
  road_type                String   (3: rural / highway / urban)
  num_lanes                Int64
  curvature                 Float64
  speed_limit                Int64
  lighting                  String   (3: daylight / dim / night — ordinal 근사)
  weather                    String   (3: clear / rainy / foggy)
  road_signs_present          Boolean (True/False — pl.Boolean, 0/1 아님)
  public_road                 Boolean
  time_of_day                 String   (3: morning / afternoon / evening)
  holiday                    Boolean
  school_season               Boolean
  num_reported_accidents        Int64

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
road_signs_present/public_road/holiday/school_season은 pl.Boolean dtype — 문자열 매핑 로직이
아니라 dtype == pl.Boolean 분기로 처리해야 함(캐스팅 시 True/False를 1/0으로 직접 변환 가능,
별도 인코딩 테이블 불필요). lighting은 daylight/dim/night 밝기 순서로 ordinal 근사 가능.

domain note: curvature x speed_limit x weather 조합, num_reported_accidents(과거 사고 이력)가
위험도 예측의 직관적 핵심 신호 — feature_engineering 후보. target nunique=98로 적어 회귀보다
순위/구간 예측에 가까운 성격일 수 있음 — 예측값 clip [0,1] 권장."""
