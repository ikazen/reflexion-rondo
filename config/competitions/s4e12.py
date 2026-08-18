"""Kaggle Playground s4e12 — Insurance Premium Prediction 대회 config (metric=rmsle).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s4e12"
NAME              = "Insurance Premium Prediction"
TARGET            = "Premium Amount"
METRIC            = "rmsle"
TASK_TYPE         = "regression"
METRIC_SIGN       = -1
IS_CLASSIFICATION = False
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s4e12/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s4e12 (Insurance Premium Prediction)
task: regression  metric: RMSLE  target: Premium Amount (컬럼명에 공백 포함 —
  pl.col("Premium Amount")로 접근)
rows: 1200000  features: 19
target range: 20.0 - 4999.0  mean: 1102.54  nunique: 4794 (right-skew 가능성 — RMSLE 채택과 정합)
결측: Age 1.6%, Annual Income 3.7%, Marital Status 1.5%, Number of Dependents 9.1%,
  Occupation 29.8%, Health Score 6.2%, Previous Claims 30.3%, Vehicle Age <0.1%,
  Credit Score 11.5%, Insurance Duration <0.1%, Customer Feedback 6.5%

feature dtypes (as seen by feature_fn):
  Age                    Float64 (has nulls)
  Gender                 String   (2: Male / Female)
  Annual Income          Float64 (has nulls)
  Marital Status         String   (4: Married / Single / Divorced, has nulls)
  Number of Dependents   Float64 (has nulls)
  Education Level        String   (4: High School / Bachelor's / Master's / PhD — ordinal)
  Occupation              String   (4: Employed / Self-Employed / Unemployed, has nulls — 29.8%
                                         결측)
  Health Score            Float64
  Location                String   (3: Urban / Suburban / Rural)
  Policy Type             String   (3: Basic / Comprehensive / Premium — ordinal)
  Previous Claims          Float64 (has nulls — 30.3% 결측, 청구 이력 없음을 의미할 가능성)
  Vehicle Age              Float64 (has nulls)
  Credit Score              Float64 (has nulls)
  Insurance Duration          Float64 (has nulls)
  Policy Start Date         String   (datetime 문자열, 예: "2023-08-02 15:21:39.097737" — 파싱 필요,
                                          167381개 고유값)
  Customer Feedback         String   (3: Poor / Average / Good — ordinal, has nulls)
  Smoking Status             String   (2: Yes / No)
  Exercise Frequency          String   (4: Rarely / Monthly / Weekly / Daily — ordinal)
  Property Type              String   (3: Apartment / Condo / House)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
컬럼명에 공백 포함 다수 — pl.col("...")로 정확히 인용 필요. Policy Start Date는
pl.col(...).str.to_datetime()으로 파싱 후 연/월/요일 등 파생 필요(원본 문자열 그대로는 무의미).
Education Level/Policy Type/Customer Feedback/Exercise Frequency는 자연 순서로 ordinal 매핑
권장. Previous Claims 결측은 "청구 없음(0)"과 "정보 없음(missing)"이 다를 수 있어 별도 indicator
컬럼 추가 고려.

domain note: Health Score x Age x Smoking Status, Vehicle Age x Credit Score 조합이 보험료 산정의
직관적 핵심 신호 — feature_engineering 후보."""
