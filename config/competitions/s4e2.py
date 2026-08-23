"""Kaggle Playground s4e2 — Multi-Class Prediction of Obesity Risk 대회 config (metric=accuracy).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s4e2"
NAME              = "Multi-Class Prediction of Obesity Risk"
TARGET            = "NObeyesdad"
METRIC            = "accuracy"
TASK_TYPE         = "multiclass"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s4e2/data/"
EXTRA_TRAIN_PATHS: list[str] = []
ACTIVE            = False  # False면 daemon 큐 리필(_sweep_queue_refill) 대상 제외 (#227, Milestone v1.6.0)

EDA_CARD = """competition: playground-series-s4e2 (Multi-Class Prediction of Obesity Risk)
task: multiclass classification (7 classes)  metric: accuracy  target: NObeyesdad
rows: 20758  features: 17
target classes (fairly balanced, no imbalance handling needed): Obesity_Type_III 4046 (19.5%) /
  Obesity_Type_II 3248 (15.6%) / Normal_Weight 3082 (14.8%) / Obesity_Type_I 2910 (14.0%) /
  Insufficient_Weight 2523 (12.2%) / Overweight_Level_II 2522 (12.1%) / Overweight_Level_I 2427 (11.7%)
no missing values anywhere (feature_fn does not need null handling for this competition).

feature dtypes (as seen by feature_fn):
  Age                              Float64
  Height                            Float64  (meters)
  Weight                            Float64  (kg)
  FCVC                              Float64  (frequency of vegetable consumption, 1-3 scale)
  NCP                               Float64  (number of main meals, 1-4 scale)
  CH2O                              Float64  (water intake, 1-3 scale)
  FAF                               Float64  (physical activity frequency, 0-3 scale)
  TUE                               Float64  (time using technology devices, 0-2 scale)
  Gender                            String   (Female / Male)
  family_history_with_overweight    String   (yes / no)
  FAVC                              String   (yes / no — frequent high-caloric food consumption)
  CAEC                              String   (no / Sometimes / Frequently / Always — ordinal, eating between meals)
  SMOKE                             String   (yes / no)
  SCC                               String   (yes / no — calorie consumption monitoring)
  CALC                              String   (no / Sometimes / Frequently — ordinal, alcohol consumption)
  MTRANS                            String   (Walking / Bike / Public_Transportation / Motorbike / Automobile)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
ordinal 컬럼(CAEC/CALC)은 자연 순서로 명시 매핑하는 것이 sorted() 알파벳 순서보다 신호를 보존한다
(예: no < Sometimes < Frequently < Always).

domain note: 이 데이터는 실측이 아니라 SMOTE로 합성 증강된 원본 UCI 데이터셋 기반이라 알려져 있다
(Kaggle 대회 설명 참고). Weight/Height로 BMI를 직접 계산하는 파생 피처가 타깃(비만도 분류)과
강한 상관을 가질 가능성이 높다 — 다만 원본 라벨링이 BMI 임계값 기반이라 이 파생 피처가 사실상
타깃을 거의 결정해버리는 leak에 가까운 신호일 수 있음에 유의(대회 표준 접근이지만 과적합 여지)."""
