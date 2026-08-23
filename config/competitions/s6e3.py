"""Kaggle Playground s6e3 — Customer Churn Prediction 대회 config (metric=auc).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s6e3"
NAME              = "Customer Churn Prediction"
TARGET            = "Churn"
METRIC            = "auc"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s6e3/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변
ACTIVE            = False  # False면 daemon 큐 리필(_sweep_queue_refill) 대상 제외 (#227, Milestone v1.6.0)

EDA_CARD = """competition: playground-series-s6e3 (Customer Churn Prediction)
task: binary classification  metric: AUC  target: Churn (Yes/No — 문자열, 이진 인코딩 필요)
rows: 594194  features: 19
target rate: No 77.5% / Yes 22.5% (minority=Yes, imbalanced)
no missing values

feature dtypes (as seen by feature_fn):
  gender             String   (2: Male / Female)
  SeniorCitizen      Int64    (0/1)
  Partner            String   (2: Yes / No)
  Dependents         String   (2: Yes / No)
  tenure             Int64    (개월 수)
  PhoneService       String   (2: Yes / No)
  MultipleLines      String   (3: Yes / No / No phone service)
  InternetService    String   (3: DSL / Fiber optic / No)
  OnlineSecurity     String   (3: Yes / No / No internet service)
  OnlineBackup       String   (3: Yes / No / No internet service)
  DeviceProtection   String   (3: Yes / No / No internet service)
  TechSupport        String   (3: Yes / No / No internet service)
  StreamingTV        String   (3: Yes / No / No internet service)
  StreamingMovies    String   (3: Yes / No / No internet service)
  Contract           String   (3: Month-to-month / One year / Two year — ordinal)
  PaperlessBilling   String   (2: Yes / No)
  PaymentMethod      String   (4: Electronic check / Mailed check / Bank transfer (automatic) /
                                   Credit card (automatic))
  MonthlyCharges     Float64
  TotalCharges        Float64

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
"No internet service"/"No phone service"는 별도 카테고리로 유지(단순 No로 합치면 InternetService=No와
구분 안 됨 — 정보 손실). Contract는 자연 순서(Month-to-month < One year < Two year)로 ordinal
매핑하는 것이 sorted() 알파벳 순서보다 신호를 보존한다. 나머지 string은 nominal — 알파벳 순 매핑으로 충분.

domain note: tenure x Contract, MonthlyCharges x InternetService 조합이 이탈(Churn)의 직관적
핵심 신호(단기 계약·높은 요금일수록 이탈 경향) — feature_engineering 후보."""
