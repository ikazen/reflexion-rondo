"""Kaggle Playground s6e2 — Predicting Heart Disease 대회 config (metric=auc).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s6e2"
NAME              = "Predicting Heart Disease"
TARGET            = "Heart Disease"
METRIC            = "auc"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s6e2/data/"
EXTRA_TRAIN_PATHS: list[str] = []

EDA_CARD = """competition: playground-series-s6e2 (Predicting Heart Disease)
task: binary classification  metric: auc  target: Heart Disease
rows: 630000  features: 14
target classes: Absence 347546 (55.2%) / Presence 282454 (44.8%) — 거의 균형, 별도 리샘플링 불필요.
target은 string("Absence"/"Presence")이므로 학습 전 이진(0/1) 인코딩 필요.
missing values 없음 — 전 컬럼 결측 0%.

feature dtypes (as seen by feature_fn) — 전부 numeric(Int64/Float64)이지만 다수가 실제로는
UCI Heart Disease 데이터셋 표준 코드북을 따르는 범주형 정수임:
  Age                        Int64
  Sex                        Int64    (0/1 — 이진, 0=female/1=male 관례)
  Chest pain type            Int64    (1/2/3/4 — 범주형, 순서 의미 없음. one-hot 권장)
  BP                         Int64    (혈압, 연속형)
  Cholesterol                Int64    (연속형)
  FBS over 120               Int64    (0/1 — 공복혈당>120 이진 플래그)
  EKG results                Int64    (0/1/2 — 범주형)
  Max HR                     Int64    (연속형, 최대 심박수)
  Exercise angina            Int64    (0/1 — 이진)
  ST depression              Float64  (연속형)
  Slope of ST                Int64    (1/2/3 — 범주형, 순서 있을 수 있음: upsloping/flat/downsloping)
  Number of vessels fluro    Int64    (0/1/2/3 — 형광투시 조영 혈관 수, 순서 있는 count 성격)
  Thallium                   Int64    (3/6/7 — 범주형, 코드값 자체는 임의. 3=normal/6=fixed defect/
                               7=reversible defect UCI 관례. one-hot 권장, 숫자 크기로 순서 취급 금지)

encoding note: Chest pain type/EKG results/Thallium은 숫자로 저장돼 있지만 순서 없는 범주값이라
그대로 연속형으로 넣으면 잘못된 순서 관계를 모델에 주입하게 된다 — one-hot 인코딩 권장.
Slope of ST/Number of vessels fluro는 자연스러운 순서가 있어 연속형 그대로 사용 가능.

domain note: UCI Cleveland Heart Disease 데이터셋 기반 합성 데이터로 보이며, Thallium·
Number of vessels fluro·Chest pain type·ST depression·Exercise angina 조합이 심장질환
판별의 표준 핵심 신호로 알려져 있다."""
