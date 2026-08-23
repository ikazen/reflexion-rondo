"""Kaggle Playground s4e9 — Used Car Prices 대회 config (metric=rmse).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s4e9"
NAME              = "Used Car Prices"
TARGET            = "price"
METRIC            = "rmse"
TASK_TYPE         = "regression"
METRIC_SIGN       = -1
IS_CLASSIFICATION = False
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s4e9/data/"
EXTRA_TRAIN_PATHS: list[str] = []
ACTIVE            = False  # False면 daemon 큐 리필(_sweep_queue_refill) 대상 제외 (#227, Milestone v1.6.0)

EDA_CARD = """competition: playground-series-s4e9 (Used Car Prices)
task: regression  metric: rmse  target: price
rows: 188533  features: 12
price range: min 2000, 25%=17000, median=30825, 75%=49900, max 2,954,083, mean=43878, std=78819
  — heavily right-skewed with extreme outliers (luxury/exotic cars); log1p(price) 변환 후
  회귀하고 예측 시 expm1로 역변환하는 접근이 raw price 회귀보다 안정적일 가능성이 높다
  (metric_sign=-1이라 낮을수록 좋음, RMSE는 원 스케일 price 기준으로 채점되므로 역변환 필수).
missing values: fuel_type 2.7%, accident 1.3%, clean_title 11.4% — 나머지 컬럼 null 없음.

feature dtypes (as seen by feature_fn):
  model_year         Int64    (1974-2024)
  milage             Int64
  brand              String   (57 unique — 저카디널리티, one-hot/target encoding 가능)
  model              String   (1897 unique — 고카디널리티, target/frequency encoding 필요, one-hot 비권장)
  engine             String   (1117 unique, 자유형식 텍스트. 예: "295.0HP 3.6L V6 Cylinder Engine
                      Flex Fuel Capability" — HP(마력)·배기량(L)·실린더 수·연료타입이 문자열에
                      섞여 있음. 정규식으로 HP/배기량/실린더 수를 숫자 피처로 추출하는 파싱이
                      raw string encoding보다 신호를 훨씬 잘 보존할 가능성이 높음)
  transmission       String   (52 unique, 자유형식. 예: "8-SPEED AT", "7-Speed Automatic",
                      "SCHEDULED FOR OR IN PRODUCTION"(사실상 결측/이상값) — speed 숫자 추출 +
                      자동/수동 이진 플래그 파싱 권장)
  ext_col            String   (319 unique — 외장 색상, 고카디널리티)
  int_col            String   (156 unique — 내장 색상, 고카디널리티)
  fuel_type          String   (8 unique: Hybrid/Gasoline/Diesel/Plug-In Hybrid/E85 Flex Fuel/
                      "not supported"/"–"(대시 문자)/None — "not supported"와 "–"는 사실상 결측이니
                      null과 함께 별도 "unknown" 카테고리로 묶는 것을 권장)
  accident           String   (3 unique: "At least 1 accident or damage reported" / "None reported"
                      / None — 이진 플래그로 인코딩 가능, null은 별도 카테고리)
  clean_title        String   (2 unique: "Yes" / None — None은 "no" 의미로 보이나 확실친 않음,
                      이진 플래그 + null 카테고리로 처리)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
model/engine/ext_col/int_col처럼 고카디널리티 컬럼은 one-hot 시 컬럼 폭발 위험 — target encoding
또는 frequency encoding 권장. train 통계만 사용하고 valid/test에 동일 적용해야 leak 없음.

domain note: milage(주행거리)·model_year(연식)가 가격과 강한 음의 상관을 가질 것으로 예상되며,
engine에서 파싱한 HP·배기량이 luxury/exotic 세그먼트(price 꼬리 부분) 판별에 핵심 신호일 가능성."""
