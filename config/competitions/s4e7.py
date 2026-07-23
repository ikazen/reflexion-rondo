from pathlib import Path

COMPETITION_ID    = "playground-series-s4e7"
NAME              = "Insurance Cross-Sell Prediction"
TARGET            = "Response"
METRIC            = "auc"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s4e7/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s4e7 (Insurance Cross-Sell Prediction)
task: binary classification  metric: AUC  target: Response
rows: 11504798  features: 10 (대회 중 최대 규모 — 학습 시간/메모리 예산 고려)
target rate: 12.3% Response=1 / 87.7% Response=0 (minority=1, imbalanced)
no missing values

feature dtypes (as seen by feature_fn):
  Gender                String   (2: Male / Female)
  Age                   Int64
  Driving_License        Int64   (0/1)
  Region_Code             Float64 (범주형 코드지만 numeric으로 인코딩돼 있음 — categorical 취급 고려)
  Previously_Insured      Int64   (0/1)
  Vehicle_Age            String   (3: < 1 Year / 1-2 Year / > 2 Years — ordinal)
  Vehicle_Damage          String   (2: Yes / No)
  Annual_Premium           Float64
  Policy_Sales_Channel    Float64 (범주형 코드지만 numeric으로 인코딩돼 있음 — categorical 취급 고려)
  Vintage                 Int64

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
Vehicle_Age는 자연 순서(< 1 Year < 1-2 Year < > 2 Years)로 ordinal 매핑하는 것이 sorted() 알파벳
순서보다 신호를 보존한다. Region_Code/Policy_Sales_Channel은 dtype만 Float64일 뿐 실질은 범주형
코드 — target encoding/frequency encoding 후보.

domain note: Previously_Insured x Vehicle_Damage 조합이 교차판매(Response) 여부의 직관적 핵심
신호(이미 보험 있고 손상 이력 없으면 가입 유인 낮음) — feature_engineering 후보. 1150만행 규모라
학습/CV 시간 예산에 유의."""
