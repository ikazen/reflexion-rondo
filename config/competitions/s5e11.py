from pathlib import Path

COMPETITION_ID    = "playground-series-s5e11"
NAME              = "Loan Payback Prediction"
TARGET            = "loan_paid_back"
METRIC            = "auc"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s5e11/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s5e11 (Loan Payback Prediction)
task: binary classification  metric: AUC  target: loan_paid_back
rows: 593994  features: 11
target rate: 79.9% paid back / 20.1% not paid back (minority=not paid back, imbalanced)
no missing values

feature dtypes (as seen by feature_fn):
  annual_income        Float64
  debt_to_income_ratio Float64
  credit_score         Int64
  loan_amount          Float64
  interest_rate        Float64
  gender               String   (3: Male / Female / Other)
  marital_status       String   (4: Single / Married / Divorced / Widowed)
  education_level      String   (5: High School / Bachelor's / Master's / PhD / Other)
  employment_status    String   (5: Employed / Self-employed / Unemployed / Retired / Student)
  loan_purpose         String   (8: Debt consolidation / Home / Car / Medical / Business /
                                     Education / Vacation / Other)
  grade_subgrade       String   (30: 세부 신용등급, 예 A1~F5 — 알파벳+숫자 조합, ordinal 근사 가능)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
grade_subgrade는 문자(A~F, 낮을수록 우량)+숫자(1~5, 낮을수록 우량) 조합이라 문자/숫자를 분리해
ordinal 매핑하면 sorted() 알파벳 순서보다 신호를 보존한다. 나머지는 nominal — 알파벳 순 매핑으로 충분.

domain note: credit_score x debt_to_income_ratio, grade_subgrade x interest_rate 조합이
상환 여부(loan_paid_back)의 직관적 핵심 신호 — feature_engineering 후보."""
