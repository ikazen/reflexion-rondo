"""Kaggle Playground s4e10 — Loan Approval Prediction 대회 config (metric=auc).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s4e10"
NAME              = "Loan Approval Prediction"
TARGET            = "loan_status"
METRIC            = "auc"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s4e10/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s4e10 (Loan Approval Prediction)
task: binary classification  metric: AUC  target: loan_status
rows: 58645  features: 11
target rate: 14.2% default (loan_status=1) / 85.8% non-default (imbalanced)
no missing values

feature dtypes (as seen by feature_fn):
  person_age                     Int64
  person_income                  Int64
  person_home_ownership          String  (MORTGAGE / OTHER / OWN / RENT)
  person_emp_length              Float64
  loan_intent                    String  (DEBTCONSOLIDATION / EDUCATION / HOMEIMPROVEMENT / MEDICAL / PERSONAL / VENTURE)
  loan_grade                     String  (A / B / C / D / E / F / G)
  loan_amnt                      Int64
  loan_int_rate                  Float64
  loan_percent_income            Float64
  cb_person_default_on_file      String  (N / Y)
  cb_person_cred_hist_length     Int64

encoding note: person_home_ownership, loan_intent, loan_grade, cb_person_default_on_file are pl.String (NOT pl.Categorical).
detect with: dtype == pl.String  or  dtype in (pl.Utf8, pl.String)
ordinal encode: mapping = {v: i for i, v in enumerate(sorted(train[col].unique().to_list()))}
               df = df.with_columns(pl.col(col).replace_strict(mapping).cast(pl.Int32))
note: loan_grade has natural ordinal order (A=best → G=worst) — consider label-encoding in that order."""
