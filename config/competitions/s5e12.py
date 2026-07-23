from pathlib import Path

COMPETITION_ID    = "playground-series-s5e12"
NAME              = "Diabetes Prediction"
TARGET            = "diagnosed_diabetes"
METRIC            = "auc"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s5e12/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s5e12 (Diabetes Prediction)
task: binary classification  metric: AUC  target: diagnosed_diabetes
rows: 700000  features: 24
target rate: 62.3% diagnosed=1 / 37.7% diagnosed=0 (다수 클래스가 양성 — 다른 대회들과 반대)
no missing values

feature dtypes (as seen by feature_fn):
  age                                  Int64
  alcohol_consumption_per_week         Int64
  physical_activity_minutes_per_week   Int64
  diet_score                           Float64
  sleep_hours_per_day                  Float64
  screen_time_hours_per_day            Float64
  bmi                                  Float64
  waist_to_hip_ratio                   Float64
  systolic_bp                          Int64
  diastolic_bp                         Int64
  heart_rate                           Int64
  cholesterol_total                    Int64
  hdl_cholesterol                      Int64
  ldl_cholesterol                      Int64
  triglycerides                        Int64
  gender                               String   (3: Male / Female / Other)
  ethnicity                            String   (5: White / Black / Asian / Hispanic / Other)
  education_level                      String   (4: No formal / Highschool / Graduate / Postgraduate
                                                     — ordinal)
  income_level                         String   (5: Low / Lower-Middle / Middle / Upper-Middle / High
                                                     — ordinal)
  smoking_status                       String   (3: Never / Former / Current)
  employment_status                    String   (4: Employed / Unemployed / Retired / Student)
  family_history_diabetes              Int64    (0/1)
  hypertension_history                 Int64    (0/1)
  cardiovascular_history               Int64    (0/1)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
education_level/income_level은 자연 순서로 ordinal 매핑하는 것이 sorted() 알파벳 순서보다 신호를
보존한다. 나머지 string은 nominal.

domain note: bmi x waist_to_hip_ratio, family_history_diabetes x age 조합이 당뇨 진단의 임상적
핵심 신호 — feature_engineering 후보."""
