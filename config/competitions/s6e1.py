from pathlib import Path

COMPETITION_ID    = "playground-series-s6e1"
NAME              = "Predicting Student Test Scores"
TARGET            = "exam_score"
METRIC            = "rmse"
TASK_TYPE         = "regression"
METRIC_SIGN       = -1
IS_CLASSIFICATION = False
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s6e1/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # BON-250: 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s6e1 (Predicting Student Test Scores)
task: regression  metric: RMSE  target: exam_score
rows: 630000  features: 12
target range: 19.599 - 100.0  mean: 62.51
no missing values

feature dtypes (as seen by feature_fn):
  age                Int64
  gender             String   (male / female / other)
  course             String   (7 categories: b.com / b.sc / b.tech / ba / bba / bca / diploma)
  study_hours        Float64
  class_attendance   Float64
  internet_access    String   (yes / no)
  sleep_hours        Float64
  sleep_quality      String   (poor / average / good — ordinal)
  study_method       String   (5 categories: coaching / group study / mixed / online videos / self-study)
  facility_rating    String   (low / medium / high — ordinal)
  exam_difficulty    String   (easy / moderate / hard — ordinal)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String
  or dtype in (pl.Utf8, pl.String).
ordinal 컬럼(sleep_quality/facility_rating/exam_difficulty)은 자연 순서로 명시 매핑하는 것이
sorted() 알파벳 순서보다 신호를 보존한다(예: low/medium/high, easy/moderate/hard).
nominal 컬럼(gender/course/internet_access/study_method)은 알파벳 순 매핑으로 충분.

domain note: study_hours x class_attendance, sleep_hours x sleep_quality 같은 상호작용이
exam_score의 직관적 원인 조합 — feature_engineering 후보로 자연스럽다. 타깃 범위가
19.6~100으로 skew가 강하지 않아 log 변환 없이 raw scale RMSE로 학습하는 것이 기본."""
