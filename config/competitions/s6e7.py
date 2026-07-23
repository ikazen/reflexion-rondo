from pathlib import Path

COMPETITION_ID    = "playground-series-s6e7"
NAME              = "Predicting Student Health Risk"
TARGET            = "health_condition"
METRIC            = "balanced_accuracy"
TASK_TYPE         = "multiclass"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s6e7/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s6e7 (Predicting Student Health Risk)
task: multiclass classification (3 classes)  metric: balanced_accuracy  target: health_condition
rows: 690088  features: 13
target classes: at-risk 592561 (85.9%) / unhealthy 57724 (8.4%) / fit 39803 (5.8%) — heavily imbalanced,
  plain accuracy would be near-trivial by always predicting at-risk — balanced_accuracy chosen for this reason
missing values present (feature_fn must handle): sleep_duration 11.0%, stress_level 12.0%, sleep_quality 8.5%,
  calorie_expenditure 7.7%, water_intake 6.3%, physical_activity_level 5.3%, smoking_alcohol 4.1%,
  diet_type 4.1%, gender 3.1%, step_count/bmi/heart_rate/exercise_duration <2%

feature dtypes (as seen by feature_fn):
  sleep_duration             Float64  (hours, has nulls)
  heart_rate                 Float64  (bpm, has nulls)
  bmi                        Float64  (has nulls)
  calorie_expenditure        Float64  (has nulls)
  step_count                 Float64  (has nulls)
  exercise_duration          Float64  (minutes, has nulls)
  water_intake                Float64  (has nulls)
  diet_type                  String   (veg / non-veg / balanced, has nulls)
  stress_level                String   (low / medium / high — ordinal, has nulls)
  sleep_quality               String   (poor / average / good — ordinal, has nulls)
  physical_activity_level     String   (sedentary / moderate / active — ordinal, has nulls)
  smoking_alcohol              String   (no / occasional / yes — ordinal, has nulls)
  gender                      String   (male / female / other, has nulls)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
ordinal 컬럼(stress_level/sleep_quality/physical_activity_level/smoking_alcohol)은 자연 순서로 명시
매핑하는 것이 sorted() 알파벳 순서보다 신호를 보존한다(예: low/medium/high, poor/average/good).
null은 train 기준 median(수치)/mode 또는 별도 "missing" 카테고리(문자열)로 처리 — 어느 쪽이든
train 통계만 사용하고 valid/test에 동일 적용해야 leak 없음.

domain note: 타깃이 건강 상태(at-risk/fit/unhealthy) 분류이므로 heart_rate/bmi/sleep_duration/
stress_level/physical_activity_level 조합이 핵심 신호일 가능성이 높다."""
