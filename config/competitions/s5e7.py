"""Kaggle Playground s5e7 — Introvert vs Extrovert 대회 config (metric=accuracy).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s5e7"
NAME              = "Introvert vs Extrovert"
TARGET            = "Personality"
METRIC            = "accuracy"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s5e7/data/"
EXTRA_TRAIN_PATHS: list[str] = []
ACTIVE            = False  # False면 daemon 큐 리필(_sweep_queue_refill) 대상 제외 (#227, Milestone v1.6.0)

EDA_CARD = """competition: playground-series-s5e7 (Introvert vs Extrovert)
task: binary classification  metric: accuracy  target: Personality
rows: 18524  features: 8
target classes: Extrovert 13699 (74.0%) / Introvert 4825 (26.0%) — moderately imbalanced,
  but metric is plain accuracy (NOT balanced_accuracy) so majority-class bias in the model
  is not penalized the way it would be under balanced_accuracy.
missing values present (feature_fn must handle): Stage_fear 10.2%, Going_outside 7.9%,
  Post_frequency 6.8%, Time_spent_Alone 6.4%, Social_event_attendance 6.4%,
  Friends_circle_size 5.7%, Drained_after_socializing 6.2%

feature dtypes (as seen by feature_fn):
  Time_spent_Alone            Float64  (hours, has nulls)
  Social_event_attendance     Float64  (has nulls)
  Going_outside                Float64  (frequency, has nulls)
  Friends_circle_size           Float64  (has nulls)
  Post_frequency                Float64  (social media posting frequency, has nulls)
  Stage_fear                    String   (Yes / No, has nulls)
  Drained_after_socializing     String   (Yes / No, has nulls)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
Stage_fear/Drained_after_socializing는 순서 없는 이진 범주(Yes/No)라 label/one-hot 인코딩이면
충분하고 ordinal 매핑 불필요.
null은 train 기준 median(수치)/mode 또는 별도 "missing" 카테고리(문자열)로 처리 — train 통계만
사용하고 valid/test에 동일 적용해야 leak 없음.

domain note: 타깃이 성격 유형(내향/외향) 이진 분류이므로 Time_spent_Alone(내향 신호) /
Social_event_attendance·Going_outside·Post_frequency(외향 신호) / Drained_after_socializing이
핵심 신호일 가능성이 높다. 이 데이터도 s4e2와 마찬가지로 Kaggle Playground 합성 데이터셋이라
피처 간 결정적 규칙(threshold rule)에 가까운 관계가 있을 수 있음에 유의."""
