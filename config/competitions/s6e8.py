"""Kaggle Playground s6e8 — Predicting Smartphone Addiction 대회 config (metric=auc).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s6e8"
NAME              = "Predicting Smartphone Addiction"
TARGET            = "addicted_label"
METRIC            = "auc"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s6e8/data/"
EXTRA_TRAIN_PATHS: list[str] = ["original.csv"]  # zahranusratt/smartphone-usage-and-
# addiction-analysis-dataset — 핵심 피처 컬럼 전부 일치. transaction_id/user_id/
# addiction_level은 대회에 없는 컬럼이라 store/train_data.py의 컬럼 교집합으로 자동
# 제외됨(addiction_level은 타깃 addicted_label과 다른 파생 라벨이라 누수 없이 정확히
# 배제돼야 함 — 의도된 동작). MinIO kaggle/s6e8/data/original.csv.
ACTIVE            = True  # False면 daemon 큐 리필(_sweep_queue_refill) 대상 제외 (#227, Milestone v1.6.0)
CPU_BUDGET_SECS   = 3600  # 기본 900s 대비 4배(#176) — 9일 실측 kill률 35%, 성공 attempt
# p99=841s로 벽에 붙어 있었고 900s 위 분포는 완전히 검열돼 있었다. kill은 산출물 0에
# CPU만 소모하므로 기다려서 측정값을 받는 쪽이 항상 낫다 — 넉넉히 열어 실제 분포를
# 먼저 확보한 뒤 영구값을 정한다.

EDA_CARD = """competition: playground-series-s6e8 (Predicting Smartphone Addiction)
task: binary classification  metric: auc  target: addicted_label
rows: 691369  features: 12
target classes: 0(non-addicted) 200895 (29.1%) / 1(addicted) 490474 (70.9%) — 중간 불균형,
  auc는 임계값 무관 지표라 별도 리샘플링 불필요.

missing values present (feature_fn must handle):
  social_media_hours 19.4%, gaming_hours 18.3%, weekend_screen_time 16.2%,
  daily_screen_time_hours 13.9%, app_opens_per_day 11.7%, notifications_per_day 9.8%,
  stress_level 8.0%, work_study_hours 7.5%, academic_work_impact 6.4%, sleep_hours 6.4%,
  age 4.2%, gender 4.2%

feature dtypes (as seen by feature_fn):
  age                        Float64  (has nulls)
  daily_screen_time_hours    Float64  (has nulls)
  social_media_hours         Float64  (has nulls)
  gaming_hours               Float64  (has nulls)
  work_study_hours           Float64  (has nulls)
  sleep_hours                Float64  (has nulls)
  notifications_per_day      Float64  (has nulls)
  app_opens_per_day          Float64  (has nulls)
  weekend_screen_time        Float64  (has nulls)
  gender                     String   (Male / Female / Other, has nulls, nominal)
  stress_level               String   (Low / Medium / High — ordinal, has nulls)
  academic_work_impact       String   (Yes / No, has nulls)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). stress_level은 자연
순서(Low < Medium < High)로 명시 매핑하는 것이 알파벳 순서보다 신호를 보존한다.
gender/academic_work_impact는 순서 없는 범주 — one-hot 권장. null은 train 기준
median(수치)/mode 또는 별도 "missing" 카테고리(문자열)로 처리, train 통계만 사용해
valid/test에 동일 적용해야 leak 없음.

domain note: 스마트폰/디지털 중독 예측이므로 social_media_hours/gaming_hours/
notifications_per_day/app_opens_per_day/daily_screen_time_hours 조합과 sleep_hours의
음의 상관, stress_level과의 상호작용이 핵심 신호일 가능성이 높다."""
