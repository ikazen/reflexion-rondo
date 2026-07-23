from pathlib import Path

COMPETITION_ID    = "playground-series-s4e6"
NAME              = "Academic Success (Predict Students' Dropout and Academic Success)"
TARGET            = "Target"
METRIC            = "accuracy"
TASK_TYPE         = "multiclass"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s4e6/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s4e6 (Academic Success — Dropout / Enrolled / Graduate)
task: multiclass classification (3 classes)  metric: accuracy  target: Target
rows: 76518  features: 36
target classes: Graduate 47.4% (36282) / Dropout 33.1% (25296) / Enrolled 19.5% (14940) —
  중간 불균형, accuracy 지표 채택(과도한 편중 아님)
no missing values

feature dtypes (as seen by feature_fn): 전 feature가 이미 정수/실수 인코딩된 상태 — string 컬럼
없음, encoding 로직 불필요. 주요 컬럼:
  Marital status, Application mode, Application order, Course,
  Daytime/evening attendance, Previous qualification,
  Previous qualification (grade), Nacionality,
  Mother's qualification, Father's qualification,
  Mother's occupation, Father's occupation, Admission grade,
  Displaced, Educational special needs, Debtor,
  Tuition fees up to date, Gender, Scholarship holder,
  Age at enrollment, International                              모두 Int64/Float64
  Curricular units 1st/2nd sem (credited/enrolled/evaluations/
    approved/grade/without evaluations)                          Int64/Float64 (학기별 학업 성과)
  Unemployment rate, Inflation rate, GDP                          Float64 (거시경제 지표)

encoding note: string 컬럼 없음 — feature_transform은 target 인코딩(Graduate/Dropout/Enrolled →
정수 라벨)만 하면 충분. 컬럼명에 공백/괄호/따옴표 포함(예: "Mother's qualification",
"Curricular units 1st sem (approved)") — pl.col(...)로 정확히 인용 필요.

domain note: Curricular units 1st/2nd sem (approved) x (enrolled) 비율(이수율)이 자퇴/졸업 판별의
핵심 신호로 알려져 있음(UCI 원본 데이터셋 분석 다수 사례) — feature_engineering 후보. Admission
grade x Previous qualification (grade) 조합도 입학 성적 기반 예측력 후보."""
