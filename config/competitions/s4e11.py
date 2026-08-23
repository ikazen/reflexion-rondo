"""Kaggle Playground s4e11 — Exploring Mental Health (Depression) 대회 config (metric=accuracy).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s4e11"
NAME              = "Exploring Mental Health (Depression)"
TARGET            = "Depression"
METRIC            = "accuracy"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id", "Name"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s4e11/data/"
EXTRA_TRAIN_PATHS: list[str] = ["original.csv"]  # hopesb/student-depression-dataset —
# 17/19 컬럼 일치(Name/Working Professional or Student 없음, 학생 전용 서브셋이라
# store/train_data.py의 컬럼 교집합+null 채움으로 자동 처리). MinIO kaggle/s4e11/data/original.csv.
ACTIVE            = True  # False면 daemon 큐 리필(_sweep_queue_refill) 대상 제외 (#227, Milestone v1.6.0)

EDA_CARD = """competition: playground-series-s4e11 (Exploring Mental Health — Depression)
task: binary classification  metric: accuracy  target: Depression
rows: 140700  features: 17 (Name 제외 — 422종 개인명, 예측에 무의미해 DROP_COLS 처리)
target rate: 18.2% Depression=1 / 81.8% =0 (minority=1, imbalanced 상황에서 accuracy 지표 — 대다수
  예측(0)만 해도 82% 근접, 실질 판별력 확보에 유의)
결측: Profession 26.0%, Academic Pressure 80.2%, Work Pressure 19.8%, CGPA 80.2%,
  Study Satisfaction 80.2%, Job Satisfaction 19.8%, Dietary Habits/Degree/Financial Stress <0.1%

feature dtypes (as seen by feature_fn):
  Gender                                    String  (2: Male / Female)
  Age                                       Float64
  City                                      String  (98개 — 값 오염 있음, 아래 데이터 품질 노트 참조)
  Working Professional or Student           String  (2: Student / Working Professional — 아래 결측
                                                          구조를 지배하는 분기 컬럼)
  Profession                                String  (65개 카테고리)
  Academic Pressure                         Float64 (Student만 값 존재, Professional은 결측)
  Work Pressure                             Float64 (Professional만 값 존재, Student는 결측)
  CGPA                                      Float64 (Student만 값 존재)
  Study Satisfaction                        Float64 (Student만 값 존재)
  Job Satisfaction                          Float64 (Professional만 값 존재)
  Sleep Duration                            String  (36개 — 값 오염 있음, 아래 참조)
  Dietary Habits                            String  (24개 — 값 오염 있음, 아래 참조)
  Degree                                    String  (116개 — 값 오염 있음, 아래 참조)
  Have you ever had suicidal thoughts ?     String  (2: Yes / No)
  Work/Study Hours                          Float64
  Financial Stress                          Float64
  Family History of Mental Illness          String  (2: Yes / No)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
컬럼명에 공백/물음표/슬래시 포함 — pl.col("Have you ever had suicidal thoughts ?") 형태로 정확히
인용해야 함.

데이터 품질 노트(중요): City/Sleep Duration/Dietary Habits/Degree 컬럼에 명백한 오염값이 섞여
있음 — 예: City에 'No'/'MSc'/'3.0' 같은 다른 컬럼의 값이나 학위명이 섞여 있고, Sleep Duration에
'Work_Study_Hours'/'Sleep_Duration' 같은 컬럼명 문자열 자체가 값으로 들어간 행이 존재. 이런
컬럼을 카테고리로 그대로 학습에 쓰면 노이즈 카테고리가 대량 생성되므로, 알려진 유효값 집합
바깥의 값은 별도 "invalid/other" 카테고리로 묶는 정제가 필요.

결측 구조 노트: Academic Pressure/CGPA/Study Satisfaction ↔ Work Pressure/Job Satisfaction은
"Working Professional or Student" 값에 따라 상호 배타적으로 결측(학생 컬럼은 직장인 행에서 결측,
그 반대도 마찬가지) — MCAR이 아니라 구조적 결측이므로 그룹별 대체 또는 "student_"/"prof_" 접두
파생 신호로 활용 권장.

domain note: Academic/Work Pressure x Financial Stress x Have you ever had suicidal thoughts?
조합이 우울증 판별의 핵심 임상 신호 후보."""
