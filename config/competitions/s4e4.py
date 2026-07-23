from pathlib import Path

COMPETITION_ID    = "playground-series-s4e4"
NAME              = "Abalone Age Prediction (Rings)"
TARGET            = "Rings"
METRIC            = "rmsle"
TASK_TYPE         = "regression"
METRIC_SIGN       = -1
IS_CLASSIFICATION = False
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s4e4/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s4e4 (Abalone Age Prediction — Rings)
task: regression  metric: RMSLE  target: Rings (전복 나이 proxy, 정수 1~29)
rows: 90615  features: 8
target range: 1 - 29  mean: 9.70  nunique: 28 (사실상 카운트형 정수 타깃)
no missing values

feature dtypes (as seen by feature_fn):
  Sex             String   (3: M / F / I(유체, immature) — nominal)
  Length          Float64
  Diameter        Float64
  Height          Float64
  Whole weight    Float64  (컬럼명에 공백 포함 — pl.col("Whole weight")로 접근)
  Whole weight.1  Float64  (원본 UCI 스키마의 Shucked weight — polars read_csv 중복 컬럼명 자동
                                리네이밍으로 추정, 실제 의미는 컬럼 순서로 추론 필요)
  Whole weight.2  Float64  (원본 UCI 스키마의 Viscera weight로 추정 — 위와 동일 사유)
  Shell weight    Float64

encoding note: Sex는 pl.String(NOT pl.Categorical), nominal 3종 — 알파벳 순 매핑으로 충분.
컬럼명이 "Whole weight", "Whole weight.1", "Whole weight.2", "Shell weight"로 저장돼 있어
원본 UCI abalone 스키마(Whole/Shucked/Viscera/Shell weight)와 이름이 다르게 매핑됨 — 컬럼명
그대로 pl.col()에 사용하고 임의로 재해석하지 말 것.

domain note: RMSLE는 절대 오차보다 상대 오차에 민감 — target이 정수 카운트형이므로 log1p 변환
학습이 RMSLE 최적화와 자연스럽게 정합. Length x Diameter x Height(부피 근사), weight 4종의
비율(Shell/Whole 등)이 나이 추정의 생물학적 핵심 신호 — feature_engineering 후보."""
