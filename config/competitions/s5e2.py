"""Kaggle Playground s5e2 — Backpack Price Prediction 대회 config (metric=rmse).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s5e2"
NAME              = "Backpack Price Prediction"
TARGET            = "Price"
METRIC            = "rmse"
TASK_TYPE         = "regression"
METRIC_SIGN       = -1
IS_CLASSIFICATION = False
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s5e2/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변
ACTIVE            = False  # False면 daemon 큐 리필(_sweep_queue_refill) 대상 제외 (#227, Milestone v1.6.0)

EDA_CARD = """competition: playground-series-s5e2 (Backpack Price Prediction)
task: regression  metric: RMSE  target: Price
rows: 300000  features: 9
target range: 15.0 - 150.0  mean: 81.41  nunique: 48212 (범위 좁고 skew 약함 — raw scale RMSE
  학습이 기본, log 변환 불필요해 보임)
결측: Brand 3.2%, Material 2.8%, Size 2.2%, Laptop Compartment 2.5%, Waterproof 2.4%,
  Style 2.7%, Color 3.3%, Weight Capacity (kg) <0.1%

feature dtypes (as seen by feature_fn):
  Brand                  String   (6: Jansport / Under Armour / Nike / Puma / Adidas / ..., has
                                        nulls)
  Material               String   (5: Nylon / Canvas / Leather / Polyester, has nulls)
  Size                   String   (4: Small / Medium / Large — ordinal, has nulls)
  Compartments            Float64
  Laptop Compartment       String   (3: Yes / No, has nulls)
  Waterproof               String   (3: Yes / No, has nulls)
  Style                   String   (4: Tote / Messenger / Backpack, has nulls)
  Color                    String   (7: Green / Gray / Black / Pink / Red / Blue / ..., has nulls)
  Weight Capacity (kg)      Float64 (컬럼명에 공백+괄호 포함 — pl.col("Weight Capacity (kg)")로
                                          접근, has nulls)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
Size는 자연 순서(Small < Medium < Large)로 ordinal 매핑 권장, 나머지는 nominal. 모든 범주형
컬럼에 2~3% 수준의 결측이 고르게 분포 — "missing" 별도 카테고리로 처리하는 것이 무난.

domain note: Compartments x Weight Capacity (kg) x Material 조합이 가격의 직관적 핵심 신호
(수납칸 많고 하중 용량 크고 고급 소재일수록 고가) — feature_engineering 후보."""
