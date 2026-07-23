from pathlib import Path

COMPETITION_ID    = "playground-series-s5e5"
NAME              = "Predict Calorie Expenditure"
TARGET            = "Calories"
METRIC            = "rmsle"
TASK_TYPE         = "regression"
METRIC_SIGN       = -1
IS_CLASSIFICATION = False
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s5e5/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s5e5 (Predict Calorie Expenditure)
task: regression  metric: RMSLE  target: Calories
rows: 750000  features: 7
target range: 1.0 - 314.0  mean: 88.28 (all positive — RMSLE-safe)
no missing values

feature dtypes (as seen by feature_fn):
  Sex          String   (male / female)
  Age          Int64
  Height       Float64  (cm)
  Weight       Float64  (kg)
  Duration     Float64  (minutes)
  Heart_Rate   Float64  (bpm)
  Body_Temp    Float64  (celsius)

encoding note: Sex is pl.String (NOT pl.Categorical).
detect with: dtype == pl.String  or  dtype in (pl.Utf8, pl.String)
ordinal encode: mapping = {v: i for i, v in enumerate(sorted(train[col].unique().to_list()))}
               df = df.with_columns(pl.col(col).replace_strict(mapping).cast(pl.Int32))

domain note: Calories burned correlates strongly with Duration x Heart_Rate and Body_Temp —
interaction/ratio features (e.g. Heart_Rate*Duration, Body_Temp-baseline) are natural
candidates. Since metric is RMSLE, predictions must stay non-negative — clip postprocess
output at 0 (or larger, since target min is 1.0)."""
