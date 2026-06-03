from pathlib import Path

COMPETITION_ID    = "playground-series-s5e3"
NAME              = "Rainfall Prediction in Australia"
TARGET            = "rainfall"
METRIC            = "auc"
TASK_TYPE         = "binary_classification"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID

EDA_CARD = """competition: playground-series-s5e3 (Rainfall Prediction in Australia)
task: binary classification  metric: AUC  target: rainfall
rows: 2190  features: 11 (all numeric)
target rate: 75.3% rain / 24.7% no-rain (minority=no-rain, imbalanced)
no missing values

feature dtypes (as seen by feature_fn):
  day            Int64
  pressure       Float64
  maxtemp        Float64
  temparature    Float64
  mintemp        Float64
  dewpoint       Float64
  humidity       Float64
  cloud          Float64
  sunshine       Float64
  winddirection  Float64
  windspeed      Float64

note: all features are numeric — no encoding needed.
note: dataset is very small (2190 rows) — overfitting risk high, use regularization."""
