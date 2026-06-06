from pathlib import Path

COMPETITION_ID   = "playground-series-s4e1"
NAME             = "Bank Customer Churn Prediction"
TARGET           = "Exited"
METRIC           = "auc"
TASK_TYPE        = "binary_classification"
METRIC_SIGN      = 1
IS_CLASSIFICATION = True
DROP_COLS        = ["id", "CustomerId", "Surname"]
DATA_DIR         = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
DATA_URL         = "http://minio.internal/kaggle/s4e1/data/"

EDA_CARD = """competition: playground-series-s4e1 (Bank Churn)
task: binary classification  metric: AUC  target: Exited
rows: 165034  features: 10
target rate: 21.2% (mild imbalance)  no missing values

feature dtypes (as seen by feature_fn):
  Geography       String  (France / Germany / Spain)
  Gender          String  (Male / Female)
  CreditScore     Int64
  Age             Float64
  Tenure          Int64
  Balance         Float64
  NumOfProducts   Int64
  HasCrCard       Float64
  IsActiveMember  Float64
  EstimatedSalary Float64

encoding note: Geography and Gender are pl.String (NOT pl.Categorical).
detect with: dtype == pl.String  or  dtype in (pl.Utf8, pl.String)
ordinal encode: mapping = {v: i for i, v in enumerate(sorted(train[col].unique().to_list()))}
               df = df.with_columns(pl.col(col).replace_strict(mapping).cast(pl.Int32))"""
