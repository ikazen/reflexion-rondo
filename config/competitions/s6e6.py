from pathlib import Path

COMPETITION_ID    = "playground-series-s6e6"
NAME              = "Predicting Stellar Class"
TARGET            = "class"
METRIC            = "balanced_accuracy"
TASK_TYPE         = "multiclass"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s6e6/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s6e6 (Predicting Stellar Class)
task: multiclass classification (3 classes)  metric: balanced_accuracy  target: class
rows: 577347  features: 10
target classes: GALAXY 377480 (65.4%) / QSO 117143 (20.3%) / STAR 82724 (14.3%) — imbalanced
no missing values

feature dtypes (as seen by feature_fn):
  alpha              Float64  (right ascension, degrees)
  delta              Float64  (declination, degrees)
  u                  Float64  (ultraviolet photometric filter magnitude)
  g                  Float64  (green photometric filter magnitude)
  r                  Float64  (red photometric filter magnitude)
  i                  Float64  (near-infrared photometric filter magnitude)
  z                  Float64  (infrared photometric filter magnitude)
  redshift           Float64  (spectroscopic redshift, key discriminator for QSO)
  spectral_type      String   (A/F, M, G/K, O/B — 4 categories)
  galaxy_population  String   (Red_Sequence, Blue_Cloud — 2 categories)

encoding note: spectral_type and galaxy_population are pl.String (NOT pl.Categorical).
detect with: dtype == pl.String  or  dtype in (pl.Utf8, pl.String)
ordinal encode: mapping = {v: i for i, v in enumerate(sorted(train[col].unique().to_list()))}
               df = df.with_columns(pl.col(col).replace_strict(mapping).cast(pl.Int32))

target note: `class` is left as string labels (GALAXY/QSO/STAR) — sklearn classifiers and
balanced_accuracy_score accept string class labels natively, no target encoding needed.

domain note: u/g/r/i/z are SDSS photometric magnitudes — ratios/differences between bands
(e.g. u-g, g-r) are standard astronomical color features and often carry strong signal.
redshift is the single strongest discriminator for QSO (quasars are typically high-redshift)."""
