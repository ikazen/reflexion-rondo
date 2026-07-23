from pathlib import Path

COMPETITION_ID    = "playground-series-s6e4"
NAME              = "Irrigation Need Prediction"
TARGET            = "Irrigation_Need"
METRIC            = "balanced_accuracy"
TASK_TYPE         = "multiclass"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s6e4/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s6e4 (Irrigation Need Prediction — Low/Medium/High)
task: multiclass classification (3 classes)  metric: balanced_accuracy  target: Irrigation_Need
rows: 630000  features: 20
target classes: Low 58.7% (369917) / Medium 37.9% (239074) / High 3.3% (21009) — heavily
  imbalanced, plain accuracy면 Low 위주 예측이 유리해지므로 balanced_accuracy 채택
no missing values

feature dtypes (as seen by feature_fn):
  Soil_Type                  String   (4: Silt / Sandy / Clay / Loamy)
  Soil_pH                    Float64
  Soil_Moisture               Float64
  Organic_Carbon               Float64
  Electrical_Conductivity      Float64
  Temperature_C                Float64
  Humidity                    Float64
  Rainfall_mm                  Float64
  Sunlight_Hours                Float64
  Wind_Speed_kmh                Float64
  Crop_Type                   String   (6: Wheat / Sugarcane / Maize / Potato / Rice / Cotton)
  Crop_Growth_Stage            String   (4: Sowing / Vegetative / Flowering / Harvest — ordinal)
  Season                      String   (3: Rabi / Kharif / Zaid)
  Irrigation_Type               String   (4: Rainfed / Sprinkler / Canal / Drip)
  Water_Source                 String   (4: Reservoir / Groundwater / River / Rainwater)
  Field_Area_hectare            Float64
  Mulching_Used                 String   (2: Yes / No)
  Previous_Irrigation_mm          Float64
  Region                       String   (5: North / South / East / West / Central)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
Crop_Growth_Stage는 생육 단계 자연 순서(Sowing < Vegetative < Flowering < Harvest)로 ordinal
매핑하는 것이 sorted() 알파벳 순서보다 신호를 보존한다. 나머지는 nominal.

domain note: Soil_Moisture x Rainfall_mm x Previous_Irrigation_mm 조합이 관개 필요도의 직관적
핵심 신호(토양 수분 낮고 최근 강수/관개 적으면 필요도 상승) — feature_engineering 후보. High
클래스가 3.3%로 매우 희소하므로 class weight/oversampling 고려 대상."""
