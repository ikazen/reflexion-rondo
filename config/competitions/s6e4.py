"""Kaggle Playground s6e4 — Irrigation Need Prediction 대회 config (metric=balanced_accuracy).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
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
ACTIVE            = True  # deep tier 두 번째 슬롯 (#544, ADR-079) — s6e6 포화로 추가. 7일 안에 확정 pipeline 0건이면 동결
FUTILITY_STOP     = True  # 365건 리플레이: fold-1 대 최종 delta 상관 0.9996, 임계 0.75 적용 시 78% 중단에 양성 8건 손실 0 (#544, ADR-076)
FOLD1_REUSE_DISCRETE = True  # fold-1 일치 203건 중 202건이 5 fold 전체 동일 (#544, ADR-061)
# ensemble 92건 중 양의 gain 0, 에러 28 (#555, ADR-080). model_swap은 17건뿐이고 build_model 래퍼(아래 EDA 카드)의 통로라 남긴다.
EXCLUDED_ACTIONS  = ("ensemble",)

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
클래스가 3.3%로 매우 희소하므로 class weight/oversampling 고려 대상.

measured lever — class probability multipliers (2026-10-26, real harness 5-fold, vs the current best pipeline):
  cv 0.97144 -> 0.97282 (+0.0014; every fold positive, +0.0003..+0.0021; paired SE 0.0003; same wall time), audit holdout
  0.97235 -> 0.97312 (+0.0008, 63k rows). Moving the decision
  boundary of the rare High class toward balanced_accuracy is a larger lever on this competition than any model size or
  hyperparameter change tried so far (those are mostly <= +0.0003). Bigger lgbm/xgboost and blends converge to the same
  level once multipliers are applied (0.9720..0.9730 on one local fold), so do not spend attempts on model size.
  Implement it as a build_model wrapper (any action_type may implement build_model; model_spec cannot express it). CV, holdout
  and submission all go through fit/predict, so all three paths get the same model. fit holds out 10% of the rows, trains on the
  other 90%, coordinate-searches one multiplier per class on the held-out probabilities, and predict returns argmax(proba * mult):

    class ClassMultLGBM:
        def __init__(self, params):
            self.params = dict(params)

        def get_params(self, deep=True):
            return {"params": self.params}

        def fit(self, X, y):
            from lightgbm import LGBMClassifier
            X, y = np.asarray(X), np.asarray(y)
            order = np.random.default_rng(self.params.get("random_state", 42)).permutation(len(y))
            cal, fit = order[: len(y) // 10], order[len(y) // 10:]
            self.model_ = LGBMClassifier(**self.params).fit(X[fit], y[fit])
            self.classes_ = self.model_.classes_
            yc = np.searchsorted(self.classes_, y[cal])
            proba = self.model_.predict_proba(X[cal])
            k = len(self.classes_)

            def score(w):
                cm = np.bincount(yc * k + (proba * w).argmax(1), minlength=k * k).reshape(k, k)
                return float(np.mean(np.diag(cm) / np.maximum(cm.sum(1), 1)))

            w, best = np.ones(k), score(np.ones(k))
            for _ in range(3):
                for c in range(k):
                    for m in np.exp(np.linspace(-2, 2, 41)):
                        w2 = w.copy()
                        w2[c] = m
                        s = score(w2)
                        if s > best:
                            best, w = s, w2
            self.mult_ = w
            return self

        def predict(self, X):
            return self.classes_[(self.model_.predict_proba(np.asarray(X)) * self.mult_).argmax(1)]

  build_model returns ClassMultLGBM(params) with the current best's LightGBM params (class_weight='balanced' stays in params).
  If the current best already uses a multiplier wrapper, do not propose it again; improve on top of it instead."""
