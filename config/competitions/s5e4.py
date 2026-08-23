"""Kaggle Playground s5e4 — Podcast Listening Time Prediction 대회 config (metric=rmse).

대회별 데이터 경로/컬럼/EDA 카드 상수만 담는다 — 로직 없음.
"""
from pathlib import Path

COMPETITION_ID    = "playground-series-s5e4"
NAME              = "Podcast Listening Time Prediction"
TARGET            = "Listening_Time_minutes"
METRIC            = "rmse"
TASK_TYPE         = "regression"
METRIC_SIGN       = -1
IS_CLASSIFICATION = False
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s5e4/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변
ACTIVE            = True  # False면 daemon 큐 리필(_sweep_queue_refill) 대상 제외 (#227, Milestone v1.6.0)

EDA_CARD = """competition: playground-series-s5e4 (Podcast Listening Time Prediction)
task: regression  metric: RMSE  target: Listening_Time_minutes
rows: 750000  features: 10
target range: 0.0 - 119.97  mean: 45.44  nunique: 42807 (범위 좁고 skew 약함 — raw scale RMSE
  학습이 기본)
결측: Episode_Length_minutes 11.6%, Guest_Popularity_percentage 19.5%, Number_of_Ads <0.1%

feature dtypes (as seen by feature_fn):
  Podcast_Name                   String   (48개 카테고리)
  Episode_Title                   String   (100개 카테고리, 예: "Episode 18" — 순번 숫자 추출 가능)
  Episode_Length_minutes            Float64 (has nulls — 11.6%, target(청취시간)의 상한과 직결되는
                                                  핵심 feature)
  Genre                           String   (10개 카테고리)
  Host_Popularity_percentage        Float64
  Publication_Day                   String   (7: Monday~Sunday — ordinal 근사 가능)
  Publication_Time                   String   (4: Morning / Afternoon / Evening / Night — ordinal)
  Guest_Popularity_percentage         Float64 (has nulls — 19.5%, 게스트 없는 에피소드일 가능성)
  Number_of_Ads                      Float64 (has nulls, 거의 없음)
  Episode_Sentiment                  String   (3: Positive / Neutral / Negative — ordinal)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
Episode_Title에서 숫자 추출(pl.col("Episode_Title").str.extract(r"(\\d+)"))이 유용한 파생 신호일
수 있음. Publication_Day/Publication_Time/Episode_Sentiment는 자연 순서 ordinal 매핑 권장.
Guest_Popularity_percentage 결측은 "게스트 없음"을 의미할 가능성이 높아 0 대체보다 별도 indicator
컬럼(has_guest) 추가를 고려.

domain note: Episode_Length_minutes는 청취시간의 물리적 상한이므로 가장 강한 단일 신호로 예상 —
Episode_Length_minutes 자체의 결측(11.6%)을 어떻게 대체하느냐가 성능에 큰 영향을 줄 수 있음."""
