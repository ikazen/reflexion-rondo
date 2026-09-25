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
EXTRA_TRAIN_PATHS: list[str] = ["original.csv"]  # sangampaudel530/original-podcast-dataset —
# 컬럼 완전 일치. MinIO kaggle/s5e4/data/original.csv.
ACTIVE            = True  # deep tier 재활성 (#332, ADR-051) — s5e2 동결로 빈 회귀 트랙 슬롯
# 교체, SNR 100.8(fleet 2위의 3배)·gap_to_p90 fleet 최대.

# 2026-09 재활성 1일 실측(#340, ADR-052): attempt 70%가 CPU 예산(3600s) kill, fleet
# CPU의 65%(89h 중 57.5h)를 이 대회 하나가 소각. ADR-044(예산 3600→10800 상향)는 이미
# s5e4에서 실패로 결론났다(킬 비율 그대로, 소각량만 3배) — 남은 개입 축은 예산이 아니라
# attempt 단가다. MAX_TRAIN_ROWS 500k→150k + N_SPLITS 5(기본)→3으로 1회 eval 비용을
# 줄인다. 데이터/fold 구조가 바뀌므로 배포 후 bin/establish_baseline.py --remeasure
# 필수(train_fingerprint 가드가 재측정 전까지 attempt를 막는다).
#
# 2026-08 처리량 진단(#135): 당시 최근 7일 rc=-9(OOM SIGKILL) 140/450건(31%) — 500k행
# 자체가 이미 무거웠다는 선행 신호였다.
MAX_TRAIN_ROWS = 150_000
N_SPLITS = 3

# 제출 CSV는 MAX_TRAIN_ROWS 축소 없이 전량(약 79.7만 행)으로 학습한다. 제출 fit에는 CV가 없어 축소할 이유가 없고, LB는 학습 행수를
# 그대로 따른다: 같은 계열 pipeline의 캡 제출 LB 13.076(백분위 46.7) -> 전량 12.834(65.5) (2026-09-25 A/B, #355, ADR-055).
SUBMIT_FULL_DATA = True
# 제출 CSV fit의 seed 수. stack ensemble은 seed마다 멤버당 inner 5-fold + 최종 fit이라, 전량 1-seed가 로컬 CPU 29.5분,
# 예전 캡 5-seed가 45.3분이었고 promote task에서는 78~125분이 걸렸다. LB 이득은 seed 평균이 아니라 학습 행수에서 나온다.
SUBMIT_BAG_SEEDS = [42]

EDA_CARD = """competition: playground-series-s5e4 (Podcast Listening Time Prediction)
task: regression  metric: RMSE  target: Listening_Time_minutes
rows: ~150000 (MAX_TRAIN_ROWS로 랜덤 샘플링 — 원본 750000행에서 CPU 예산 초과 방지, #340)  features: 10
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
