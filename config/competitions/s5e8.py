from pathlib import Path

COMPETITION_ID    = "playground-series-s5e8"
NAME              = "Bank Dataset (Term Deposit Subscription)"
TARGET            = "y"
METRIC            = "auc"
TASK_TYPE         = "binary"
METRIC_SIGN       = 1
IS_CLASSIFICATION = True
DROP_COLS         = ["id"]
DATA_DIR          = Path(__file__).parent.parent.parent / "data" / COMPETITION_ID
S3_DATA_PATH      = "s5e8/data/"
EXTRA_TRAIN_PATHS: list[str] = []  # 원본 Kaggle 데이터셋 병합용, 미설정 시 동작 불변

EDA_CARD = """competition: playground-series-s5e8 (Bank Dataset — Term Deposit Subscription)
task: binary classification  metric: AUC  target: y
rows: 750000  features: 16
target rate: 12.1% y=1(가입) / 87.9% y=0(미가입) (minority=1, imbalanced)
no missing values

feature dtypes (as seen by feature_fn):
  age            Int64
  job            String   (12: management / technician / admin. / blue-collar / services /
                                retired / self-employed / entrepreneur / unemployed / housemaid /
                                student / unknown)
  marital        String   (3: married / single / divorced)
  education      String   (4: secondary / tertiary / primary / unknown — ordinal 근사 가능)
  default        String   (2: yes / no)
  balance        Int64    (계좌 잔액, 음수 가능)
  housing        String   (2: yes / no)
  loan           String   (2: yes / no)
  contact        String   (3: cellular / telephone / unknown)
  day            Int64    (월중 일자)
  month          String   (12: jan~dec)
  duration       Int64    (마지막 통화 시간(초) — 결과와 강상관이나 콜 종료 후에만 알 수 있는 정보,
                                leakage 위험 있는 feature로 알려짐, 사용 시 주의)
  campaign       Int64
  pdays          Int64    (직전 캠페인 이후 경과일, -1=접촉 이력 없음)
  previous       Int64
  poutcome       String   (4: unknown / failure / other / success)

encoding note: 모든 string 컬럼은 pl.String(NOT pl.Categorical). detect with: dtype == pl.String.
education는 자연 순서(primary < secondary < tertiary, unknown 별도)로 ordinal 매핑 가능. month는
1~12 숫자 매핑이 sorted() 알파벳 순서보다 계절성 신호를 보존한다. 나머지는 nominal.

domain note: duration은 UCI bank-marketing 원본 데이터셋에서 강력한 leakage feature로 유명함
(통화가 길수록 가입 가능성 높지만 통화 종료 전에는 알 수 없는 정보) — 포함 여부/가중치 조정을
실험 대상으로 삼을 것."""
