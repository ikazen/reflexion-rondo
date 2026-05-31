# 명세 (스키마·dbt·API·컨트랙트)

단일 스토어 원칙: DuckDB 하나가 기록·검색·분석을 모두 맡는다 (임베딩은 벡터 컬럼, 검색은 브루트포스 코사인). 결정 근거는 `decisions.md` ADR-007.

## 1. DB 스키마 (DuckDB `raw`)

### 1.1 `raw.competitions`

```sql
competition_id  text primary key,
name            text,
task_type       text,        -- binary / multiclass / regression
metric          text,        -- auc / logloss / rmse / ...
metric_sign     int,         -- +1 높을수록 좋음, -1 낮을수록 좋음
start_ts        timestamp,
fingerprint     json         -- §1.4
```

`metric_sign`은 대회당 1회 결정. attempts에 중복 저장하지 않는다.

### 1.2 `raw.attempts`

```sql
attempt_id        text primary key,
competition_id    text,
run_ts            timestamp,
stage             text,        -- bootstrap / reflexion / exploitation
hypothesis        text,
action_type       text,        -- §3 enum
reflection_ids    text[],      -- Strategist가 실제 채택한 교훈 id
retrieval_scores  double[],
model_type        text,
params            json,
features          json,
cv_score          double,
cv_fold_var       double,      -- fold별 분산 (overfit·노이즈 감지)
lb_score          double,      -- 미제출 시 null
label             text,        -- jump / neutral / regression (Evaluator 결정, §4)
error_trace       text         -- 실패 시
```

### 1.3 `raw.reflections`

```sql
reflection_id   text primary key,
created_at      timestamp,
attempt_id      text,
competition_id  text,
embedded_text   text,           -- 검색 결과를 사람이 읽는 원문
embedding       float[1024],    -- qwen3-embedding:0.6b(embedded_text). 검색용 벡터 컬럼 (MRL로 차원 절단 가능)
full_lesson     text,
generality      text,           -- L1_local / L2_class / L3_general (Reflector)
label           text,           -- 결정적 진실값 (attempts.label 복제, 마트·검색용)
reflector_label text,           -- LLM 정성 판정 (참고용, 진실값 아님) — ADR-012
gain_vs_best    double,         -- Evaluator 계산
archived        boolean default false
```

### 1.4 `competition.fingerprint` (결정적 메타피처)

```text
n_rows, n_cols,
task_type, metric, metric_sign,
n_numeric, n_categorical, n_datetime, n_text_ish,
missing_ratio_overall,
cardinality_max, cardinality_mean,
target_balance (분류: minority class ratio) or target_skew (회귀),
size_class ('tiny' <10k / 'small' <100k / 'mid' <1M / 'large')
```

타깃 통계는 train fold 평균/분산으로만 계산 (test 누수 방지).

### 1.5 `raw.pipelines` (코드 메모리)

```sql
pipeline_id          text primary key,
attempt_id           text,
competition_id       text,
fingerprint_snapshot json,
code                 text,        -- feature_fn + model_fn 소스 (§5)
cv_score             double,
gain_vs_best         double
```

cold-start 시 유사 fingerprint에서 검색해 그대로 baseline으로 재사용.

### 1.6 벡터 검색 (DuckDB, 별도 스토어 없음)

별도 벡터DB 없이 `raw.reflections.embedding`(§1.3) 컬럼을 직접 검색한다 (ADR-007).

```sql
-- 메타필터로 후보를 좁힌 뒤 브루트포스 코사인 + 재순위
select r.reflection_id, r.full_lesson, r.generality,
       array_cosine_similarity(r.embedding, $query_vec) as sim,
       -- 재순위: 효과 좋은 교훈 가중 (reflection_impact 마트 LEFT JOIN)
       sim * (1 + greatest(-$k, least($k, coalesce(i.avg_gain, 0)))) as score
from raw.reflections r
left join reflection_impact i using (reflection_id)
where r.archived = false
  and (r.generality in ('L2_class','L3_general')
       or r.competition_id = $competition_id)        -- cold-start 메타필터(architecture §7)
order by score desc
limit $k;
```

검색 결과의 `full_lesson`/`embedded_text`를 사람이 그대로 읽어 디버깅한다(`runbook.md §6`).
규모가 커져 브루트포스가 느려지면 `vss` 확장으로 HNSW 인덱스만 추가 (ADR-007 승격 트리거).

### 1.7 `submission_budget`

```sql
competition_id  text,
day             date,
count           int          -- 일별 제출 수. Submit 단계가 SELECT-then-UPDATE
```

## 2. enum 정의

`action_type` (Strategist 출력 강제):
- `feature_engineering` (target encoding, interaction, binning, …)
- `model_swap` (lgbm ↔ catboost ↔ xgboost ↔ tabpfn)
- `hyperparam_search` (Optuna 트라이얼 수/공간 지정)
- `cv_strategy` (kfold ↔ stratified ↔ group ↔ time)
- `preprocessing` (결측 처리, 스케일, 인코딩)
- `ensemble` (averaging, stacking)

`generality` (Reflector 출력):
- `L1_local`: 이 대회 칼럼명/특이값 의존. transfer 대상 아님.
- `L2_class`: 비슷한 fingerprint 부류에서 통하는 패턴.
- `L3_general`: 정형 대회 보편 원칙.

`label` (Evaluator 결정, §4): `jump` / `neutral` / `regression`.

## 3. 지표 레지스트리

Evaluator는 `metric` 텍스트를 `(callable, sign)`으로 매핑하는 레지스트리를 둔다. 미등록 지표는 대회 등록 시 거부.

| metric | metric_class | sign | 상태 |
|---|---|---|---|
| auc (roc_auc) | binary_proba | +1 | 구현 |
| logloss | binary_proba | -1 | 구현 |
| accuracy | classification | +1 | 구현 |
| f1 | classification | +1 | 구현 |
| mcc | classification | +1 | TBD |
| rmse | regression_error | -1 | 구현 |
| mae | regression_error | -1 | 구현 |
| qwk (quadratic weighted kappa) | ordinal | +1 | TBD |
| map@k | ranking | +1 | TBD |

`metric_class`는 transfer 유사도에서 부류 불일치 페널티에 쓰인다 (`architecture.md` §7).

## 4. label·gain 결정 규칙 (Evaluator, 결정적)

fold별 점수의 표준편차 `fold_std = sqrt(cv_fold_var)`가 측정 노이즈. 직전 best 대비 부호 정렬 델타:

```text
delta = metric_sign * (cv_score - prev_best_cv)
gain_vs_best = delta

jump        if delta >  z * fold_std
regression  if delta < -z * fold_std
neutral     otherwise
```

- `z`: fold_std 배수. 기본 1.0, 캘리브레이션 대상(`decisions.md` TBD). 더 보수적으로 `fold_std/sqrt(k)` (표준오차)를 쓸 수 있음.
- `prev_best_cv`가 없으면(첫 attempt) label = `neutral`, gain = null.
- 실패 attempt(`error_trace` 존재)는 label = `regression`, gain = null.

Reflector는 이 숫자를 보고 **왜 그런 결과가 나왔는지**(교훈 본문)만 쓴다. 정성 판정은 `reflector_label`로 분리.

## 5. Coder 컨트랙트 (feature_fn + model_fn)

Coder는 두 함수만 생성한다. IO·k-fold 분할·시드·CV 루프는 Evaluator가 소유하므로 Coder 코드에 등장하지 않는다.

```python
# 누수 방지: 통계는 train에서만 학습하고 valid엔 적용만 한다.
# Evaluator가 매 fold마다 (train_fold, valid_fold)로 호출한다.
def build_features(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    target: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ...

# sklearn 호환 estimator(fit / predict[ _proba ]) 반환.
def build_model(params: dict) -> Estimator:
    ...
```

Evaluator 실행 골격(개념):

```text
for tr_idx, va_idx in folds(seed):
    Xtr, Xva = build_features(train[tr_idx], train[va_idx], target)
    m = build_model(params); m.fit(Xtr, ytr)
    preds = m.predict(Xva); score = metric(yva, preds)
cv_score = mean(scores); cv_fold_var = var(scores)
```

검증 게이트(실행 전): 시그니처 일치, 허용 import만, 금지 호출(파일/네트워크/`eval`) 없음. 위반 시 재생성 1회, 실패면 `error_trace` 기록 후 Reflect로 진행.

격리 실행: `runtime/`가 컨테이너/nsjail에서 위 골격을 돌린다 (ADR-013, `runbook.md`).

## 6. dbt 모델

`stg_attempts`: `raw.attempts` ⨝ `raw.competitions`로 `metric_sign` 노출.
인과 귀속용 view `stg_attempts_reflexion_only`는 `stage='reflexion'` 필터.

### 6.1 `score_progression.sql` (대회 내 진보)

```sql
select
  competition_id,
  row_number() over (partition by competition_id order by run_ts) as attempt_no,
  run_ts, cv_score, lb_score,
  max(metric_sign * cv_score) over (
    partition by competition_id
    order by run_ts rows between unbounded preceding and current row
  ) * metric_sign as best_so_far
from {{ ref('stg_attempts') }}
```

### 6.2 `reflection_impact.sql` (어떤 반성이 점프를 만들었나)

```sql
with scored as (
  select
    competition_id, reflection_ids,
    metric_sign * cv_score
    - max(metric_sign * cv_score) over (
        partition by competition_id
        order by run_ts rows between unbounded preceding and 1 preceding
      ) as gain_vs_best
  from {{ ref('stg_attempts_reflexion_only') }}
),
per_reflection as (
  select unnest(reflection_ids) as reflection_id, gain_vs_best
  from scored where reflection_ids is not null
)
select
  reflection_id,
  count(*)                                          as times_applied,
  round(avg(gain_vs_best), 5)                       as avg_gain,
  sum(case when gain_vs_best > 0 then 1 else 0 end) as jumps,
  round(max(gain_vs_best), 5)                       as best_jump
from per_reflection
group by reflection_id
order by avg_gain desc
```

### 6.3 `cold_start_progression.sql` (transfer 효과 측정)

대회별 bootstrap 첫 시도의 "그 대회 최종 best 대비 비율"이 누적 경험과 함께 어떻게 변하는가.

```sql
with first_boot as (
  select competition_id,
         arg_min(cv_score, run_ts) as first_cv,
         min(run_ts)               as first_ts
  from {{ source('raw', 'attempts') }}
  where stage = 'bootstrap'
  group by competition_id
),
best_per_comp as (
  select competition_id,
         max(metric_sign * cv_score) * metric_sign as best_cv,
         any_value(metric_sign)                    as metric_sign
  from {{ ref('stg_attempts') }}
  group by competition_id, metric_sign
)
select
  row_number() over (order by f.first_ts) as competition_no,
  f.competition_id, f.first_ts,
  f.first_cv, b.best_cv,
  (b.metric_sign * f.first_cv) / nullif(b.metric_sign * b.best_cv, 0) as warm_start_ratio
from first_boot f
join best_per_comp b using (competition_id)
order by f.first_ts
```

`warm_start_ratio`가 대회 번호에 따라 우상향하면 transfer 작동. 아니면 fingerprint/generality 튜닝 필요.

### 6.4 디버깅 조인 (효과 좋은 교훈 전문 보기)

```sql
select r.reflection_id, r.full_lesson, r.generality, i.avg_gain, i.jumps
from {{ ref('reflection_impact') }} i
join {{ source('raw', 'reflections') }} r using (reflection_id)
where r.archived = false
order by i.avg_gain desc
limit 20
```

## 7. Ollama Cloud 연동

드롭인: 로컬과 동일 인터페이스. host를 `https://ollama.com`로 두고 `OLLAMA_API_KEY`만 설정.

```python
import os
from ollama import Client

cloud = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
)
resp = cloud.chat(
    model="deepseek-v4-pro",           # Strategist. Reflector=glm-5, Coder=qwen3-coder-next (ADR-016)
    messages=[...],
    format=hypothesis_schema_dict,     # ollama-python: JSON Schema dict 허용
)

# 임베딩은 로컬 (클라우드 키가 /api/embed 미인가일 수 있음)
local = Client(host="http://localhost:11434")
local.embed(model="qwen3-embedding:0.6b", input=note_text)   # 1024d
```

비용 통제 레버:
1. 모델 배정(ADR-016): Strategist/Reflector는 추론 모델, Coder는 코드 특화 모델.
2. 구조화 출력(JSON Schema): 디코딩 단계 검증으로 재시도 제거.
3. 컨텍스트 캐싱: EDA 카드·교훈 패키지를 안정 유지해 캐시 활용.
4. 클라우드 밖은 클라우드 밖에: 임베딩·CV·dbt는 로컬.
5. 도구 호출은 native `/api/chat`. (원격 `/v1`은 tool calling이 깨질 수 있음)
