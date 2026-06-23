# 아키텍처

컴포넌트 배치와 데이터 흐름, 그리고 두 시스템 목표(intra-competition gain / inter-competition transfer)를
보장하는 메커니즘을 기술한다. 스키마·API 등 구체 명세는 `spec.md`, 결정 근거는 `decisions.md` 참조.

## 1. 노드 배치

| 컴포넌트 | 위치 | 비고 |
|---|---|---|
| Strategist (정책) | Ollama Cloud | deepseek-v4-pro (ADR-016) |
| Reflector (성찰) | Ollama Cloud | kimi-k2.6 (ADR-016) |
| Coder (실행) | Ollama Cloud | qwen3-coder-next (ADR-016) |
| 임베딩 | Mac Ollama 서버 | qwen3-embedding:8b (1024d, MRL) — ADR-008 |
| Evaluator (CV · 지표 · param selection · label) | WSL2 로컬 | 결정적 코드 |
| 생성 코드 실행 | ops-vm | subprocess 격리 (`runtime/isolate.py`), tmpdir + 600s timeout |
| Memory (검색) | ops-vm | Postgres + pgvector (vector(1024), <=> 코사인) |
| Orchestrator | ops-vm | daemon + `raw.cycle_queue` 폴링, Airflow DockerOperator 연동 |
| Warehouse + 분석 뷰 | ops-vm | Postgres raw 스키마 (SQL view, psycopg2 경유) |

설계 의도: **추론만 클라우드, 나머지는 로컬에서 시작.** 병목/비용이 데이터로 잡히면 그때 분산화.

## 2. 데이터 흐름

```text
              (retrieve)                                   (CV score)
 Postgres -----------------> [Strategize] -> [Generate] -> [Evaluate: k-fold]
 (pgvector검색)              (Cloud)        (Cloud)         (Local, 결정적)
     ^                                                            |
     | (lessons)                                                  v
 [Reflect] <------------ best 후보 --------------------- [Submit script] -> LB
 (Cloud) ----> Postgres(competitions/attempts/reflections[+vector]) + 분석 뷰(SQL view)
     |
     +--------------- next attempt -----------------> [Strategize]
```

외부 채널(§8, ADR-019, 미구현): Kaggle 화이트리스트 source → 주간 추출 LLM → `raw.external_ideas` 별도 게이트웨이. Strategize 가 **reflexion 단계만** 톰슨 샘플링으로 노출. reflections 풀과 검색·마트·가중치 모두 분리.

## 3. Reflexion 슈퍼사이클 (1 super-cycle = 1 retrieve + 3 parallel attempts + 1 promote)

운영 경로는 Airflow DAG `reflexion_rondo_cycle`이다. `bin/run_daemon.py`의 direct 모드는
`AIRFLOW_URL`이 없을 때 쓰는 로컬 smoke/test fallback이며, 슈퍼사이클이 아니라 단일 `run_cycle()`
attempt만 실행한다.

Airflow DAG `reflexion_rondo_cycle` 4태스크 구조:

1. **Retrieve** (`bin/run_retrieve_task.py`): 검색 키 → Postgres/pgvector 코사인 검색으로 교훈 top-k. 동시에 `action_bandit`(Beta-Bernoulli)에서 Thompson sample 1회로 3개 attempt에 서로 다른 `action_type`을 배정(`assign_super_cycle_actions`). 결과를 `raw.super_cycle_context`에 upsert.
2. **Attempt × 3** (병렬, `bin/run_attempt_task.py`): 각 attempt는 배정받은 `action_type`으로 강제 실행.
   - **Strategize**: EDA 카드 + 검색 교훈 + forced_action_type → 가설 1개. 실제 채택 교훈 id 출력.
   - **Generate**: 가설 → `class Patch` (action_type별 허용 훅만 구현). **bootstrap 외 단계는 best 코드를 `prev_code`로 받아 한 군데만 수정** (1변경 규율, §4).
   - **Evaluate**: k-fold CV + 지표 (inner holdout으로 param 사전 선정). 결정적 코드. `label`·`gain_vs_best` 계산.
   - **Persist**: `raw.attempts`에 기록 (`super_cycle_id`, `was_promoted=NULL`).
3. **Promote** (`bin/run_promote_task.py`): 3개 attempt 중 `gain_vs_best` 최대값 → winner. `was_promoted=true/false` 플래그 업데이트. Reflect 호출:
   - **winner**: jump/regression/error일 때만 reflect.
   - **loser**: neutral 포함 전부 reflect ("이 시도는 효과 없었다"도 학습 신호).
4. **Submit?**: (별도 스크립트) best 후보 → submission CSV/Kaggle. `submission_budget` 스키마는 있으나 `bin/submit.py`의 자동 예산 enforcement와 `lb_score` 기록은 아직 미구현이다.

`action_bandit`(BON-109): `reflexion` 단계 attempt 완료 시 action_type별 α/β 업데이트. jump/gain>0 → α++, regression/error → β++, neutral → 소량 양방향.
`assign_super_cycle_actions`는 super_cycle retrieve에서만 action_type을 강제 배정한다. 정상 reflexion 사이클에서 `get_action_prior`는 LLM Strategist 프롬프트에 텍스트 prior로만 제공되며, 최종 action 선택은 LLM이 한다(advisory, regret 보장 없음 — ADR-005/014).

피드백 신호 정책: **CV = 주 신호**(무제한·결정적), **LB = 확인용 희소 신호**. 일일 제출 예산 게이트와 CV-LB 상관/shake 기록은 운영 목표이며, 현재 submit 경로에는 자동화되어 있지 않다.

## 4. Stage 라벨과 1변경 규율

| stage | 의미 | 1변경 규율 |
|---|---|---|
| `bootstrap` | 새 대회 진입 직후 N=3~5회. warm-start 시드 + 안전 베이스라인 정렬 | 예외 (큰 변경 허용) |
| `reflexion` | 정상 루프. 시도당 1변경 | 강제 |
| `exploitation` | best 후보 안정화 (시드 변경, 앙상블 결정 등) | 예외 |

1변경 규율은 문서 규약이 아니라 코드로 강제된다: `reflexion`/`exploitation` 단계는 best(에러 없는) attempt의 저장 코드(`code_path`)를 `prev_code`로 Coder에 주입하고 "한 군데만 수정"을 지시한다. bootstrap만 면제(prev_code 없음 → from-scratch).

`reflection_impact`는 `reflexion` 단계 attempt만 집계한다 (`stg_attempts`에서 필터). 인과 귀속을 깨끗하게 유지.

> **귀속의 한계 (의도적 단순화):** 현재 시스템은 retrieval ON/OFF ablation 없이 `reflection_impact` **상관**으로
> 교훈 효과를 추정한다. "교훈 없이 시도만 늘려도 CV가 오르는가"라는 귀무가설을 엄밀히 배제하지 않는다.
> 1변경 규율 + 실제 채택 교훈 id 기록이 상관의 신뢰도를 높이지만 인과 증명은 아니다.
> 신호가 모호하면 ablation 도입을 후속 과제로 둔다 (Linear BON-22, 트리거 ADR-015).

## 5. 생성 코드 격리 실행

Coder가 만든 `class Patch`는 `runtime/isolate.py`가 subprocess로 실행한다 (ADR-013).
- tmpdir에 source.py / input.json / train.parquet 기록 → `runtime/runner.py` subprocess 실행 → output.json 수거.
- 타임아웃 600s. 에러·타임아웃은 `error_trace`로 기록되어 Reflector가 실패에서 교훈을 뽑는다.
- subprocess 환경변수는 allowlist 필터링 (`OMP_NUM_THREADS` 등 포함, BON-104).
- `OMP_NUM_THREADS=2` / `OPENBLAS_NUM_THREADS=2` / `MKL_NUM_THREADS=2` — worker-vm 2코어에서 CPU 포화 방지.

## 6. 컴포넌트와 역할

LLM 역할 3개:

| 역할 | 입력 → 출력 |
|---|---|
| Strategist | EDA 카드 + 검색 교훈 + stage → 가설 1개 + `action_type` + 채택 교훈 id |
| Coder | 가설 + 파이프라인 컨트랙트 → `class Patch` (action_type별 허용 훅) |
| Reflector | (가설, 코드, retrieved_ids, CV 결과, best 대비 델타, feature_importance, fold var, 에러 trace) → 교훈 본문 + `generality` + (참고) `reflector_label` |

비-LLM 컴포넌트:
- **Evaluator**: k-fold CV + 지표 + inner holdout param selection 캡슐화. 결정적 시드. `label`·`gain_vs_best` 계산.
- **Memory/Retriever**: Postgres/pgvector `vector(1024)` 컬럼 + 임베딩 + 메타필터 + MMR 재순위 (코사인 `<=>`, BON-98).
- **Fingerprinter**: 결정적 메타피처 계산기.

역할별 모델 배정은 ADR-016 참조 (세 역할을 처음부터 분리). Reflexion 관점에서 **Actor = Strategist(정책) + Coder(실행)**, **Reflector = self-reflection**, Evaluator = 결정적 코드.

## 7. Cross-Competition Transfer

> **상태: 부분 구현.** `store/fingerprint.py`, `memory/transfer.py`,
> `bin/start_competition.py`, `raw.pipelines`, `cold_start_progression` 뷰는 구현되어 있다.
> 새 대회 등록 시 유사 대회/교훈/seed pipeline id를 `runs/cold_start/{competition_id}.json`에 저장하고,
> `bin/run_reflexion.py --cold-start`가 이를 읽어 첫 bootstrap 컨텍스트와 seed code로 사용한다.
> 자동 품질 검증/가중치 캘리브레이션은 아직 운영 데이터가 더 필요하다.

사용자 목표의 핵심: *"다른 Playground Series 하나 넣으면 예전 경험에 기반해 빠르게 시작."* 이를 메커니즘으로 보장한다.

- **Fingerprint** (`store/fingerprint.py`): Polars로 1회 계산하는 결정적 메타피처(스키마는 `spec.md`). 같은 데이터셋이면 항상 같은 값. 타깃 통계는 train fold 평균/분산만 사용해 누수 방지.
- **유사 대회 검색** (`memory/transfer.py`): Postgres에서 competition fingerprint를 읽어 Python에서 가중 거리 top-k를 계산한다. 가중치 — `task_type`/`metric_class` 불일치 = 큰 페널티, `size_class` 차이 = 중간, missing/cardinality 차이 = 작게.
- **교훈 일반화 레벨** (`generality`): `L1_local`(이 대회 전용, transfer 제외) / `L2_class`(유사 fingerprint 부류) / `L3_general`(정형 대회 보편).
- **Cold-start 검색**: 새 대회 N+1 →
  1. `similar = find_similar_competitions(fp_new, k=3)`
  2. 벡터 메타필터: `(competition_id IN similar AND generality='L2_class') OR generality='L3_general'`, `archived=false`
  3. Top-K 교훈 → Strategist 첫 컨텍스트
  4. `raw.pipelines`에서 `competition_id IN similar AND gain_vs_best > 0` 코드 1~2개를 seed 후보로 저장
  5. `bin/run_reflexion.py --cold-start` 첫 bootstrap에서 seed code를 `prev_code`로 주입
- **측정**: `cold_start_progression` 마트 — 누적 경험량 vs 새 대회 첫 시도의 best 대비 비율(`warm_start_ratio`). 우상향하면 transfer 작동. 아니면 fingerprint 가중치/generality 라벨링/검색 메타필터 점검.

### Cold-start 절차 (새 대회 시작)

1. 데이터셋 다운로드 (Kaggle API)
2. Fingerprint 계산 → `raw.competitions` insert
3. 유사 대회 검색 (위)
4. 시드 후보 구성: 유사 대회 `gain_vs_best > 0` 코드 1~2개 + 도메인 무관 안전 베이스라인 1개(LightGBM + 타깃 인지 k-fold)
5. Bootstrap attempts로 시드 실행, Strategist 컨텍스트를 L2/L3 교훈으로 워밍
6. `reflexion` 단계로 전환

### Transfer 리스크

- Playground Series는 task 다양성이 좁아 transfer 신호가 약할 수 있음 → 측정 마트가 검증 도구이자 튜닝 신호.
- fingerprint가 비슷해도 metric 부류가 다르면 노하우가 어긋남 → `metric_class` 페널티로 보호.
- 시드 파이프라인의 hard-coded 경로/칼럼명은 cold-start에서 깨질 수 있음 → Coder 컨트랙트(주입식 IO)를 강제해 회피.

## 8. 외부 아이디어 채널

> **상태: 설계만 (미구현, ADR-019).** 스키마/스케줄러+추출/Strategist 통합은 Linear BON-86~89.

목적: Strategist+Coder prior 의 천장을 외부 ML 지식으로 들어 올린다 (ADR-019). 누적 시도가 같은 모델 prior 안에 갇히지 않도록 외부 자극을 별도 채널로 주입하되, earned knowledge 풀(`raw.reflections`)을 오염시키지 않는다.

- **격리된 게이트웨이**: `raw.external_ideas` 가 `raw.reflections` 와 완전 분리(목표 스키마는 `spec.md` §1.11). retrieval / `reflection_impact` 마트 / 검색 score 가중치에 안 섞임.
- **소스 → 추출 → 게이트웨이**: 주간 systemd timer(ADR-017 daemon 흡수) 가 화이트리스트 소스(우승 writeup / pinned tips / gold·silver solution) 조회 → 추출 LLM(Strategist/Reflector 와 다른 호출) → 가드 4개(실측 수치 인용 / 다수 동의 / 조건부 진술만 / 500자 상한 + 코드 블록 분리) → `raw.external_ideas` insert (Beta(1, 1) 균일 prior).
- **노출 = stage 게이팅 + 톰슨 샘플링**: `reflexion` Strategist 만, `applies_when` fingerprint 1차 필터 → 각 후보 θ ~ Beta(α, β) 샘플 → top-3. `bootstrap`/`exploitation` 은 외부 idea 차단 (cold-start lessons + seed_code 가 이미 외부 신호, exploitation 은 안정화 우선).
- **승격 = 시스템 기본 루프**: 외부 idea 채택 → Coder 실행 → Evaluator 결정적 신호 → Reflector 정상 reflection. 검증된 부분만 자연히 lessons 풀로 진입. 외부 idea 자체는 `verified` 마킹 없이 영구 게이트웨이.
- **사후 학습**: 채택+jump → α++, 채택+regression → β++, 미채택 무변화. `external_idea_bandit` 뷰가 사후 상태 노출. 자동 archive: `trials ≥ 10 AND posterior_mean < 0.1` (source 단위 archive 는 사람 수동).

### Cold-start lessons 와 구분 (ADR-010 vs ADR-019)

| | Cold-start lessons (ADR-010) | External ideas (ADR-019) |
|---|---|---|
| 출처 | 우리 시스템이 다른 대회에서 측정한 reflection | 외부인의 미검증 주장 |
| 신뢰 | 측정 결과의 자연어 추상화 | LLM 추출, 가드 4개로 1차 정제 |
| 저장 | `raw.reflections` (L2/L3 generality) | `raw.external_ideas` (분리) |
| 검색 | retrieval + retrieval_score 가중치 | 톰슨 샘플링 + fingerprint 필터 |
| 노출 위치 | `## Retrieved Lessons` | `## External Ideas` (reflexion 단계만) |
| 노출 대상 | Strategist | Strategist 만 (Coder/Reflector 차단) |

### 외부 채널 리스크

- **추출 노이즈**: 일화 잡음이 false signal 양산 — 가드 4개로 1차 방어, α/β 누적으로 사후 정제.
- **천장 vs 1변경 규율 충돌**: 외부 복합 아이디어를 한 사이클에 다 적용 못함 — 같은 idea 가 다음 사이클 다시 톰슨 샘플링되면 다른 부분/다른 `action_type` 으로 채택. 의도된 점진 분해.
- **노출 대상 격리 필요**: Coder 노출 시 검증 안 된 코드 카피 위험 → `prev_code` 위 1변경 컨트랙트(ADR-014) 파괴. Reflector 노출 시 자기 추론을 외부 주장으로 정당화 → ADR-016 cross-family critic 효과 깎임. Strategist 단일 진입점으로 격리.
- **승격 평가의 인과 한계**: `reflection_impact` 와 같은 상관 한계(ADR-015). ablation 도입 시 같이 캘리브레이션.
- **톰슨 staleness**: 시간 decay 없음 — 누적 우위 idea 가 영구 우위. 트렌드 변화 시 weekly decay 도입 가능.
