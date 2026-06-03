# 아키텍처

컴포넌트 배치와 데이터 흐름, 그리고 두 시스템 목표(intra-competition gain / inter-competition transfer)를
보장하는 메커니즘을 기술한다. 스키마·API 등 구체 명세는 `spec.md`, 결정 근거는 `decisions.md` 참조.

## 1. 노드 배치

| 컴포넌트 | 위치 | 비고 |
|---|---|---|
| Strategist (정책) | Mac Ollama 서버 | 테스트: qwen3.5:14b / 프로덕션: deepseek-v4-pro (ADR-016) |
| Reflector (성찰) | Mac Ollama 서버 | 테스트: qwen3.5:14b / 프로덕션: glm-5 (ADR-016) |
| Coder (실행) | Mac Ollama 서버 | 테스트: devstral-small-2 / 프로덕션: qwen3-coder-next (ADR-016) |
| 임베딩 | Mac Ollama 서버 | qwen3-embedding:8b (1024d, MRL) — ADR-008 |
| Evaluator (CV · 지표 · Optuna · label) | WSL2 로컬 | 결정적 코드 |
| 생성 코드 실행 | WSL2 로컬 | **현재 in-process `exec`** (격리 미구현, §5는 계획) |
| Memory (검색) | 로컬 | DuckDB (벡터 컬럼 + 브루트포스 코사인) |
| Orchestrator | 로컬 | 단순 Python 러너 + cron |
| Warehouse + 분석 뷰 | 로컬 | DuckDB (스토어·검색·분석 단일화, dbt 아닌 SQL view) |

설계 의도: **추론만 클라우드, 나머지는 로컬에서 시작.** 병목/비용이 데이터로 잡히면 그때 분산화.

## 2. 데이터 흐름

```text
              (retrieve)                                   (CV score)
 DuckDB ------------------> [Strategize] -> [Generate] -> [Evaluate: k-fold]
 (벡터검색)                  (Cloud)        (Cloud)         (Local, 결정적)
     ^                                                            |
     | (lessons)                                                  v
 [Reflect] <------------ best 후보 --------------------- [Submit? <=5/day] -> LB
 (Cloud) ----> DuckDB(competitions/attempts/reflections[+embedding]) + 분석 뷰(SQL view)
     |
     +--------------- next attempt -----------------> [Strategize]
```

## 3. Reflexion 루프 (1 attempt = 1 cycle)

1. **Retrieve**: 검색 키 = `(competition_fingerprint, last_attempt_summary 또는 seed_query)` → DuckDB 벡터 검색(브루트포스 코사인) + 메타필터로 교훈 top-k.
2. **Strategize**: EDA 카드 + 검색 교훈 + 현재 stage → 다음 가설 1개 (`action_type` enum 강제). 실제 채택한 교훈 id를 함께 출력.
3. **Generate**: 가설 → `feature_fn` + `model_fn` (컨트랙트는 `spec.md`). 시드·k-fold·IO는 Evaluator가 주입. **bootstrap 외 단계는 직전 best 파이프라인을 `prev_code`로 받아 한 군데만 수정** — 1변경 규율을 코드로 강제(§4).
4. **Evaluate**: k-fold CV + 지표 + (필요 시) Optuna. **결정적 코드. CV 델타·fold 분산으로 `label`도 여기서 계산** (LLM 아님).
5. **Submit?**: 제출 예산 남았을 때만. best 후보 → LB.
6. **Reflect**: (가설, 코드, retrieved_ids, CV 결과, best 대비 델타, feature_importance, fold variance, 에러 trace) → 교훈 본문 + `generality`. Reflector의 정성 판정(`reflector_label`)은 참고용으로만 기록.
7. **Persist**: DuckDB 단일 스토어에 기록(임베딩은 벡터 컬럼). 분석 뷰(SQL view)는 자동 반영. 생성 코드는 `runs/code/`에 로컬 저장(사람 검토용).

피드백 신호 정책: **CV = 주 신호**(무제한·결정적), **LB = 확인용 희소 신호**(하루 예산 내, CV-LB 상관/shake 감지).

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

> **상태: 계획 (미구현).** 현재 `cycle/run.py`는 생성 코드를 in-process `exec`로 실행한다(`runtime/`은 스텁). 아래는 무인 cron 운용 전 도입할 목표 설계.

Coder가 만든 `feature_fn`/`model_fn`은 cron에서 무인 실행되므로 **컨테이너/nsjail로 격리**한다 (ADR-013).
- 시간/메모리 상한, 네트워크 차단, 파일시스템 화이트리스트.
- 행/OOM/무한루프가 한 워커를 죽이지 않게 격리 경계에서 강제 종료.
- 격리 위반·타임아웃은 `error_trace`로 기록되어 Reflector가 실패에서 교훈을 뽑는다.

## 6. 컴포넌트와 역할

LLM 역할 3개:

| 역할 | 입력 → 출력 |
|---|---|
| Strategist | EDA 카드 + 검색 교훈 + stage → 가설 1개 + `action_type` + 채택 교훈 id |
| Coder | 가설 + 파이프라인 컨트랙트 → `feature_fn` + `model_fn` |
| Reflector | (가설, 코드, retrieved_ids, CV 결과, best 대비 델타, feature_importance, fold var, 에러 trace) → 교훈 본문 + `generality` + (참고) `reflector_label` |

비-LLM 컴포넌트:
- **Evaluator**: k-fold CV + 지표 + Optuna 캡슐화. 결정적 시드·budget. `label`·`gain_vs_best` 계산.
- **Memory/Retriever**: DuckDB 벡터 컬럼 + 임베딩 + 메타필터 + 재순위 (브루트포스 코사인, ADR-007).
- **Fingerprinter**: 결정적 메타피처 계산기.

역할별 모델 배정은 ADR-016 참조 (세 역할을 처음부터 분리). Reflexion 관점에서 **Actor = Strategist(정책) + Coder(실행)**, **Reflector = self-reflection**, Evaluator = 결정적 코드.

## 7. Cross-Competition Transfer

> **상태: 설계만 (Phase 3 미구현).** `memory/transfer.py`·`cold_start_progression` 뷰·`start_competition.py`의 cold-start 절차가 아직 없다. 시드용 코드 소스도 미정(생성 코드는 `runs/code/`에 사람 검토용으로만 저장 — 시드 풀 승격 다리 필요). 아래는 목표 메커니즘.

사용자 목표의 핵심: *"다른 Playground Series 하나 넣으면 예전 경험에 기반해 빠르게 시작."* 이를 메커니즘으로 보장한다.

- **Fingerprint** (`store/fingerprint.py`): Polars로 1회 계산하는 결정적 메타피처(스키마는 `spec.md`). 같은 데이터셋이면 항상 같은 값. 타깃 통계는 train fold 평균/분산만 사용해 누수 방지.
- **유사 대회 검색** (`memory/transfer.py`): fingerprint 가중 유클리드 거리 top-k. DuckDB SQL로 충분(메타피처 수십 차원). 가중치 — `task_type`/`metric_class` 불일치 = 큰 페널티, `size_class` 차이 = 중간, missing/cardinality 차이 = 작게.
- **교훈 일반화 레벨** (`generality`): `L1_local`(이 대회 전용, transfer 제외) / `L2_class`(유사 fingerprint 부류) / `L3_general`(정형 대회 보편).
- **Cold-start 검색**: 새 대회 N+1 →
  1. `similar = find_similar_competitions(fp_new, k=3)`
  2. 벡터 메타필터: `(competition_id IN similar AND generality='L2_class') OR generality='L3_general'`, `archived=false`
  3. Top-K 교훈 → Strategist 첫 컨텍스트
  4. `raw.pipelines`에서 `competition_id IN similar AND gain_vs_best > 0` 코드 1~2개를 시드 후보로 큐잉
  5. Bootstrap에서 시드 실행 → 베이스라인 CV 확보
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
