# 기술 결정 이력 (ADR)

각 항목은 결정·근거 중심. "무엇"의 상세는 `architecture.md`/`spec.md` 참조.

## ADR-001 — 학습 대신 Reflexion
- 결정: fine-tuning이 아닌 경험 기반 RAG 루프.
- 근거: 인프라/비용 부담 적고, 교훈이 사람이 읽고 디버깅 가능한 자연어로 남음.

## ADR-002 — 도메인은 Kaggle 정형 대회
- 결정: CSV 제출형 정형 대회.
- 근거: 객관 피드백, API 자동화, CPU GBDT, 자주 열림. 코드 대회는 노트북 재실행이라 완전 자동화 불가 → 제외.

## ADR-003 — CV 주 신호, LB 확인용
- 결정: 성찰은 CV 델타 기준, LB는 희소 확인.
- 근거: CV는 무제한·결정적. "trust your CV" 정석.
- **[2026-08 #104 amend]**: "LB는 확인용"의 전제가 CV가 신뢰 가능하다는 데 있었는데, CV 자체가 오염(ADR-024/025 참조)될 수 있음이 드러나면서 LB가 유일한 외부 검증 신호가
되는 경우가 생겼다. `cv_lb_calibration` 뷰 + 발산 트립와이어를 추가해, CV는 개선인데 LB가 악화된 제출이 나오면 해당 pipeline을 격리하고 대회 auto-submit을
중단한다(ADR-026). CV가 주 신호라는 결정 자체는 유지 — LB는 이제 "확인"을 넘어 CV 신뢰가 깨졌을 때의 차단 신호로도 쓰인다.

## ADR-004 — 추론은 Ollama Cloud, 임베딩은 Mac 로컬 서버
- 결정: Strategist/Reflector/Coder는 Ollama Cloud Pro(`OLLAMA_CLOUD_BASE_URL=https://ollama.com`, Bearer 인증).
임베딩(`qwen3-embedding:8b`)만 Mac Ollama 서버 로컬 유지. 오케스트레이션·저장소·CV·분석 뷰는 WSL2 로컬.
- 코드 라우팅: 에이전트 3개는 `OLLAMA_CLOUD_BASE_URL` + `OLLAMA_API_KEY`, retriever는 `OLLAMA_BASE_URL`만 사용(키 없음).
- 근거: Cloud Pro 모델(deepseek-v4-pro / glm-5 / qwen3-coder-next) 품질이 로컬 14b 대비 명확히 높다고 판단해 전환. 임베딩은 클라우드 키 인가 범위 밖이고 로컬
8b로 충분하므로 분리 유지. ADR-016 패밀리 다양성은 Cloud 전환으로 자동 확보.

## ADR-005 — Evaluator는 결정적 코드
- 결정: 채점은 코드로만. LLM-as-judge 금지.
- 근거: 피드백 객관성. 성찰 오염 방지.

## ADR-006 — `reflexion` 단계에 한해 시도당 변경 1개
- 결정: `reflexion` stage의 attempt만 단일 변경 강제. `bootstrap`/`exploitation`은 예외.
- 근거: 인과 귀속은 유지하되, cold-start 비효율을 피한다.
- **[뒤집힘, ADR-037/#232]** `evaluator/contract.py`의 하드 리젝트 강제는 폐지 — 프롬프트 가이드
("주 초점 훅")로만 남는다. 훅 합성(base 실행 후 patch 적용)이 도입되면서 여러 훅을 건드려도
이전 개입이 사라지지 않아, 제한의 실익(causal attribution) 대비 비용(강제 자체가 "1변경=축적"을
보장하지 못했다는 #232의 발견)이 역전됐다고 판단.

## ADR-007 — DuckDB 단일 스토어 (검색·분석 통합)
- 결정: 별도 벡터DB(Chroma) 없이 DuckDB 하나에 모두 둔다. 임베딩은 `reflections.embedding`을 `FLOAT[768]` 컬럼으로 저장하고, 검색은
`array_cosine_similarity` 브루트포스 + 메타필터로 수행. 분석 마트도 같은 DB.
- 대안: (a) Chroma 별도 + DuckDB dual-write (v2 초안), (b) pgvector로 전부 Postgres.
- 근거: 이 프로젝트의 벡터 규모는 누적 1만~수만 건 수준이라 768차원 브루트포스 코사인이 수십 ms로 충분 — ANN 인덱스(Chroma/pgvector HNSW)는 조기 최적화. dual-write를 없애
검색·분석·기록의 정합성을 한 트랜잭션으로 보장하고, zero-server(ADR-011)와 DuckDB의 OLAP 강점(마트 window 쿼리)을 동시에 유지. pgvector는 분석까지 Postgres로 옮겨야
이득이 생겨 ADR-011과 충돌하므로 제외.
- 승격 트리거: 벡터 수십만 건 초과로 브루트포스 지연이 체감되면 DuckDB `vss` 확장(HNSW)으로 인덱스만 추가. 그래도 부족하면 그때 전용 벡터DB 재검토.
- **[2026-06-06 amend]** BON-98: DaemonAPI + main loop write-write 충돌(DuckDB 파일락, commit d9ad514) 발동으로 재고 트리거 발동. 대안 (b)
pgvector 전환 확정. `store/schema.sql` Postgres 재작성, psycopg2 + pgvector 도입. ADR-011 수정: DuckDB 단일 서버 → ops-vm Postgres +
pgvector. 데이터 마이그레이션 없음(새 스키마 시작).

## ADR-008 — 임베딩은 로컬 (qwen3-embedding:8b)
- 결정: `qwen3-embedding:8b`(1024차원, MRL 32~1024 절단 가능)를 Mac Ollama 서버에서 로컬 실행. 스키마는 `reflections.embedding float[1024]`.
- 대안: nomic-embed-text(768d, v2 초안) / embeddinggemma:300m(경량 768d) / qwen3-embedding:0.6b(경량 이전 버전).
- 근거: 검색 품질이 transfer 메커니즘을 직접 좌우하는데, qwen3-embedding:8b가 2026 MTEB v2 오픈웨이트 최상위권이고 Ollama Cloud 키 인가 범위 밖이라 로컬 유지.
0.6b에서 8b로 업그레이드한 것은 교훈 검색 품질 향상이 목적 — 기존 임베딩과 차원(1024d)은 동일하므로 스키마 변경 없음. 저장 압박 시 MRL로 차원 절단.

## ADR-009 — LLM 출력은 JSON Schema 강제
- 결정: 가설/교훈을 스키마로 강제.
- 근거: 파싱 안정성 + 재시도 제거.

## ADR-010 — Cross-Competition Transfer를 1등 시민으로
- 결정: `competitions.fingerprint`, `reflections.generality`, `raw.pipelines`를 핵심 스키마에 포함. 시스템 목표에 "대회 간 cold-start 개선"을 명시.
- 대안: 자연어 유사도 단일 검색 키.
- 근거: 사용자의 진짜 목표("새 대회 빨리 시작")가 메커니즘으로 보장되고 마트로 측정 가능해야 한다.

## ADR-011 — 운영 스택은 단순 Python 러너로 시작
- 결정: Airflow/MLflow 미사용. cron + Python + DuckDB.
- 대안: Airflow + MLflow 풀스택.
- 근거: 1~2 워커 규모에 과체급. 복잡도 증가 시 Prefect로 승격.
- **[2026-06 amend]** BON-110: 3 attempt 병렬 실행 필요로 Airflow 채택. daemon이 `raw.cycle_queue` 폴링 후 Airflow DAG
`reflexion_rondo_cycle`을 트리거. DAG는 retrieve → attempt_0/1/2 (병렬) → promote 4태스크 구조. `AIRFLOW_URL` 없는 direct daemon mode는
운영 대체 경로가 아니라 로컬 smoke/test용 단일 attempt fallback이다. MLflow는 여전히 미사용.
- **현재 상태:** 운영 store는 DuckDB가 아니라 ops-vm Postgres + pgvector다(ADR-007 amend). daemon은 큐/API/페이싱을 맡고, 운영 attempt 병렬화는 Airflow가 맡는다.

## ADR-012 — label은 결정적 임계값으로 계산, Reflector 판정은 참고용
- 결정: `label`(jump/neutral/regression)·`gain_vs_best`는 **Evaluator가 CV 델타와 fold 분산으로 결정적 계산**한다. Reflector(LLM)의 정성 판정은
`reflector_label`로 별도 기록하되, 마트·검색의 진실값으로는 쓰지 않는다.
- 대안: Reflector가 label을 직접 부여 (v2 초안).
- 근거: jump/regression 판정은 점수 움직임에 대한 채점이므로 ADR-005(LLM-as-judge 금지)에 귀속된다. Playground는 fold 노이즈 수준의 델타 싸움이라 "노이즈 vs 진짜
점프" 경계를 임계값으로 명시해야 한다. 정성 판정은 디버깅 참고로 가치가 있어 폐기하지 않고 분리 보관.
- **[2026-07 amend]** BON-194: `LABEL_Z` 1.0 → 2.0. 1σ는 통계적으로 유의하지 않아 노이즈가 상시 "jump"로 라벨링되고 그 노이즈가 `reflection_impact`
검색 부스팅에 그대로 반영됐다. 2.0σ를 방어적 기본값으로 확정, 대회 데이터 축적 후 fold_std 실측 분포로 재캘리브레이션 예정.
- **[2026-07 amend]** BON-267: jump 판정 기준을 harness의 절대-마진(`delta > LABEL_Z * fold_std`)에서 promotion 게이트와 동일한 paired
per-fold 유의성 검정(`is_significant_gain`, BON-247)으로 통일. 절대-마진 기준은 수렴한 대회에서 사실상 도달 불가해(7447건 중 jump 0건 실측) label과 promotion
판정이 어긋났고, 그 결과 bandit·stagnation·reflection이 전부 "성공 신호 0"으로 고착돼 있었다. `cycle/run.py`에서 eval 직후 `is_significant_gain`으로
label을 재확정(harness가 절대-마진으로 jump 판정했더라도 paired 미달이면 neutral로 강등)하며, `LABEL_Z` 자체는 promotion 용도로 유지.

## ADR-013 — 생성 코드는 컨테이너/nsjail로 격리 실행
- 결정: Coder가 생성한 `class Patch`는 격리 런타임에서 실행. 시간/메모리 상한, 네트워크 차단, FS 화이트리스트.
- 대안: timeout만 적용 / 신뢰 후 직접 실행.
- 근거: cron 무인 루프에서 LLM 생성 코드를 실행하므로 OOM·행·우발적 네트워크 접근이 워커를 죽이거나 환경을 오염시킬 수 있다. 격리 경계가 안정성과 재현성을 보장한다.
- **현재 구현(BON-191):** `runtime/isolate.py`가 `runtime/runner.py`를 subprocess로 실행. preexec_fn에서
`os.unshare(CLONE_NEWNET)`으로 network namespace를 분리해 subprocess egress 차단(DockerOperator `cap_add=["SYS_ADMIN"]` 필요).
CAP_SYS_ADMIN 없는 환경(로컬 mac 등)에선 조용히 스킵. rlimit(AS/CPU) + timeout 병행. env allowlist가 시크릿 env 제거.
- amend(BON-275, 2026-07-19): 타임아웃 600s→1200s 상향. s5e5(75만 행) 5-seed bagging이 기존 600s를 넘겨 매번 타임아웃으로 실패하던 문제 — eval과 동일한 값으로 통일.

## ADR-014 — Coder 컨트랙트는 class Patch + hook 분리
- 결정: 산출물은 `class Patch` 하나. action_type에 허용된 훅(hook)만 구현하고, 나머지는 현재 best pipeline이 fallback으로 제공한다. 훅은 `preprocess` /
`feature_transform` / `param_candidates` / `build_model` / `postprocess_predictions` / `ensemble_spec` 6종. IO/k-fold
하니스/파라미터 선정은 Evaluator가 소유.
- 대안: 전체 스크립트 자유 생성 / `feature_fn`+`model_fn` 두 함수 분리 (이전 방식).
- 근거: hook 분리는 action_type 귀속을 코드 레벨로 강제하고(feature_transform만 바꾸는 게 feature_engineering), 1변경 규율을 컨트랙트로 보장한다. best
pipeline을 base class로 두고 patch가 단일 훅만 override하면 cold-start seed 코드도 안전하게 재사용 가능. `validate_patch()` (AST 레벨)가 실행 전 위반을
차단한다.
- **[2026-06 BON-113]**: `feature_fn`+`model_fn` → `class Patch` with hooks로 전환. `materialize_best_pipeline()`이 이전 best와
신규 patch를 AST 레벨에서 병합해 누적 pipeline을 유지한다.
- **[2026-07-05 BON-268 amend]**: `validate_patch()`(AST 정적 검사)에 pandas-only API
금지(`.groupby`/`.map_dict`/`.take`/`.apply`/`.iterrows`/`.applymap`/`.get_dummies` — polars 1.41.2 실물에 `hasattr`로 대조해 확정,
`value_counts`는 polars Series에 실존해 의도적으로 제외) + candidate patch 자신의 undefined-name 검사(실행 격리 모델과 동일 범위)를 추가.
`agents/coder.py` 프롬프트에도 동일 금지 목록 반영.
- **[2026-07-22 #42 amend]**: 정적 검증은 코드 생성 *이후*에만 컨트랙트 위반("action_type=X may not implement hooks: [...]")을 잡아 재시도해도 같은
실수가 반복됐다(s6e7 실측: model_swap이 feature_transform을 구현하려는 시도 다수). `agents/coder.py.generate_code()`가
`evaluator/contract.py._ALLOWED_HOOKS`(source of truth)를 직접 import해, 매 호출 user 메시지에 이번 action_type이 허용하는 hook만 동적으로
강조하도록 변경 — 생성 이전 단계 가드 추가. 같은 커밋에서 multiclass 라벨 왕복(round-trip) 규칙도 컨트랙트에 명시: 타깃을 정수로 인코딩했으면 `postprocess_predictions`에서
원래 문자열 라벨로 되돌려야 한다(`ValueError: Mix of label input types` 방지, 실측 45건).
- **[2026-08 #74 amend]**: `ensemble` action_type의 자유형 wrapper 클래스(직접 fit/predict 구현)는 실행이 exec된 클래스 몸체 내부라 정적 검증도 런타임
안전망도 못 미치는 크래시(super() 오용, 생성자 stale kwarg, 자기 fit() 안에서 하위 모델 재구성)를 계속 냈다. 6번째 훅
`ensemble_spec(self, ctx) -> dict | None`을 추가해 Patch는 "무엇을 조합할지"(멤버 모델·파라미터·결합 방식)만 선언하고,
harness(`evaluator/harness.py`)가 모델 생성·적합·결합을 전담한다. 자유형 `build_model` 기반 ensemble 훅은 병행 허용 — 크래시율 재비교 후 폐기 여부 결정. 상세는
ADR-023.

## ADR-015 — 인과 귀속은 상관 기반으로 시작, ablation 보류
- 결정: 자가 개선 효과는 `reflection_impact` 상관 + 1변경 규율 + 실제 채택 교훈 id로 추정. retrieval ON/OFF ablation은 도입하지 않는다.
- 대안: 인터리브 ablation / 별도 memory-OFF 대조 대회.
- 근거: 초기 단순성 우선. 단 이는 인과 증명이 아니라는 한계를 명시하고(`architecture.md` §4), 신호가 모호하면 ablation을 후속 과제로 승격한다.
- **[2026-07 amend]** BON-195: 상관 기반 귀속은 "그냥 인과가 약하다" 수준을 넘어 **자기강화(rich-get-richer) 편향**을 갖는다 — avg_gain 높은 교훈이
`_apply_impact_score`에서 부스팅되어 더 자주 검색되고, 그 결과 avg_gain이 계속 유지/상승하는 루프가 생긴다. 완화책: (1) z-score를 배치 로컬이 아닌
전역(`reflection_impact` 전체) 통계로 계산해 배치 구성에 따른 흔들림 제거(`memory/retriever._global_gain_stats`), (2) `_IMPACT_W` 0.25 → 0.15로
부스팅 강도 감쇠. attempt gain을 인용된 교훈에 균등 배분하는 근본 문제(Coder 변경분과 교훈 기여분 미분리)는 미해결 — ablation 도입 시 함께 재검토.
- **[2026-07-22 #43 amend]**: 전역 z-score 통계(위 amend)가 metric 스케일을 구분하지 않고 `avg_gain`을 pool한다는 별도 문제 발견 — rmse degenerate
예측(모델이 완전히 빗나감)이 만드는 원시 `gain_vs_best`(s6e1 실측 -105448, baseline rmse~8.75 대비)가 전역 std를 부풀려 auc/accuracy 스케일 교훈들의
z-score를 0으로 수렴시켰다. 근본 원인은 `evaluator/harness.py`에서 처리: 기존 "baseline보다 100배 좋으면 스케일 누수로
raise"(`_REGRESSION_IMPLAUSIBLE_BASELINE_RATIO`) 가드에 대칭으로 "100배 나쁘면 gain_vs_best를 하한 클립"을 추가(raise 아님 — label 판정은 클립 전
delta로 유지, DB에 저장되는 값만 스케일 폭주 차단). `_global_gain_stats`의 metric_class별 분리는 이 가드로 전역 std가 안정되는지 배포 후 재측정한 뒤 필요시 별도
판단(reflection_impact가 reflection_id 단위 집계라 분리가 비자명함 — 단순함 우선).
- **[2026-08 #97 amend]**: `_REGRESSION_IMPLAUSIBLE_BASELINE_RATIO`를 100 → 10으로 하향. s5e5가 이 가드보다 먼저
`_check_preprocess_target_leak`(ADR-024)에 잡혔어야 할 preprocess 누수였는데도 100배 문턱을 통과해(실측 gain 44배) 승격까지 갔던 사례가 계기 — 이 비율은
preprocess 누수 검사가 못 잡는 결과 기반 2차 방어선이라 문턱을 낮춰도 정상 개선을 오탐할 여지가 적다고 판단.
- **[2026-07-22 #58 amend]**: 위 가드(클립)로도 `reflection_impact` 전역 z-score 오염(mean=-4.22, std=139.19 실측)이 해소되지 않음을 확인 — 클립은
극단값만 완화할 뿐 metric 스케일 자체(rmse 원시 단위 vs auc 0~1)를 정규화하지 않아 근본 해결이 아니었다. `gain_vs_best_relative` 컬럼(regression_error는
`gain_vs_best/baseline_cv` 상대값, 나머지는 패스스루) 신설로 전역 통계를 metric 스케일 상대화로 교체하고, `reflection_impact` 뷰가 이 컬럼만 집계하도록 재정의(값이 없는
legacy row는 제외, raw `gain_vs_best`로 폴백하지 않음).

## ADR-016 — LLM 역할별 모델 배정 (Actor 분리 + Reflector 패밀리 다양성)
- 역할 매핑: Reflexion의 **Actor = Strategist(정책) + Coder(실행)**, **Self-Reflection = Reflector**, Evaluator는 결정적 코드(ADR-005).
- 결정: **처음부터 3모델 분리.**
  - **Strategist**(정책, 추론 모델) — `glm-5.2` (2026-06-24 `deepseek-v4-pro`에서 변경. 대안 `deepseek-v4-pro`, `kimi-k2.6`).
  - **Reflector**(성찰, 추론 모델) — `kimi-k2.6` (대안 `glm-5`). **Strategist와 다른 패밀리**로 고정 — glm(Strategist) ≠ kimi(Reflector) 유지.
  - **Coder**(실행, 코드 모델) — `gpt-oss:120b` (2026-07-02 `qwen3-coder-next`→`qwen3.5:397b`, 2026-07
  `qwen3.5:397b`→`gpt-oss:120b` 재변경. 대안 `glm-4.7`).
- 근거: Coder 분리는 코드 특화 모델이 컨트랙트 준수에 유리(태스크 성격). Reflector를 다른 패밀리로 두는 건 **상관된 맹점** 완화 — 같은 모델이 가설을 내고 스스로 성찰하면 자기 추론을
합리화한다. ADR-005가 채점에서 LLM을 뺐어도 Reflector의 정성 진단·generality 라벨링엔 자기편향이 남으므로 교차 패밀리가 교훈 품질을 높인다.
- **[2026-07 amend]** BON-236: `qwen3-coder-next` deprecate 예정으로 `qwen3.5:397b`로 교체. 태그 확정 전 ops-vm에서 cloud `/api/tags`
실측 조회로 정확한 문자열 확인함(웹 검색은 `qwen3.5:397b-cloud`로 나왔으나 실제 API 응답은 bare `qwen3.5:397b` — BON-188 `glm-5.2:cloud` 접미사 오타 전례
재발 방지).
- **[2026-07 amend]** BON-240: `qwen3.5:397b`가 동일 프롬프트에서 `qwen3-coder-next` 대비 출력 토큰 9배(reasoning 과다)로 사이클당 지연·비용이 커짐.
같은 시기 코더 전문 라인(`qwen3-coder-next`/`480b`, `devstral` 계열)이 Ollama Cloud에서 전부 내려가 `gpt-oss:120b`로 교체.
`MODEL_CODER_REASONING_EFFORT`(기본 `medium`)로 reasoning 강도 조절.
- 비용: 세 역할 모두 사이클당 1회라 분리해도 호출 수는 안 늘고 설정만 는다. 처음부터 교차 패밀리 critic을 확보하는 편이 교훈 품질에 유리하다고 보고 단계적 분리(2→3)는 두지 않는다. 단순
베이스라인이 필요하면 Reflector를 Strategist 모델로 잠시 묶을 수 있으나 기본은 3모델.
- 주의: 모델 ID는 변동성이 크다. 확정 전 `ollama.com/search?c=cloud`에서 현재 태그 재확인.

## ADR-017 — 무인 24/7 운용은 worker-vm + 단일 daemon (Phase 5)
- 결정: WSL2 로컬 → nexus-prime **worker-vm**(2 OCPU ARM64 / 12GB, always-on)으로 옮겨 무인 상주. 오케스트레이션은 cron이 아닌
**단일 장수 프로세스**(`bin/run_daemon.py` + systemd `Restart=always`). 추론만 Ollama Cloud, **임베딩은 Mac Ollama 유지**(ADR-004/008
재확인). 추적: 마일스톤 Phase 5, BON-67~70.
- 대안: (a) WSL2 유지 + 데몬화 — 데스크톱이라 진짜 24/7 아님(슬립/재부팅), dev 머신과 충돌. (b) mac-server — M1 10c/32GB로 강하지만 intermittent(가정
NAT)라 무인 호스트 부적합. (c) cron + 파일락 유지(ADR-011).
- 근거:
  - **호스트**: 이 루프의 wall-clock은 Ollama Cloud 추론 대기가 지배하고 처리량은 Cloud rate-limit(5h 세션 + 주간 cap)이 이미 throttle한다. 따라서
  always-on 노드는 약해도(2 ARM/12GB) 충분 — "강한 CPU"보다 "진짜 24/7"이 우선. nexus-prime에 ML 전용 여유 노드는 없고 worker-vm이 유일한 여유 always-on
  노드(airflow edge-worker만 상주).
  - **daemon > cron** (ADR-011 정련): 단일 24/7 워커에선 데몬이 cron 중첩/DuckDB 파일락(runbook §4) 문제를 제거하고, Ollama 페이싱 상태를 메모리에 들고
  self-throttle 한다. ADR-011의 "Prefect 승격은 워커 ≥3"은 유지(BON-24) — 데몬화는 승격이 아니라 단일 워커의 단순화.
  - **임베딩 Mac 유지**: 임베딩은 매 사이클 retrieve+persist 2회로 빈번하다. Cloud로 보내면 추론 3역할에 써야 할 한도/과금을 갉아먹어 사이클 처리량 자체가 준다. Mac이 사실상
  always-on이므로 ADR-004/008 분리를 그대로 둔다. 단 Mac 일시 불통(슬립)에 대비해 daemon은 임베딩 호출에 retry/backoff, 실패 시 해당 사이클만 스킵(크래시 금지).
  - **격리 = subprocess `os.unshare(CLONE_NEWNET)`** (ADR-013의 "컨테이너 vs nsjail" TBD 확정, BON-191): 초기 설계의 "Docker
  `--network none`"은 컨테이너 레벨을 의미했으나, eval/task 컨테이너 자체는 Postgres/MinIO/Ollama 접근에 네트워크가 필요하다. 격리 경계는 **생성 코드 subprocess**.
  Python 3.12 `os.unshare(CLONE_NEWNET)`을 preexec_fn에서 호출해 subprocess에게 격리된 network namespace를 부여 — subprocess에서 나가는 모든 연결
  차단. 컨테이너에 `cap_add=["SYS_ADMIN"]` 필요(네트워크 namespace 생성용). eval 컨테이너는 시크릿 마운트 없음(secrets는 Airflow Variable env로만 주입,
  allowlist가 제거). mem/cpu/timeout 상한으로 OOM 리스크를 흡수. OOM·타임아웃은 워커 사망이 아니라 `error_trace`→교훈이 된다.
  - **Postgres 영속 + 백업**: Postgres raw 스키마의 competitions/attempts/reflections/pipelines가 누적 교훈이자 transfer 자산이다. 백업 대상은
  DuckDB 파일이 아니라 ops-vm Postgres 데이터와 MinIO/로컬 code artifact다.
- 한계: 12GB는 대형 데이터셋에서 빠듯 — 격리 컨테이너 mem-limit로 OOM을 lesson화해 흡수하되, 빈발하면 mac-server 디스패치(하이브리드)를 재고한다.

## ADR-018 — 통합 웹은 aggregator 패턴 (각 워크로드 자체 API + ops-vm 통합 UI)
- 결정: kaggle.<your-domain> 공개 웹은 각 워크로드가 read/admin API를 자체 제공하고, ops-vm의 별도 aggregator 웹(신규 repo `aggregator-web`(이름
미정), 신규 Linear 프로젝트)이 두 API를 호출해 단일 UI로 렌더한다. 공유 publish layer는 두지 않는다. rondo daemon은 Postgres raw 스키마 위에 FastAPI 라우터를
제공하고, droid controller는 자체 API endpoint만 합의한다. 추적: BON-75~77 (rondo 측) + droid BON-72 재정의 + 신규 aggregator 프로젝트.
- 대안:
  - (a) **공유 Postgres publish layer + 통합 웹**: daemon이 attempt 요약을 Postgres에 push → ops-vm 웹이 직접 read. dual-write
  일관성·schema 강제 결합·pgvector 등 새 컴포넌트 부담.
  - (b) **별개 웹 ×2, 같은 Postgres**: 도메인 단절을 인스턴스 단절로 표현. publish layer 부담은 (a)와 동일.
  - (c) **각 워크로드 API + aggregator** (선택): publish layer 폐기, 각 워크로드의 자체 store가 그대로 truth.
- 근거:
  - **도메인 단절 결정과 일관** (2026-06-04): lesson/work_unit을 두 시스템에 분리하기로 한 시점에 "공유 store"의 의미가 약해졌다. transfer가 불가능한 두 도메인(kaggle
  노하우 ↛ 게임 조작)을 한 스키마에 묶을 이유 없음. API contract만 공통.
  - **자체 store truth 유지**: rondo는 Postgres raw 스키마를 truth로 유지한다. 별도 publish hook·마이그레이션 도구·dual-write 일관성 검증이 불필요한 것이 가장 큰 단순화.
  - **repo 자율성**: 각 워크로드가 스키마/저장 방식을 자유롭게 진화. contract 변경 시에만 aggregator 동기 업데이트.
  - **보안 모델 자연**: daemon API는 tailnet only, public 노출은 ops-vm aggregator 하나로 집중. worker-vm은 공인 IP 없음(infra-lookup 결과)이라 외부
  노출 자체가 불가능하므로 이 분할이 강제이자 이득.
  - **자랑 의도 달성**: 한 페이지에서 두 시스템 표시.
- 한계/위험:
  - **API contract 합의 비용 1회**: 양 repo가 공통 응답 모양 합의. JSON schema 1장 + 명시적 버전.
  - **daemon에 HTTP 추가**: rondo daemon은 LLM 호출이 동기 블로킹이면 API 응답 지연. asyncio 또는 워커 스레드 분리로 흡수. ADR-017의 "단일 장수 프로세스" 문구는 깨지지
  않음(같은 프로세스 안 라우터 추가).
  - **aggregator 가용성 의존**: 한 API down 시 부분 표시(degraded mode 처리 필요).
  - **인증 라우팅**: viewer 무인증 / admin path는 별도 internal 도메인 또는 Caddy의 path matcher로 tailnet only 분리. SSO 미도입(nexus-prime R6 트리거 미발동) 가정.

## ADR-019 — 외부 아이디어 채널: 분리된 게이트웨이 + 톰슨 샘플링 노출

> 현재 코드에는 `raw.external_ideas`, `external_idea_bandit`, Strategist 프롬프트 통합이 아직 없다. 아래는 채택된 설계 방향이다.

- 결정:
  - Kaggle 우승 writeup / pinned tips / 유사 fingerprint 대회 솔루션 스레드 등 외부 소스에서 추출한 ML 아이디어를 별도 테이블 `raw.external_ideas` 에 보관하고
  Strategist 프롬프트에만 별도 섹션으로 노출. **reflections 풀 / `reflection_impact` 마트 / 검색 score 가중치에 일절 섞지 않는다.**
  - **외부 아이디어 자체는 `verified` / `promoted` 마킹 없이 영구 게이트웨이로 둔다.** 검증 가치는 채택된 사이클의 정상 reflection 이 lessons 풀에 들어가는 것으로만 살린다 —
  시스템 기본 루프가 곧 승격 경로.
  - **복합 아이디어를 atomic action 으로 쪼개지 않는다.** `idea_text` 원문 보존, Strategist 가 읽은 후 자기 출력에서 `action_type` 을 결정. 외부 단에는 enum 강제
  없음(추출 LLM 의 `probable_action_type` 은 nullable 추정값에 한함).
  - **노출은 stage 게이팅 + 톰슨 샘플링 (Beta-Bernoulli)** — `bootstrap`/`exploitation` 은 외부 idea 차단, `reflexion` 단계만 노출.
  `applies_when` fingerprint 1차 필터 → 각 후보 θᵢ ~ Beta(αᵢ, βᵢ) 샘플 → **top-3 노출**. 모든 idea 균일 Beta(1, 1) prior. 채택 + jump →
  α++, 채택 + regression → β++, **미채택 무변화** (Strategist 가 단지 다른 걸 선호했을 뿐, 실패 신호 아님).
  - **Archive 정책 = 자동 idea + 수동 source 분리**: idea 단위는 `trials ≥ 10 AND posterior_mean < 0.1` 자동 archive(sustained-bad 자연
  감쇠). source 단위는 사람이 사후 평가해 화이트리스트 업데이트(자동화 안 함 — 좋은/나쁜 스레드 판단은 측정만으로 부족).
  - **추출 LLM 가드 4개 모두 적용**: (i) 실측 수치 인용 있는 글만, (ii) upvote/다수 동의 임계값 통과, (iii) 조건부 진술만(절대 추천 차단), (iv) `idea_text` 500자
  상한 + 코드 블록 분리. 추출 단계 시스템 프롬프트에 포함.
- 대안:
  - (a) **lessons 풀에 `source='external'` + L3_general 직접 삽입**: 검색 단일 채널이라 우아하지만 earned knowledge 오염. `gain_vs_best` 가 null
  이라 `reflection_impact` 계산 분기 필요. Strategist 가 "측정된 교훈"과 "남의 주장"을 구분 못함.
  - (b) **시드 코드/큐로 변환** (cold-start path 연장): LLM 이 아이디어를 `class Patch` 까지 만들어 큐잉. 검증 안 된 코드가 Coder/Evaluator 비용 부담.
  cold-start 는 유사 대회 `gain_vs_best > 0` 검증 코드라 외부 아이디어와 신뢰 수준이 다름.
  - (c) **분리된 게이트웨이 + Strategist 프롬프트 힌트** (선택).
  - 노출 정책 대안: **정적 top-K (최근 N일 + score 정렬)** — 외부 source 수가 주 N=5 수준이라 빈번한 후보 중복. 톰슨 샘플링이 적은 풀에서 exploration/exploitation 균형을 자동 학습.
- 근거:
  - **천장 상승의 외부 경로**: 현 시스템은 Strategist+Coder prior 안에서만 가설을 낸다(ADR-014). 누적 학습은 "알려진 도구를 이 데이터셋에 언제·어떻게 적용할지" 의 조건부 정책일
  뿐, 새 기법 도입 경로가 없다. 외부 주입이 prior 천장을 들어 올리는 가장 직접적인 수단.
  - **분리 = ADR-005 정합**: LLM-as-judge 금지의 본질은 "측정 안 된 LLM 판정을 진실값으로 못 쓴다". 외부 아이디어는 정확히 측정 안 된 주장 — earned reflection(측정의
  자연어 추상화)과 같은 풀에 두면 retrieval score 가중치(`sim * (1 + avg_gain)`)와 `reflection_impact` 인과 귀속(ADR-015)이 동시에 오염된다. 분리하면 두 자산
  모두 손상 없음.
  - **Cold-start lessons(ADR-010) 와 구분**: cold-start lessons 는 *우리 시스템이 다른 대회에서 측정한* L2/L3 reflection(검증 자산).
  external_ideas 는 *외부인의 미검증 주장*. 둘 다 "다른 컨텍스트의 교훈을 현재 대회에 끌고 옴"이지만 신뢰 수준이 달라 풀·검색·프롬프트 섹션 모두 분리.
  - **노출 대상은 Strategist 만**: Coder 에 외부 텍스트(특히 코드 조각)를 노출하면 검증 안 된 코드 직접 카피 위험 — `prev_code` 위 1변경 컨트랙트(ADR-014)가 깨짐.
  Reflector 에 노출하면 자기 추론을 외부 주장으로 정당화 — ADR-016 의 cross-family critic 효과가 깎임. 외부 신호는 가설 생성 단계에만 영감으로 들어가고, 구현·성찰은 측정 결과에만
  기반하도록 분리.
  - **자연 승격**: 외부 아이디어 → Strategist 채택 → Coder 실행 → Evaluator 결정적 신호 → Reflector 정상 reflection. 승격 경로가 시스템 기본 루프와 동일해서 별도 검증 로직 불필요.
  - **Atomic 분해 안 함의 함의**: Strategist 가 복합 아이디어를 단일 사이클에 다 적용 못함 — 1변경 규율(ADR-006)과 충돌하지 않음. 같은 idea 가 다음 사이클에 다시 톰슨 샘플링으로
  뽑히면 다른 부분 / 다른 `action_type` 으로 채택 가능. 의도된 점진 분해.
  - **톰슨 샘플링 fit**: 외부 source 가 주 N=5 수준으로 적어 정적 top-K 는 빈번한 후보 중복. Beta-Bernoulli 가 α/β 만으로 사후 효과를 누적해 좋은 idea 자동 식별, 나쁜
  idea 자연 감쇠. **α/β 자체가 모니터링 상태라 별도 채택률 마트 불필요** — 분석 뷰 `external_idea_bandit` 하나로 노출.
  - **source 품질이 결정적**: Kaggle Discussions hotness 는 일화 잡음이 커서 LLM 추출이 false signal 을 양산. 화이트리스트 — (i) 종료된 유사 fingerprint
  대회 우승 writeup(ADR-010 fingerprint 매칭과 결합), (ii) 대회 pinned "Tips & Tricks", (iii) gold/silver solution 스레드. 양보다 질(주당 N=5
  수준).
  - **추출 LLM 분리**: 같은 모델이 가설/성찰/외부 추출 셋을 다 하면 다양성이 0. Strategist/Reflector 와 다른 호출. 빈도가 주 1회 수준이라 비용은 무시 가능.
- 데이터 모델 (스키마 상세는 `spec.md`):
  -
  `raw.external_ideas(idea_id, fetched_at, source_url, source_kind, idea_text, probable_action_type NULLABLE, applies_when_json, confidence, alpha float default 1.0, beta float default 1.0, archived, adopted_attempt_ids)`
  - `probable_action_type` 은 추출 LLM 추정값(nullable). Strategist 가 채택 시 자기 `action_type` 을 자유 결정 — 외부 enum 강제 없음.
  - `applies_when` 은 fingerprint 메타필터(task_type, metric_class, size_class). 톰슨 샘플링의 1차 필터.
  - `alpha`/`beta` — 균일 Beta(1, 1) 시작. 채택+jump → α++, 채택+regression → β++, 미채택 무변화.
  - `adopted_attempt_ids` — 채택 attempt 역추적(디버깅·사후 분석). 운영 결정은 α/β 가 dominant.
  - `raw.attempts.adopted_external_idea_ids` 컬럼 신설 — attempt → reflection 으로 승격된 뒤에도 외부 출처 역추적 정합.
  - 분석 뷰 `external_idea_bandit(idea_id, posterior_mean=α/(α+β), trials=α+β, last_adopted_at, ...)`.
- Strategist 통합:
  - **Stage 게이팅**: `bootstrap`/`exploitation` 은 `## External Ideas` 섹션 자체를 빼고 `reflexion` 단계만 포함.
  - 기존 `## Retrieved Lessons` 와 분리된 `## External Ideas` 섹션.
  - 후보 선택: `applies_when` 매치 + `archived=false` 풀 → 각 idea θᵢ ~ Beta(αᵢ, βᵢ) 샘플 → top-3 노출.
  - 프롬프트 톤: **"영감용, 채택 안 해도 됨"** — 1변경 규율의 안정 학습이 외부 잡음에 끌려가지 않도록.
  - Strategist 출력에 `external_idea_ids` 필드 신설(`reflection_ids` 와 같은 패턴 — `agents/strategist.py:70` 의 `valid_ids` 가드 재사용).
- 한계/위험:
  - **LLM 추출 노이즈**: hot 스레드도 일화 중심. confidence 라벨 + dedup + source 화이트리스트로 1차 방어. α/β 누적으로 사후 정제(나쁜 idea 자연 감쇠, 나쁜 source 는 수동 archive).
  - **새 외부 의존**: Kaggle 스크래핑/API rate, scheduler 추가. ADR-017 daemon 운용에 단일 systemd timer 로 흡수.
  - **큐레이션 부담 지속**: source 화이트리스트는 한 번 정한다고 끝이 아님 — 톰슨 샘플링이 idea 단위는 자동 정제하지만 source 단위는 사람이 평가.
  - **승격 평가의 인과 한계**: external_ideas adoption → gain 의 인과 귀속도 `reflection_impact` 와 같은 상관 한계(ADR-015)를 갖는다. ablation 도입 시 같이 캘리브레이션.
  - **톰슨 샘플링 staleness**: 시간 decay 없음 — 오래 누적된 좋은 idea 가 우위를 영구 유지. 트렌드 변화 시 새 cold-start(Beta(1,1)) 가 못 따라잡을 수 있음. 신호 보이면 weekly decay 도입.
- 추적: Linear BON-XX 시리즈(스키마 / 스케줄러+추출 / 화이트리스트 / Strategist 통합 4단계 분리 예정).

## ADR-021 — preselect_params는 단일 inner holdout (nested CV 미적용)

- 결정: `evaluator/harness.py: preselect_params`는 80/20 단일 inner split으로 best params를 선택한다.
  per-fold nested CV(k^2 모델 피팅)는 도입하지 않는다.
- 근거: playground-series 규모(수만~수십만 행)에서 k=5 × inner k=5 = 25회 피팅은 사이클당 LLM 호출
  대기에 비해 무시할 수 없는 추가 비용이다. 낙관 편향 잔존(inner split이 outer fold와 같은 train에서 추출)은
  인지하되 실험적으로 허용한다.
- 한계: inner split과 outer fold가 완전히 겹치므로 params 선택에 약한 낙관 편향이 있다. CV 점수 자체에는
  영향 없음(preselect는 model build에만 쓰이고 CV score 계산 loop는 선택된 params로만 실행).
- 후속 후보: per-fold nested CV(비용 허용 시), random hyperparameter search(fixed budget).

## ADR-020 — 밴딧은 advise-only, 최종 action 결정은 LLM

- 결정: `cycle/action_optimizer.py`의 Beta-Bernoulli 밴딧은 **advisory**로만 동작한다.
  정상 사이클(`reflexion` stage)에서 밴딧 posterior 샘플은 Strategist 프롬프트의 텍스트 hint로만
  주입되고(`get_action_prior` → `action_prior` dict), 최종 `action_type` 결정은 LLM(Strategist)이
  자유롭게 내린다. regret 최적화 보장 없음.
  `super_cycle`에서는 `assign_super_cycle_actions`가 Thompson 샘플로 attempt 3개에 서로 다른
  action을 강제 배정한다(`forced_action` 경로). 이쪽은 탐색 강제화 목적이라 LLM 자유 선택 없음.
- 근거: Strategist가 EDA 컨텍스트·교훈·스테이지를 종합해 선택하는 것이 밴딧 단순 posterior보다
  정교하다. 밴딧은 "최근 N회 어떤 action이 효과적이었는지"를 수치로 요약해 힌트로 제공하는 용도.
  epsilon-greedy 강제 혼합(안 B)은 Strategist 자유도를 훼손하고 regret 이론이 LLM 반응에 적용되기
  어려워 도입하지 않는다.
- 후속 후보: epsilon-greedy 혼합(super_cycle 내 비율 강제), contextual bandit(EDA fingerprint 피처 입력).
- cross-ref: ADR-005(LLM-as-judge 금지), ADR-014(컨트랙트), `cycle/action_optimizer.py` docstring.

## ADR-022 — 이미지 빌드는 Airflow DAG로 이관, 태그 source of truth가 daemon/task에서 갈라짐

- 결정: daemon+task 이미지의 **빌드+registry push**는 airflow-stack의 `reflexion_rondo_deploy` DAG(ops 큐 docker.sock 재사용,
airflow-stack decisions.md L29)가 담당한다. 이 repo의 `deploy/release.sh`는 더 이상 ops-vm에 SSH해서 빌드하지 않는다 — registry에 해당 태그가 이미
존재하는지 확인만 하고, daemon의 실제 컷오버(compose.yml 태그 bump+재시작)만 수행한다.
- **태그의 source of truth가 이미지별로 갈라진다**: daemon은 계속 git(`deploy/compose.yml`, 이 repo)이 진실이고 release.sh가 push한다. task는
**Airflow Variable**(`rondo_task_image_version`, airflow-stack 관리)이 진실이 되고, `reflexion_rondo_deploy` DAG가 빌드 직후 즉시
bump한다 — git push도 GitDagBundle의 60초 지연도 없다.
- 근거:
  - **여러 repo 재사용**: reflexion-rondo뿐 아니라 다른 repo도 같은 방식(ops 큐 docker.sock)으로 이미지 배포를 하게 될 예정 — repo마다 ops-vm 상주 체크아웃+전용
  SSH 키를 만드는 대신, `dags/lib/image_deploy.py`(airflow-stack) 공용 헬퍼가 매 실행마다 임시 디렉터리로 clone→build→push 한다.
  - **신규 credential 불필요**: 이 repo가 public이라 clone에 인증이 없고, `registry.internal:5000`도 무인증(HTTP insecure, tailnet 경계로만
  보호)이다. private repo가 이 메커니즘을 쓰게 되면 그때 공유 read-only PAT이 필요해진다(이 repo엔 해당 없음).
  - **daemon은 남겨둔 이유**: daemon의 "배포"는 compose.yml 태그 bump(git write, 이 repo)+ops-vm 재시작이라 airflow-stack의 credential 경계 밖
  작업이다. Airflow DAG에 이 repo의 git write credential을 새로 심는 대신, 지금처럼 사용자 로컬(WSL) git credential로 release.sh가 처리하는 편이 새
  credential 없이 끝난다.
- 트레이드오프:
  - **DAG Versioning과의 결합 약화** (airflow-stack ADR-L27 참조): task 이미지 태그가 Variable로 빠지면서, 특정 DagRun이 정확히 어떤 이미지로 돌았는지 이제 git
  log가 아니라 Airflow의 rendered-template을 봐야 안다.
  - **daemon/task 버전 불일치 창**: 두 이미지가 같은 커밋에서 함께 빌드되지만, task Variable은 즉시 반영되고 daemon은 release.sh를 별도로 돌릴 때까지 구버전으로 남을 수
  있다. 둘 다 repo 전체를 COPY해 빌드하므로 코드 차이는 없지만 "지금 어느 버전이 떠있나"를 daemon/task 따로 확인해야 한다.
- cross-ref: issue #15(release.sh 사전검증 순서 수정, 이번 daemon 전용 버전에도 유지), issue #17(release.sh 축소), airflow-stack decisions.md L29/R2.

## ADR-023 — ensemble은 선언형 프리미티브(`ensemble_spec`), 자유형 wrapper와 병행

- 결정: `ensemble`/`bootstrap` action_type에 6번째 훅 `ensemble_spec(self, ctx) -> dict | None`을 추가한다. Patch는
`{"members": [{"model": <registry key>, "params": {...}}, ...], "method": "weighted_average"|"majority_vote", "weights": [...]}`만
반환하고, 모델 생성·적합·결합은 `evaluator/harness.py`가 고정 레지스트리(`lgbm`/`xgboost`/`catboost`/`hgb`/`random_forest`/`ridge`)로 전담한다. 기존
자유형 `build_model` 기반 ensemble(직접 wrapper 클래스 작성)은 병행 허용한다.
- 대안: (a) 몽키패치로 자주 나오는 wrapper 실수 패턴을 사후 교정 — 범위·리스크가 커서 보류. (b) 자유형을 전면 금지하고 `ensemble_spec` 강제 — 기존에 잘 동작하는 wrapper까지 막을 근거가 없어 기각.
- 근거: `ensemble` 크래시는 exec된 클래스 몸체 내부(super() 오용, 생성자 stale kwarg 하드코딩, 자기 fit() 안에서 하위 모델 재구성)에서 나서 정적 검증도
`_build_model_safe` 같은 런타임 안전망도 원천적으로 못 미쳤다 — harness가 그 코드를 볼 수 없기 때문. 선언형으로 바꾸면 멤버 구성·적합·결합이 전부 신뢰 코드 안에서 일어나 이 클래스의
크래시가 구조적으로 사라진다.
- 한계: 레지스트리에 없는 모델은 `ensemble_spec`으로 표현 불가 — 그런 경우는 자유형 경로가 여전히 유일한 선택지. 두 경로의 에러율을 배포 후 재비교해 자유형 폐기 여부를 판단한다.

## ADR-024 — audit holdout을 추론조건(dummy target)으로 재현하고 승격 차단 게이트로 승격

- 결정: `runtime/runner.py:_eval_holdout`이 holdout10의 타깃을 실제 추론(`bin/submit.py`)과 동일하게 dummy 상수로 치환한 뒤
preprocess/feature_transform을 태우고, 채점만 원본 타깃으로 한다. `ConfirmResult.holdout_regressed`를 신설해 `confirmed`에 AND 결합 — 후보
holdout이 현재 best(콜드스타트면 BasePipeline) 대비 악화되면 승격을 거부한다. baseline holdout을 측정 못 하면(에러) "정보 없음"으로 보고 막지 않는다(악화 확정과는 다름).
- 대안: holdout을 기록만 하고 게이트에 안 씀(기존 동작) — cross-seed confirm이 이미 승격 신뢰도를 담당한다고 봤으나, cross-seed는 seed만 바꾼 CV라 seed 불변
누수(ADR-025)에 장님이라는 게 드러났다.
- 근거: 기존 holdout은 타깃이 살아 있는 채로 파이프라인을 통과해 preprocess의 valid-target 의존 누수를 실제 추론 조건과 다르게(=누수를 그대로 재현하며) 측정했다. dummy
target 치환으로 holdout이 실제 추론 조건의 복제가 되면서, 이 게이트 하나로 누수뿐 아니라 train/test skew 전반을 승격 전에 잡는다.
- 한계: holdout10 자체가 없는 대회(데이터가 90/10 split을 감당 못함)는 이 게이트가 작동하지 않는다.

## ADR-025 — 누수 파이프라인은 삭제 아닌 격리, baseline은 확정 파이프라인만(phantom-max 폐지)

- 결정: `raw.pipelines.invalid_reason`(text, nullable) 컬럼을 추가해 확정 후 누수로 밝혀진 행을 격리 표시한다(삭제하지 않음 — 이력 보존). 모든 baseline 조회
경로(`cycle/run.py:_prev_best`/`_prev_best_fold_scores`, `bin/blend.py`, `bin/submit.py`, `cycle/materialize.py`)는
`invalid_reason IS NULL` 필터를 공유한다. 확정 파이프라인이 하나도 없던 대회를 위해 "전체 attempt의 max(cv)로 폴백"하던 phantom-max 분기를 제거하고, 대신 (a)
bootstrap 종료 시 최고 attempt를 `confirm_and_measure`로 검증해 자동 baseline을 세우는 경로(`cycle/run.py:establish_bootstrap_baseline`)와
(b) 기존 정체 대회를 위한 소급 스크립트(`bin/establish_baseline.py`, top-k 순회 + 첫 통과 승격)를 추가했다.
- 대안: 누수 확정 시 해당 행을 delete — attempt 이력·디버깅 근거가 사라져 기각. phantom-max를 그대로 두고 임계값만 조정 — max 순서통계량은 N이 늘수록 문턱이 같이 올라가 정직한
소폭 개선이 영원히 못 넘는 자기강화 데드락이라 근본 대응이 아니라고 판단.
- 근거: phantom-max는 확정 파이프라인 없는 대회의 콜드스타트를 풀기 위한 임시 봉합이었는데, 수백 draw의 상위 꼬리를 문턱으로 쓰다 보니 그 자체가 새로운 데드락을 만들었다(확정 파이프라인 0건인
대회에서 jump 라벨도 0건으로 실측 상관됨). baseline을 "확정된 것만"으로 좁히고 그 확정 절차 자체를 자동화(bootstrap 종료 시)·소급(기존 대회)으로 나눠 풀면 데드락과 phantom을 동시에
해소한다.
- 한계: `establish_baseline.py`의 top-k 폴백은 순위가 높은 candidate가 phantom(비정상적으로 좋은 CV)이면 cross-seed/holdout/scale-leak 가드에서 순차
탈락하며 다음 순위로 내려간다 — 후보 풀 자체가 얕으면(top-k 전부 phantom) 여전히 baseline을 못 세울 수 있다.

## ADR-026 — cv-LB 발산 트립와이어, 해제는 수동만

- 결정: 뷰 `cv_lb_calibration`(대회별 제출 시계열의 delta_cv/delta_lb/부호 일치)을 신설하고, `bin/api.py:refresh_submission_row`가 제출 결과를 받을
때마다 `_detect_cv_lb_divergence`로 판정한다. CV는 개선인데 LB가 악화된 제출이 나오면 원천 pipeline에 `invalid_reason='cv_lb_divergence'`를 표기하고 해당
대회의 `raw.competitions.auto_submit_paused_reason`을 세워 auto-submit을 중단한다. **자동 해제 없음** — 사람이 원인을 확인하고
`auto_submit_paused_reason`을 직접 NULL로 되돌려야 재개된다.
- 대안: 발산 감지 후 자동으로 이전 baseline으로 롤백 — 발산 원인이 다양해서(진짜 누수/우연한 shake-up/데이터 drift) 자동 판단이 오히려 위험하다고 판단해 기각.
- 근거: LB 회수가 자동화(ADR-003 amend)되면서 발산을 감지할 수 있게 됐지만, 감지만으로는 재발을 못 막는다 — auto-submit을 계속 돌리면 같은 문제로 반복 소모된다. 수동 해제는 의도적인
마찰이다: 자동 시스템이 "이 pipeline은 신뢰 못 함"이라고 판단했으면, 그 판단을 뒤집는 건 자동화가 아니라 사람의 확인이어야 한다.
- 한계: 해제를 잊으면 해당 대회는 무기한 자동 제출이 멈춘다 — 운영자가 주기적으로 `auto_submit_paused_reason IS NOT NULL`인 대회를 점검해야 한다(`docs/runbook.md`).
- **추가 결정(2026-08, #175)**: 원래 판정(`delta_cv > 0 and delta_lb < 0`, 크기 무관 + 단발 관측)이 실제로는 노이즈에만 반응했다 — 배포 이후 실제로 정지된 5개
대회(s4e4/s6e8/s6e4/s6e3/s6e5)의 `|delta_lb|`가 전부 `|prev_lb|`의 0.05% 미만이었고, 배포 이전 관측된 큰 발산(4e12 8.3%, 5e10 434%)은 전부 그보다 한
자릿수 이상 큰 폭이었다. **해제 자동화가 없는 상태에서 노이즈에 반응하면 그건 확률적 래칫**이다 — 제출할 때마다 대회가 정지될 확률이 생기고, 정지된 대회는 절대 돌아오지 않는다.
`_detect_cv_lb_divergence`를 (a) `|prev_lb|`의 0.1% 데드밴드(metric 스케일 auc/accuracy처럼 bounded든 rmse/rmsle처럼 unbounded든 실측 상
노이즈/실발산 군집을 정확히 가른다) + (b) 최근 3개 delta 중 2개 이상일 때만 정지, 로 변경. `_apply_cv_lb_divergence_tripwire`의 pipeline 격리 UPDATE가 0행
매치하면(#178 — auto-submit이 confirm 안 거친 attempt를 제출한 경우 raw.pipelines에 행 자체가 없음) 조용히 넘기지 않고 경고 로그를 남긴다.

## ADR-027 — 격리 subprocess 메모리 상한은 RSS가 아닌 VSZ(RLIMIT_AS) 기준

- 결정: `runtime/isolate.py`의 attempt 격리 subprocess 메모리 상한은 `RLIMIT_AS`(가상 주소공간, VSZ)로 건다. 기본값 6GiB,
`EVAL_MEM_LIMIT_BYTES`로 대회/큐별 override 가능(기존 메커니즘 유지).
- 대안: 더 낮은 값(1.5GiB)으로 tight하게 제한 — 실측으로 폐기됨(아래 근거).
- 근거: `RLIMIT_AS`는 물리 RSS가 아니라 가상 주소공간 상한이다. numpy/scipy/sklearn/lightgbm/catboost/xgboost 같은 라이브러리는 실제 쓰는 물리 메모리가 적어도
공유 라이브러리 mmap·BLAS 스레드풀 등으로 VSZ를 널찍하게 예약한다 — 1.5GiB는 이런 라이브러리를 import하는 것만으로 부족해서, 물리 메모리가 12GB나 남는 워커에서도 신규 대회 부트스트랩
attempt 전체가 실패하는 회귀를 냈다. 6GiB로 되돌린 뒤 Airflow 실측(super-cycle 3678건 전수)으로 재확인한 실제 동시성 기준 근소 초과분은 물리 RSS 여유로 흡수 가능하다고 판단.
- 한계: 이 특성 때문에 "물리 메모리가 충분한 환경"에서도 VSZ 예약이 큰 스택(예: WSL2의 다른 Python 배포판)에서는 동일 6GiB가 부족할 수 있다 — 운영 검증된 워커
환경(mac-server/worker-vm/ops-vm task 컨테이너) 밖에서 이 스크립트를 돌릴 땐 먼저 이 한계를 의심할 것.

## ADR-028 — eval CPU 상한은 커널 RLIMIT_CPU가 아니라 부모 폴링 워치독이 집행

- 결정: `runtime/isolate.py`의 attempt 격리 subprocess CPU 예산(기본 900초, `EVAL_CPU_BUDGET_SECS`로 override)은 기존 RSS 워치독과 같은 2초
폴링 루프가 `/proc/<pid>/stat`으로 직접 감시해 초과 시 명시적 `error_trace`("cpu budget exceeded: ...")를 남기고 선제 kill한다. 커널 `RLIMIT_CPU`는
폴링이 놓쳤을 때만 발동하는 soft<hard 백스톱(`budget+60`/`budget+120`)으로 강등한다. `cycle/run.py`의 eval 재시도 루프는 예산을 eval 회차가 아니라
**attempt 전체 기준**으로 집행한다 — 1회차가 예산을 다 쓰면 2회차는 아예 돌리지 않는다.
- 대안: (a) 기존처럼 `RLIMIT_CPU(soft=hard=900)`만으로 집행 — 이번 결정으로 폐기. (b) soft<hard로 켜되 백스톱이 아니라 주 집행 수단으로 유지(SIGXCPU를 자식이 직접
처리) — 자식이 LLM 생성 코드라 신호 핸들러를 신뢰할 수 없어 기각.
- 근거: 리눅스는 `RLIMIT_CPU`의 hard 한도를 soft보다 먼저 검사해서, soft==hard로 걸면 SIGXCPU 경고 단계 없이 곧장 SIGKILL(rc=-9)로 죽인다. 이는 커널 OOM
killer 사망과 문자열이 완전히 동일해 원인 구분이 불가능했고, 2026-08-07 처리량 진단이 이걸 전부 OOM으로 오판해 RSS 워치독(#154)을 배포했지만 효과가 없었다(배포 후 2일간 RSS 워치독
발동 1회, 반면 rc=-9 kill은 113건/22.4h=계산의 40%, 성공 attempt 대비 peak RSS는 여유가 컸다 — 메모리가 아니라 CPU가 원인이었음을 실측으로 확인). 게다가
`cycle/run.py`가 이 무의미한 rc=-9 원문을 LLM 재생성 피드백으로 그대로 넘겨 2회차 eval도 같은 자리에서 또 죽었다(rc=-9 attempt 113건 전부가 예외 없이 이 경로) —
attempt당 최대 소모가 ~1800초(16분+)까지 갔다. 부모 폴링으로 옮기면 원인이 명시된 에러를 남길 수 있고, attempt 단위 예산으로 재시도의 최악 소모를 절반으로 자르고, 재생성 피드백을 "더 싼
파이프라인을 써라" 같은 실행 가능한 지시로 바꿔 재시도가 낭비가 아니라 실제 성공 기회가 되게 한다. 예산 값 900은 그대로 유지한다 — 성공 attempt의 실측 wall time(p99=728초,
max=1112초)이 이 근처라 낮추면 성공을 에러로 바꿀 위험이 있고, 신규 `peak_cpu_sec` 컬럼(성공/실패 무관 기록, `peak_rss_bytes`와 동일 계약)으로 다음 사이클에 재조정할 근거를
모은다.
- 한계: `/proc/<pid>/stat` 폴링은 2초 주기라 그 사이 짧게 폭증하는 CPU 소모는 최대 2초 지연 뒤에야 감지된다(RSS 워치독과 동일한 한계, 실무상 무시 가능).
ADR-027(`RLIMIT_AS` 6GiB)은 이 변경과 무관하게 그대로 유지된다.
- **추가 결정(2026-08, #176)**: `CycleConfig.cpu_budget_secs`(`comp.CPU_BUDGET_SECS` → env(`EVAL_CPU_BUDGET_SECS`) →
`DEFAULT_CPU_BUDGET_SECS` 순 폴백)로 대회별 override를 추가했다. 전역 env가 아니라 대회별 상수로 둔 이유: kill은 fleet 전반(s6e5/s5e8/s5e4 등)에 있지만
daemon 큐가 순차 처리라 전역으로 올리면 모든 대회의 최악 벽시계가 같이 늘어난다. s6e8은 900s 기본값 대비 kill률 35%(직전 활동일 80%), 성공 attempt p99가 841s로 벽에 붙어
있었고 900s 위 분포는 완전히 검열돼 있었다 — kill은 산출물 0에 CPU만 소모하므로 기다려서 측정값을 받는 쪽이 항상 낫다는 판단 하에 `CPU_BUDGET_SECS=3600`(4배)으로 설정, 실제
분포를 관측한 뒤 영구값을 정한다.

- **추가 결정(2026-08-26, #182 — 영구값 확정)**: #176이 s6e8만 3600s로 열어둔 덕에 900s 위 분포를 처음으로 관측할 수 있게 됐고, 그 결과 기본값을
`DEFAULT_CPU_BUDGET_SECS = 3600`으로 올리고 s6e8의 대회별 override는 제거한다(중복). 실측(2026-08-19~26, 성공 attempt `peak_cpu_sec`):
s6e8(예산 3600s) p50=1349 / p90=1909 / p99=2269 / max=3244, kill률 11.5%. 반면 900s 대회들은 max가 전부 850~900에 붙어 있고
kill률이 s4e12 67% / s4e10 29% / s5e4 25% / s4e11 20%, fleet 일별 22~32%였다. **max가 상한에 붙어 있다는 건 분포가 검열(censored)됐다는
뜻**이고, 검열이 풀린 s6e8의 p50이 900s의 1.5배라는 건 "900s면 충분한데 일부가 폭주"가 아니라 **애초에 900s가 중앙값보다 작았다**는 뜻이다.
kill은 산출물 0인데 CPU는 전액 소모하므로 기다려서 측정값을 받는 쪽이 항상 낫다.
- #176이 "전역으로 올리면 모든 대회의 최악 벽시계가 같이 늘어난다"며 대회별 override를 택했던 전제는 fleet 동결(ADR-032)로 사라졌다 — 큐 리필 대상이
deep tier 5개뿐이라 전역 기본값 상향의 영향 범위가 그 5개와 같다. 대회별 override 메커니즘 자체는 그대로 남긴다.
- **벽시계 상한(`DEFAULT_TIMEOUT`)과의 결합(#207)**: `eval_isolated`의 벽시계 상한 1200s는 CPU 예산과 무관하게 별도로 걸려 있어서, CPU 예산만
3600s로 올리면 1200s 벽에서 잘려 아무 효과가 없다(성공 attempt의 CPU/벽시계 비는 p50 1.8~2.5, p90 약 4 — 병렬성 낮은 attempt일수록 벽시계가 먼저
닿는다). 호출자가 `timeout_sec`을 명시하지 않으면 `max(DEFAULT_TIMEOUT, cpu_budget)`을 쓰도록 바꿔 **CPU 예산이 벽시계에 선점당하지 않는다는
불변식**을 세운다. 벽시계 상한의 역할은 계산량 제한이 아니라 CPU를 안 쓰는 행(hang) 감지로 한정된다. `DEFAULT_TIMEOUT` 상수 자체는 1200s로 유지 —
CPU 예산이 그보다 작게 설정된 대회에서는 여전히 hang 감지 하한으로 동작한다.
- 한계: attempt당 최악 벽시계가 길어져 큐 처리량(대회당 일일 attempt 수)이 줄 수 있다. 다만 지금 줄어드는 건 어차피 kill로 버려지던 몫이라 순손실이
아니다 — 배포 후 fleet kill률과 대회별 일일 성공 attempt 수를 함께 관측한다.

## ADR-029 — 생성 코드의 `n_jobs=-1` 등 무제한 병렬성은 정적 거부

- 결정: `evaluator/contract.py:validate_patch`에 `n_jobs`/`thread_count`/`num_threads`/`nthread`/`n_threads` 키워드 인자가 0 이하
리터럴(전부/거의 전부 코어 요청)이면 거부하는 AST 검사를 추가한다. 값이 변수·표현식이면 판정하지 않는다(기존 pandas-only 검사와 동일하게 과소탐지를 오탐보다 우선).
- 대안: (a) Docker/cgroup 레벨에서 하드 CPU quota로 강제 — airflow-stack `reflexion_rondo_cycle.py`의 `cpus=1.5`가 이미 이 의도였으나 docker
provider가 `cpu_shares`(상대 가중치)로만 반영해(`CpuQuota=0` 실측 확인) 하드 캡이 아니다. 근본적으로는 더 맞는 위치지만 다른 repo(airflow-stack) 작업이고 fleet
처리량이 슬롯 대비 포화 상태가 아니라 급하지 않음 — 별도 후속. (b) 이 값 그대로 두고 무시 — 아래 실측 때문에 기각.
- 근거: `OMP_NUM_THREADS=2`/`OPENBLAS_NUM_THREADS=2`/`MKL_NUM_THREADS=2`(`deploy/Dockerfile`)는 BLAS/OpenMP 레벨 스레딩만 제한하고 이
파라미터들과는 무관하게 동작한다는 걸 이 세션에서 직접 재현: `OMP_NUM_THREADS=2` 환경에서 LightGBM `n_jobs=-1`은 20 threads/15.9x cores-equivalent,
CatBoost `thread_count=-1`은 21 threads/15.0x, scikit-learn `RandomForestClassifier(n_jobs=-1)`은 43 threads/15.6x를 썼다.
Airflow attempt 컨테이너의 CPU 상한이 위 대안(a)처럼 사실상 장식이라, LLM 생성 코드 하나가 이 파라미터를 쓰면 같은 호스트(특히 big 큐 슬롯 2개가 4vCPU를 공유하는
mac-server)의 sibling attempt를 실제로 굶길 수 있다. #159(eval CPU 예산 워치독)의 kill은 이 문제를 해결하지 못한다 — attempt 자신의 CPU-초 소진을 더 빨리 채워
자신은 더 빨리 죽지만, sibling이 그 사이 굶는 것 자체는 막지 못한다.
- 한계: 이름 기반 정적 lint라 `getattr`/문자열 조합/변수 경유 등으로 우회 가능하다(파일 상단 docstring — 보안 경계 아님, 정직한 실수를 값싸게 재생성으로 돌려보내는 soft guard).
실제 강제 경계는 여전히 Docker/cgroup 레벨이어야 하며, 그 후속(대안 a)이 남아 있다.

## ADR-030 — bandit/lesson 보상 신호는 attempt-time label이 아니라 confirm 결과를 반영

- 결정: `cycle/promotion.py`에 순수 함수 `effective_label(original_label, confirm)`을 추가한다. `label=="jump"`인데
`confirm.confirmed is False`(cross-seed 미재현 또는 holdout 악화)면 하류 학습 신호(`update_bandit` 호출의 `label`, `reflect()`에 넘기는
`AttemptContext.label`)에는 `"regression"`으로 다운그레이드한다. `raw.attempts.label`(attempt 생성 시점 DB 값)은 건드리지 않는다 — 그 시점의 잠정 판정은
그대로 사실이고, 다운그레이드 대상은 오직 나중에 계산되는 보상/lesson 신호뿐이다. `cycle/run.py`(직접모드, `run_attempt_core`)와
`bin/run_promote_task.py`(프로덕션 airflow 모드) 양쪽에 적용한다.
- 대안: (a) 같은 아이디어의 재생성을 코드 내용(hash) 기준으로 캐싱해 재검증을 건너뛰기 — 증상(반복되는 22분짜리 confirm)만 가리고 원인(왜 같은 아이디어가 계속 최고 후보로 뽑히는가)은 안
건드린다. 바이트 단위로 동일한 코드만 잡아 사소한 변형에는 무력하다 — 기각. (b) `update_bandit`을 confirm 완료까지 지연 — 프로덕션은 retrieve/attempt×3/promote가 별도
Airflow task/컨테이너라 "승자가 될지" 자체가 attempt 생성 시점엔 미정이라 구조적으로 불가능. 대신 승자에 한해 confirm 이후 보정 delta를 추가하는 현재 설계를 택함.
- 근거: `cycle/action_optimizer.py:update_bandit`은 `label=="jump"`에 α+=1.0(최강 보상)을 준다. `cycle/run.py`가 `defer_promotion`
여부와 무관하게 이 함수를 무조건 호출하는데, confirm(cross-seed+holdout, `bin/run_promote_task.py`에서 승자만 별도로 나중에 실행)은 이 시점에 아직 모른다 — 실제로
`bin/run_promote_task.py`는 `update_bandit`을 아예 호출하지 않았다(grep 확인, #164 전). 그 α/β를 소비하는 `assign_super_cycle_actions`는
`get_action_prior`(advisory, LLM이 최종 결정)와 달리 LLM 개입 없는 결정론적 top-N 배정이라 안전판도 없다. 실측(2026-08-09~10): s6e1의 `preprocessing`
후보가 cv_score 소수점 10자리까지 동일한 채로 32회 재생성됐고 매번 holdout에서 거부됐다 — jump→α+=1.0→다음 cycle 당첨 확률↑→재생성→다시 jump→... 자기강화 루프가
confirm 결과와 무관하게 돌아간 것으로 설명된다. `reflect()`도 같은 결함 — confirm 이전 raw label을 그대로 lesson에 반영해 strategist가 "이 방향 성공했다"고 계속
학습했다.
- 한계: bandit은 decay(0.95)가 있는 노이즈 추정기라, 이 보정은 이전에 쐈던 α+=1.0을 수학적으로 정확히 되돌리는 게 아니라 β 쪽에 새 delta를 더하는 것이다(반대 방향 신호를 추가하는
것) — 여러 cycle에 걸쳐 수렴하는 설계고, 단발 보정으로 즉시 상쇄되진 않는다. 효과(재생성 빈도 감소)는 배포 후 며칠 관찰이 필요하며 즉시 검증 가능한 항목이 아니다.

## ADR-031 — auto-submit은 confirmed pipeline만 제출

- 결정: `bin/api.py:_best_attempt()`가 `raw.attempts` 전체 max cv_score가 아니라 `raw.pipelines`(cross-seed+holdout 확정,
`invalid_reason is null`)로 후보를 제한한다. 확정 pipeline이 없는 대회는 auto-submit이 "no confirmed pipeline"으로 skip한다.
- 대안: escape hatch(`bin/submit.py --attempt-id`, 미확정 attempt를 사람이 명시 지정하는 용도)를 auto-submit에서 그대로 두고 `_best_attempt()`에
별도 신뢰도 필터만 추가 — 필터 기준을 새로 설계해야 하고, escape hatch의 원래 의도(사람의 명시적 override)와 자동화 경로가 계속 뒤섞여 기각.
- 근거: `_start_submission`이 항상 attempt_id를 넘기고 `_kaggle_submit`이 이를 `--attempt-id`로 그대로 전달하는데, 이 플래그는 원래 "사람이 명시적으로 고른
attempt(미확정이어도 허용)" escape hatch였다(`bin/submit.py` docstring). 자동화가 매번 이 경로를 타면서 promotion 게이트(cross-seed 재현+holdout
감사)가 자동 제출에는 사실상 무의미해졌다 — reflexion 사이클은 3-attempt 단위로 돌고 그 승자만 유의성 검정을 통과해야 confirm이 실행되는데, `_best_attempt()`는 그 검증과
무관하게 역대 전체 attempt 중 cv_score 최댓값을 고른다. 수백~수천 회 시도 중 fold 노이즈로 우연히 최고값이 나온 attempt가 뽑히는 건 통계적으로 사실상 필연(multiple
comparisons/승자의 저주)이고, 그게 그대로 Kaggle에 제출됐다. 실측(2026-08): 완료 제출 74건 중 confirmed 출처는 29건(39%)뿐, 61%는 검증을 한 번도 통과한 적 없는
코드였다.
- 한계: `_kaggle_submit`/`bin/submit.py` 자체는 변경하지 않았다 — confirmed attempt의 code_path는 cross-seed/holdout 평가 시점에 이미
self-contained 파이프라인으로 검증되고 merge-verify까지 통과했으므로 `raw.pipelines.code`와 기능적으로 동등하다고 보고 CSV 캐시(`download_submission_csv`,
attempt_id 키)도 그대로 재사용한다. 확정 pipeline이 아예 없는 대회(콜드스타트 등)는 auto-submit이 완전히 멈추는데, 이건 이미 ADR-025(phantom-max 폐지)가 처리하는
콜드스타트 baseline 확립 경로(`establish_bootstrap_baseline`/`bin/establish_baseline.py`)에 의존한다.

## ADR-032 — fleet은 27개 동시 운영에서 deep tier 5개로 좁힌다 (breadth-first 폐기)

- 결정: `config/competitions/*.py`에 `ACTIVE` bool을 추가한다. `ACTIVE=False`인 대회는
  `bin/run_daemon.py:_sweep_queue_refill`의 idle 재보급 대상에서 제외된다(#227, Milestone
  v1.6.0). 27개 중 5개만 `ACTIVE=True`로 유지: `s6e8`(binary/auc), `s4e12`(regression/rmsle),
  `s4e10`(binary/auc), `s5e4`(regression/rmse), `s4e11`(binary/accuracy). 나머지 22개는
  attempts 이력을 보존한 채 동결한다(삭제 아님 — 이력 보존 원칙은 ADR-025와 동일).
- 선정 기준: (1) 최근 confirmed 갱신(0.4~24일 이내, "아직 얕은 과실이 남아있는" 대회 우선),
  (2) `#225` 스파이크 실험으로 헤드룸이 실측 검증된 대회(s4e10), (3) task_type/metric 조합
  다양성(binary·regression × auc·rmsle·rmse·accuracy 4종 커버 — 새 역량(원본데이터/stacking/
  튜닝)이 특정 문제 유형에만 통하는지 아닌지 판별 가능하도록).
- 대안: 27개 전체 유지, 컴퓨트만 증설 — 2026-08-23 진단(3중 병렬 조사)이 이미 반증. attempts
  최다 3개 대회(s5e3 3381/s4e1 2956/s4e10 2897, fleet 평균 6~8배)가 정확히 가장 오래
  정체된 대회(마지막 개선 이후 각각 2566/2271/1224회, 60~67일 무변화) — 컴퓨트를 더 부어도
  같은 얕은 툴킷을 반복 재탕할 뿐 전환율이 안 오른다는 게 실측으로 확인됨.
- 근거: breadth-first 전략의 원래 근거는 ADR-010("cross-competition transfer를 1등 시민으로")의
  "경험이 쌓일수록 새 대회 콜드스타트가 빨라진다"는 가설이었는데, `#76`(2026-08-02) 실측이 이미
  이 가설을 반증했다(retrieval 사용/미사용 median gain 완전 동일). breadth를 유지할 근거가
  없어진 상태에서 27-way 분산은 attempts만 늘리고 confirmed pipeline 전환율(0.31%, 29857건→94건)은
  그대로인 낭비였다. Milestone v1.6.0의 새 역량(#228 원본데이터/#229 선언형 모델/#230 Optuna
  튜닝/#231 진짜 stacking)은 attempt당 비용이 900s 예산보다 큰 별도 레인(#230)을 요구하므로,
  이를 27개 대회에 동시 적용할 컴퓨트 여유가 없다 — 소수에 집중해야 검증 자체가 가능하다.
- 한계: 동결된 22개 대회는 새 역량의 혜택을 못 받는다. deep tier에서 새 역량이 검증되면 순차
  확대할지, 계속 5개로 유지할지는 v1.6.0 로드맵 완료 후 재검토(4주 성공 기준 참고).

## ADR-038 — 제출 예산은 CV 개선이 아니라 정보 획득으로 배분한다

- 결정: `bin/api.py:auto_submit`의 대상 선정을 "대회 전역 best가 바뀌었나"에서 "오늘 남은 예산으로 무엇을 배우나"로 바꾼다. 대상 대회는
`comp.ACTIVE`(ADR-032 deep tier)를 직접 읽고, 대회별 일일 예산(`SUBMISSIONS_PER_DAY`, 기본 2) 안에서 **아직 한 번도 제출된 적 없는 confirmed
pipeline을 cv 상위순으로** 내보낸다. 미제출 백로그가 없을 때만 기존 "best 갱신 + 유의성 검정" 경로로 떨어진다. 예산은 별도 카운터 테이블이 아니라
`raw.kaggle_submissions`의 당일(UTC, `status <> 'error'`) 건수를 직접 세서 집행한다.
- 대안: (a) 유의성 게이트를 그냥 완화 — 게이트의 원래 목적(fold 노이즈 수준 재제출로 LB가 랜덤워크하는 것 방지)이 그대로 살아있어 기각. 지금 문제는
게이트가 너무 빡세다는 게 아니라 **예산이 남는데도 안 쓴다**는 것이다. (b) 같은 best를 매일 재제출 — LB 점수가 같아 정보량이 0이다. (c) 죽어 있던
`raw.submission_budget` 카운터 테이블을 살려서 예산 집행 — 카운터는 드리프트하지만 제출 이력은 안 하므로 기각하고 테이블은 삭제했다.
- 근거: 2026-08-26 실측으로 deep tier 5개의 일일 제출이 0~1건(가용 25건/일의 3%)이었고, 스킵 사유는 전부 `best unchanged` / `gain not significant`
였다. 동시에 **아직 한 번도 제출 안 한 confirmed pipeline이 deep tier에만 37건**(s4e10 14 / s4e11 7 / s5e4 6 / s6e8 6 / s4e12 4) 쌓여
있었다. 역대 누적 cv-LB 쌍이 78건뿐인 상태에서, 이 백로그는 공짜로 얻을 수 있는 47%의 추가 표본이다. ADR-003 amend(#104)가 이미 "CV 자체가 오염될 수
있으니 LB가 유일한 외부 검증 신호가 되는 경우가 있다"고 인정했는데, 그 검증 신호를 예산의 3%만 쓰고 있었다.
- 대상 대회를 `ACTIVE`로 바꾼 이유: 기존엔 "최근 24h 내 attempt가 있는 대회"로 간접 판정했다(`bin/run_daemon.py:331-340` 주석이 "그게 실제로 원하는
동작"이라고 적어둔 것). 그 간접 판정은 attempt가 안 도는 날엔 제출할 백로그가 있어도 대회를 통째로 빠뜨린다 — 실제로 2026-08-25 실행에서 deep tier
5개 중 s5e4가 대상 목록에서 아예 빠졌다.
- 한계: 미제출 백로그를 소진하고 나면(예상 4일 내) 제출 빈도는 다시 confirmed 승격 속도에 묶인다. 백로그가 마른 뒤에도 주 1회 LB 갱신이 안 되면
"매일 제출할 만한 새 후보를 만드는" 쪽(튜닝 레인/모델 다양성)이 진짜 병목이라는 뜻이므로, 그때는 제출 로직이 아니라 생성 쪽을 봐야 한다.
- ADR-031(confirmed pipeline만 제출)은 그대로 유지된다 — 이 결정이 바꾸는 건 "몇 건"이지 "무엇을"이 아니다.

## ADR-034 — model_swap도 선언형 프리미티브(`model_spec`), `ensemble_spec`과 레지스트리 공유

- 결정: `model_swap`(+ `ensemble`/`bootstrap`) action_type에 7번째 훅 `model_spec(self, ctx) -> dict | None`을
추가한다. Patch는 `{"model": <registry key>, "params": {...}}`만 반환하고, 생성자 호출은 `evaluator/models.py`
(ADR-023의 앙상블 멤버 레지스트리를 `_ENSEMBLE_MODEL_REGISTRY`에서 이 모듈의 `MODEL_REGISTRY`로 승격·공유)가
전담한다. 레지스트리를 `extra_trees`/`elastic_net`으로 확장했다. 기존 자유형 `build_model`은 병행 허용.
`PatchedPipeline.ensemble_spec`/`model_spec`은 서로 상대 훅(및 `build_model`)의 존재를 상호 억제 조건에 넣어,
base가 어느 한쪽을 정의해도 patch가 다른 쪽(또는 자유형)을 선택하면 그 의도가 상속에 가려지지 않게 한다
(#239가 고친 ensemble_spec/build_model 상호 억제를 model_spec까지 대칭 확장).
- 대안: (a) `model_swap`은 그대로 자유형 `build_model`만 허용 — ADR-023이 해소한 문제(super() 오용, stale
kwarg, 하드코딩된 오래된 클래스명)가 단일 모델 교체에도 동일하게 재발하므로 기각. (b) 레지스트리를
`ensemble_spec` 전용으로 남기고 `model_spec`은 별도 레지스트리를 새로 둔다 — 멤버 모델과 단일 모델이 실제로
같은 라이브러리 집합을 쓰므로 분리할 이유가 없고 드리프트 위험만 커져 기각.
- 근거: ADR-023의 논거(exec된 코드 내부 실패는 harness가 볼 수도 안전망을 못 걸 수도 없음)가 앙상블 멤버뿐
아니라 단일 모델 교체에도 그대로 적용된다 — model_swap의 build_model도 같은 클래스의 실패(제거된 kwarg,
오탈자 클래스명)를 겪어왔다. 레지스트리 자체를 harness/models.py 공용 모듈로 옮겨 ensemble_spec과 model_spec이
동일한 `build_registry_model`/`construct_with_kwarg_retry`를 공유하게 하면, 레지스트리 확장(신규 모델 추가)이
두 훅 모두에 자동으로 적용돼 드리프트가 구조적으로 불가능해진다.
- 한계: 레지스트리에 없는 모델은 `model_spec`으로 표현 불가 — 자유형 `build_model`이 여전히 유일한 선택지.
`EvalResult.model_type`(model_spec이 선언한 레지스트리 이름, 자유형/ensemble은 None)을 attempt 테이블에
영속화해 모델 다양성을 추적할 수 있게 했으나 아직 분석 대시보드는 없다.

## ADR-035 — Optuna 튜닝은 confirmed pipeline의 model_spec/ensemble_spec만 대상, harness 로직은 그대로 재사용

- 결정: `evaluator/tuner.py`(#230)는 raw.pipelines의 확정 pipeline이 `model_spec` 또는 `ensemble_spec`을
정의한 경우에만 튜닝 가능하다 — 자유형 `build_model` pipeline은 튜닝 대상에서 제외한다. 튜닝은 `PipelineContext`에
X/y/scoring을 추가하는 대신, trial마다 `model_spec`(또는 `ensemble_spec`의 특정 멤버)만 오버라이드하는 얇은
어댑터 객체(`_SingleModelTrialPipeline`/`_EnsembleMemberTrialPipeline`)로 confirmed pipeline을 감싸고
`evaluator/harness.py:evaluate_pipeline`을 그대로 호출한다 — CV 방법론(is_original 인지 분할, leak 가드,
스케일 누수 가드)이 attempt 평가와 완전히 동일하게 유지된다. 결과는 `raw.tuned_params`에 영속화하고,
`PipelineContext.tuned_params`(추가 필드, `best_params`와 동일한 advisory 계약)로 다음 attempt에 흘려보낸다.
- 대안: (a) 자유형 `build_model` pipeline도 params만 치환해 튜닝 — LLM이 wrapper 클래스 안에서 params를
어떻게 소비하는지(생성자 kwarg인지, 내부에서 무시하는지) harness가 정적으로 알 수 없어 안전하게 오버라이드할
방법이 없다(ADR-023/ADR-034가 이미 해소한 문제의 재발). (b) `PipelineContext`에 X/y/scoring을 노출해 Coder가
직접 Optuna를 호출하는 훅을 만든다 — 이슈 배경이 명시한 대로 "탐색 책임을 LLM에 되돌리면 지금 없애려는 실패
패턴(90s 예산 안에서 대충 만든 3~12개 후보 argmax)이 그대로 재발"하므로 기각. (c) 튜닝 전용 별도 CV 로직을
harness와 독립적으로 구현 — attempt 평가와 다른 방법론으로 측정한 "개선"이 실제 attempt에서 재현 안 될 위험이
있어 기각, harness 재사용이 유일하게 안전한 선택.
- 근거: model_spec/ensemble_spec(ADR-023/034)이 이미 "harness가 신뢰 코드로 모델을 직접 생성"하는 경계를
만들어뒀다 — 이 경계 덕분에 trial마다 params만 바꿔 끼우는 얇은 어댑터가 안전하게 가능해졌다. #230이 #229를
전제조건("keystone")으로 삼은 이유가 바로 이것 — model_spec 없이는 trial마다 안전하게 오버라이드할 지점
자체가 없었다.
- 한계: 자유형 build_model 확정 pipeline은 model_swap action으로 model_spec 기반 대안이 먼저 confirmed돼야
튜닝 가능해진다 — 즉시 전체 fleet에 적용되지 않고, model_spec 채택이 선행돼야 하는 순차적 의존성이 있다.
튜닝 프로세스 자체는 확정 pipeline의 preprocess/feature_transform/postprocess_predictions을 그대로 신뢰하고
in-process(runtime/isolate.py의 네트워크 네임스페이스 격리 없이) 실행한다 — 이미 contract 검증+cross-seed
확정을 통과한 코드를 다회 재실행하는 것이므로 신규 LLM 코드 최초 실행보다 위험이 낮다고 판단했으나,
attempt 평가와 동일한 격리 수준은 아니다(트레이드오프: 수백 trial을 subprocess당 감싸면 매 trial마다
parquet 왕복+프로세스 기동 비용이 튜닝의 효율성 이점을 상쇄한다).

## ADR-036 — ensemble_spec `method="stack"`은 inner-fold OOF + 강제 회귀 meta, `bin/blend.py` 폐기

- 결정: `ensemble_spec`에 `"method": "stack"` + `"meta": {"model": ..., "params": {...}}`를 추가한다
(`evaluator/harness.py:_fit_predict_stack`, #231). 멤버는 outer CV fold의 train을 inner K-fold(기본 5)로 나눠
만든 out-of-fold(OOF) 예측으로 meta를 학습하고, 실제 검증/제출 예측 시엔 outer train 전체로 재적합한 멤버를
쓴다(표준 stacking 관례 — meta가 멤버의 train-fit 성능이 아니라 일반화 성능을 학습, 누수 방지). meta는
`resolve_model_class(name, is_classification=False)`로 **항상 회귀 변형**을 강제 생성한다 —
`metric_class`가 `binary_proba`/`regression_error`일 때만 지원(연속값 출력 필요). 대체된 파이프라인 밖
사후 blend(`bin/blend.py`, Ridge, `raw.blend_weights`)와 그 dashboard 패널을 함께 삭제한다 —
`bin/submit.py`가 그 값을 전혀 소비하지 않는 죽은 코드였음을 코드로 재확인.
- 대안: (a) meta도 `ctx.is_classification`을 따라 분류 변형으로 생성 — `ridge` 같은 일부 레지스트리
분류 변형(`RidgeClassifier`)은 `predict_proba` 자체가 없어 즉시 깨지고(#229/#239에서 이미 겪은 함정과
동일 클래스), 있는 모델이어도 "연속 확률을 이산화해 다시 분류"하는 손실 없는 정보를 버리는 셈이라 기각.
(b) discrete label(`metric_class == "classification"`)도 stacking 지원 — 멤버 출력이 문자열/카테고리
라벨이라 회귀 meta로 조합할 연속 공간 자체가 없고, 분류 meta를 쓰려면 라벨 인코딩·멀티클래스 처리가
추가로 필요해 범위가 커져 기각 — `majority_vote`가 이미 그 케이스를 커버. (c) `bin/blend.py`를 남겨두고
stacking과 병행 — 두 메커니즘이 같은 문제(예측 조합)를 다른 층위(파이프라인 밖 vs 안)에서 풀면서 하나는
실제 제출에 배선조차 안 돼 있어 유지 비용만 있고 이득이 없어 기각.
- 근거: `bin/blend.py`는 여러 **확정 pipeline**의 OOF를 사후에 Ridge로 섞는 방식이었는데, 그 가중치를
저장(`raw.blend_weights`)만 하고 `bin/submit.py`가 읽지 않아 프로덕션 제출에 전혀 영향을 주지 못했다
(`docs/spec.md` §1.13 기존 인지 사항, 이번에 코드로 재확인). `ensemble_spec`은 이미 harness가 신뢰
코드로 모델을 생성·적합하는 경계를 갖고 있어(ADR-023), 같은 경계 안에 OOF 기반 meta 학습을 추가하는
것이 파이프라인 밖에서 별도 스크립트로 사후 조합하는 것보다 구조적으로 더 안전하고, 실제 제출
경로(`_fit_predict_ensemble` → `fit_predict`, #226/#239)가 이미 그대로 재사용된다.
- 한계: inner K-fold OOF 계산 때문에 멤버 수 × inner_splits(5)만큼 추가 fit이 필요해 `weighted_average`/
`majority_vote`보다 확실히 느리다 — attempt의 900s 예산 안에서 멤버 수가 많거나 데이터가 크면 시간 초과
위험이 있다(deep tier 백테스트로 실측 확인, PR 코멘트 참고). meta 모델 자체의 하이퍼파라미터 탐색은
없다(#230 Optuna 튜닝 레인이 `model_spec` 기반이라 `ensemble_spec`의 `meta`는 아직 튜닝 대상 밖).

## ADR-037 — 훅 1개 제한(ADR-006) 폐지, 합성 가능 훅은 완전 교체 대신 base 실행 후 patch 적용

- 결정: `evaluator/contract.py:validate_patch`가 action_type별 훅 개수를 더 이상 하드 리젝트하지
않는다(ADR-006 뒤집기) — `_ALLOWED_HOOKS`는 `agents/coder.py` 프롬프트의 "주 초점 훅" 가이드로만
남는다. 그 대신 `preprocess`/`feature_transform`/`postprocess_predictions`/`param_candidates`
4개 훅("합성 가능 훅")은, patch가 정의하고 확정 best pipeline도 이미 정의하고 있으면 완전 교체가
아니라 **합성**된다: preprocess/postprocess_predictions는 순차 체이닝(base 실행 후 그 결과에
patch 적용), param_candidates는 리스트 합집합(중복 dict 제거), feature_transform은 컬럼 단위
합집합(base/patch가 각자 같은 원본에서 독립 파생, 동명 컬럼은 patch가 이김 — 타깃 드롭 계약 때문에
순차 체이닝 불가). `build_model`/`ensemble_spec`/`model_spec`은 합성 대상 아님(단일 값이라 조합
불가, 항상 patch가 완전 교체). patch가 `override = ["<hook>", ...]`를 선언하면 그 훅은 합성 대신
완전 교체된다. **이 규칙은 두 곳에서 동일하게 적용된다** — attempt 평가 시점(`evaluator/harness.py:
PatchedPipeline`, base가 순수 `BasePipeline()`이 아니라 이미 축적된 `PatchedPipeline`이고 그 체인
어딘가에 해당 훅의 실제 정의가 있을 때만 합성)과 승격 후 다음 라운드 base 생성 시점
(`cycle/materialize.py`, base/patch 소스를 AST 레벨에서 rename+wrapper 합성). 두 곳이 다르면
측정된 cv_score와 실제 배포 동작이 어긋나는, 이번 세션에 반복 수정한 버그 클래스(#226/#83/#239)가
그대로 재발한다. 부수로 top-level helper 이름 충돌(실제로 다른 정의)도 warning에서 error로
승격했다 — 완전히 동일한 재정의는 모호함이 없어 에러 대상이 아니다(`_real_collisions`).
- 대안: (a) `materialize.py`만 고치고 `PatchedPipeline`은 그대로(측정=배포 일치 안 됨) — attempt가
실제로 측정한 cv_score가 승격 후 실제 동작과 달라지는 심각한 정합성 문제라 기각, 이번 세션에 겪은
버그 클래스와 정확히 같음. (b) 모든 합성 가능 훅을 전부 순차 체이닝 — `feature_transform`은 계약상
타깃을 drop해야 하는데 체이닝하면 두 번째 호출이 이미 없는 타깃을 또 drop하려다 죽어 기각, 컬럼
합집합으로 우회. (c) `_ALLOWED_HOOKS` 완전 삭제(프롬프트 가이드도 없앰) — causal attribution
신호(어떤 action_type이 보통 어떤 훅을 건드리는지)가 Reflector/bandit 쪽에 여전히 유용해 프롬프트
가이드로는 유지, 강제만 풂. (d) base가 `BasePipeline()`이어도 무조건 합성 — feature_transform의
BasePipeline 기본값(타깃 제외 전체 컬럼 그대로 통과)과 합성하면 미인코딩 원본 컬럼이 새고,
param_candidates는 의미 없는 빈 `{}` 후보가 매번 끼어들어(`tests/test_submit.py` 실측 회귀) 기각 —
`_chain_defines`로 base 체인에 그 훅의 실제 정의가 있을 때만 합성.
- 근거: "1변경 규율로 점진 축적"이 의도였지만 실제로는 `materialize.py`의 `{**base, **patch}`가
동명 훅을 통째로 치환해, 같은 훅을 나중 attempt가 다시 건드리면 이전 개선이 LLM이 우연히
복붙했을 때만 살아남았다(#232 배경) — 합성으로 바꾸면 harness 자체가 축적을 보장한다.
훅 개수 제한은 causal attribution(ADR-006 원래 근거)을 위한 것이었는데, 합성이 있으면 여러 훅을
건드려도 이전 개입이 사라지지 않아 제한의 실익이 줄어든다.
- 근거(실측, merge 전 백테스트): 실제 confirmed pipeline(s4e10) 위에 feature_engineering patch
2라운드를 연속으로 얹었을 때, 2라운드 materialize 결과의 `feature_transform`이 실제 실행 시점에
1라운드·2라운드 엔지니어링 컬럼을 **둘 다** 포함함을 확인(이전 로직이면 1라운드 컬럼은 사라졌을 것).
- 한계: `param_candidates` 합성이 base 체인에 이미 여러 라운드 누적되면 탐색 후보 수가 계속
늘어난다 — `_MAX_PARAM_CANDIDATES`(harness.py)가 여전히 캡을 걸지만, 상한 근처에서 새 patch의
후보가 밀려날 수 있다. `feature_transform` 컬럼 합집합은 동명 충돌 시 patch가 무조건 이겨 base의
같은 이름 컬럼이 조용히 사라질 수 있다(의도된 override 시맨틱과 동일 원리이나, 이름을 안 바꾸고
다른 의미로 재사용하면 혼란 여지).

---

## 미정 항목 (TBD)

| 항목 | 제안 | 상태 |
|---|---|---|
| Strategist 모델 | glm-5.2 (대안 deepseek-v4-pro, kimi-k2.6) | ADR-016 |
| Reflector 모델 | kimi-k2.6 (Strategist와 다른 패밀리) | ADR-016 |
| Coder 모델 | gpt-oss:120b (2026-07 qwen3.5:397b에서 변경, 대안 glm-4.7) | ADR-016 (BON-236, BON-240) |
| 스토어 (검색+분석) | Postgres + pgvector (벡터 컬럼) | 확정, ADR-007 amend (BON-98) |
| 벡터 인덱스 | 브루트포스 → 필요 시 pgvector HNSW | 승격 조건부 |
| Ollama Cloud 요금제 | Pro($20, 동시 3) | 시작값 |
| 시작 대회 | playground-series-s4e1 (Bank Churn, 이진/AUC, ~16.5만 행) | 확정 |
| 임베딩 모델 | qwen3-embedding:8b(로컬, 1024d) | 확정, ADR-008 |
| 격리 런타임 | `os.unshare(CLONE_NEWNET)` preexec + rlimit + timeout | 구현 완료, ADR-017 (BON-191) |
| 운용 호스트 | worker-vm + 단일 daemon(systemd) | 확정, ADR-017 (Phase 5) |
| label 임계값 z | fold_std 배수(2.0) | 확정(방어적 기본값), ADR-012 amend (BON-194) — 대회 데이터 축적 후 재캘리브레이션 |
| fingerprint 거리 가중치 | task/metric 큼·size 중간·기타 작음 | TBD (대회 누적 후 캘리브레이션) |
| 외부 아이디어 source 화이트리스트 | 우승 writeup / pinned tips / gold·silver solution | TBD, ADR-019 (큐레이션 필요) |
| 외부 아이디어 추출 LLM | Strategist/Reflector 와 다른 호출 | TBD, ADR-019 |
| 외부 아이디어 주입 빈도·수량 | 주 1회 N=5 (제안) | TBD, ADR-019 |
