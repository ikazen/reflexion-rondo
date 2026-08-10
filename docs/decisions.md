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
- **[2026-08 #104 amend]**: "LB는 확인용"의 전제가 CV가 신뢰 가능하다는 데 있었는데, CV 자체가 오염(ADR-024/025 참조)될 수 있음이 드러나면서 LB가 유일한 외부 검증 신호가 되는 경우가 생겼다. `cv_lb_calibration` 뷰 + 발산 트립와이어를 추가해, CV는 개선인데 LB가 악화된 제출이 나오면 해당 pipeline을 격리하고 대회 auto-submit을 중단한다(ADR-026). CV가 주 신호라는 결정 자체는 유지 — LB는 이제 "확인"을 넘어 CV 신뢰가 깨졌을 때의 차단 신호로도 쓰인다.

## ADR-004 — 추론은 Ollama Cloud, 임베딩은 Mac 로컬 서버
- 결정: Strategist/Reflector/Coder는 Ollama Cloud Pro(`OLLAMA_CLOUD_BASE_URL=https://ollama.com`, Bearer 인증). 임베딩(`qwen3-embedding:8b`)만 Mac Ollama 서버 로컬 유지. 오케스트레이션·저장소·CV·분석 뷰는 WSL2 로컬.
- 코드 라우팅: 에이전트 3개는 `OLLAMA_CLOUD_BASE_URL` + `OLLAMA_API_KEY`, retriever는 `OLLAMA_BASE_URL`만 사용(키 없음).
- 근거: Cloud Pro 모델(deepseek-v4-pro / glm-5 / qwen3-coder-next) 품질이 로컬 14b 대비 명확히 높다고 판단해 전환. 임베딩은 클라우드 키 인가 범위 밖이고 로컬 8b로 충분하므로 분리 유지. ADR-016 패밀리 다양성은 Cloud 전환으로 자동 확보.

## ADR-005 — Evaluator는 결정적 코드
- 결정: 채점은 코드로만. LLM-as-judge 금지.
- 근거: 피드백 객관성. 성찰 오염 방지.

## ADR-006 — `reflexion` 단계에 한해 시도당 변경 1개
- 결정: `reflexion` stage의 attempt만 단일 변경 강제. `bootstrap`/`exploitation`은 예외.
- 근거: 인과 귀속은 유지하되, cold-start 비효율을 피한다.

## ADR-007 — DuckDB 단일 스토어 (검색·분석 통합)
- 결정: 별도 벡터DB(Chroma) 없이 DuckDB 하나에 모두 둔다. 임베딩은 `reflections.embedding`을 `FLOAT[768]` 컬럼으로 저장하고, 검색은 `array_cosine_similarity` 브루트포스 + 메타필터로 수행. 분석 마트도 같은 DB.
- 대안: (a) Chroma 별도 + DuckDB dual-write (v2 초안), (b) pgvector로 전부 Postgres.
- 근거: 이 프로젝트의 벡터 규모는 누적 1만~수만 건 수준이라 768차원 브루트포스 코사인이 수십 ms로 충분 — ANN 인덱스(Chroma/pgvector HNSW)는 조기 최적화. dual-write를 없애 검색·분석·기록의 정합성을 한 트랜잭션으로 보장하고, zero-server(ADR-011)와 DuckDB의 OLAP 강점(마트 window 쿼리)을 동시에 유지. pgvector는 분석까지 Postgres로 옮겨야 이득이 생겨 ADR-011과 충돌하므로 제외.
- 승격 트리거: 벡터 수십만 건 초과로 브루트포스 지연이 체감되면 DuckDB `vss` 확장(HNSW)으로 인덱스만 추가. 그래도 부족하면 그때 전용 벡터DB 재검토.
- **[2026-06-06 amend]** BON-98: DaemonAPI + main loop write-write 충돌(DuckDB 파일락, commit d9ad514) 발동으로 재고 트리거 발동. 대안 (b) pgvector 전환 확정. `store/schema.sql` Postgres 재작성, psycopg2 + pgvector 도입. ADR-011 수정: DuckDB 단일 서버 → ops-vm Postgres + pgvector. 데이터 마이그레이션 없음(새 스키마 시작).

## ADR-008 — 임베딩은 로컬 (qwen3-embedding:8b)
- 결정: `qwen3-embedding:8b`(1024차원, MRL 32~1024 절단 가능)를 Mac Ollama 서버에서 로컬 실행. 스키마는 `reflections.embedding float[1024]`.
- 대안: nomic-embed-text(768d, v2 초안) / embeddinggemma:300m(경량 768d) / qwen3-embedding:0.6b(경량 이전 버전).
- 근거: 검색 품질이 transfer 메커니즘을 직접 좌우하는데, qwen3-embedding:8b가 2026 MTEB v2 오픈웨이트 최상위권이고 Ollama Cloud 키 인가 범위 밖이라 로컬 유지. 0.6b에서 8b로 업그레이드한 것은 교훈 검색 품질 향상이 목적 — 기존 임베딩과 차원(1024d)은 동일하므로 스키마 변경 없음. 저장 압박 시 MRL로 차원 절단.

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
- **[2026-06 amend]** BON-110: 3 attempt 병렬 실행 필요로 Airflow 채택. daemon이 `raw.cycle_queue` 폴링 후 Airflow DAG `reflexion_rondo_cycle`을 트리거. DAG는 retrieve → attempt_0/1/2 (병렬) → promote 4태스크 구조. `AIRFLOW_URL` 없는 direct daemon mode는 운영 대체 경로가 아니라 로컬 smoke/test용 단일 attempt fallback이다. MLflow는 여전히 미사용.
- **현재 상태:** 운영 store는 DuckDB가 아니라 ops-vm Postgres + pgvector다(ADR-007 amend). daemon은 큐/API/페이싱을 맡고, 운영 attempt 병렬화는 Airflow가 맡는다.

## ADR-012 — label은 결정적 임계값으로 계산, Reflector 판정은 참고용
- 결정: `label`(jump/neutral/regression)·`gain_vs_best`는 **Evaluator가 CV 델타와 fold 분산으로 결정적 계산**한다. Reflector(LLM)의 정성 판정은 `reflector_label`로 별도 기록하되, 마트·검색의 진실값으로는 쓰지 않는다.
- 대안: Reflector가 label을 직접 부여 (v2 초안).
- 근거: jump/regression 판정은 점수 움직임에 대한 채점이므로 ADR-005(LLM-as-judge 금지)에 귀속된다. Playground는 fold 노이즈 수준의 델타 싸움이라 "노이즈 vs 진짜 점프" 경계를 임계값으로 명시해야 한다. 정성 판정은 디버깅 참고로 가치가 있어 폐기하지 않고 분리 보관.
- **[2026-07 amend]** BON-194: `LABEL_Z` 1.0 → 2.0. 1σ는 통계적으로 유의하지 않아 노이즈가 상시 "jump"로 라벨링되고 그 노이즈가 `reflection_impact` 검색 부스팅에 그대로 반영됐다. 2.0σ를 방어적 기본값으로 확정, 대회 데이터 축적 후 fold_std 실측 분포로 재캘리브레이션 예정.
- **[2026-07 amend]** BON-267: jump 판정 기준을 harness의 절대-마진(`delta > LABEL_Z * fold_std`)에서 promotion 게이트와 동일한 paired per-fold 유의성 검정(`is_significant_gain`, BON-247)으로 통일. 절대-마진 기준은 수렴한 대회에서 사실상 도달 불가해(7447건 중 jump 0건 실측) label과 promotion 판정이 어긋났고, 그 결과 bandit·stagnation·reflection이 전부 "성공 신호 0"으로 고착돼 있었다. `cycle/run.py`에서 eval 직후 `is_significant_gain`으로 label을 재확정(harness가 절대-마진으로 jump 판정했더라도 paired 미달이면 neutral로 강등)하며, `LABEL_Z` 자체는 promotion 용도로 유지.

## ADR-013 — 생성 코드는 컨테이너/nsjail로 격리 실행
- 결정: Coder가 생성한 `class Patch`는 격리 런타임에서 실행. 시간/메모리 상한, 네트워크 차단, FS 화이트리스트.
- 대안: timeout만 적용 / 신뢰 후 직접 실행.
- 근거: cron 무인 루프에서 LLM 생성 코드를 실행하므로 OOM·행·우발적 네트워크 접근이 워커를 죽이거나 환경을 오염시킬 수 있다. 격리 경계가 안정성과 재현성을 보장한다.
- **현재 구현(BON-191):** `runtime/isolate.py`가 `runtime/runner.py`를 subprocess로 실행. preexec_fn에서 `os.unshare(CLONE_NEWNET)`으로 network namespace를 분리해 subprocess egress 차단(DockerOperator `cap_add=["SYS_ADMIN"]` 필요). CAP_SYS_ADMIN 없는 환경(로컬 mac 등)에선 조용히 스킵. rlimit(AS/CPU) + timeout 병행. env allowlist가 시크릿 env 제거.
- amend(BON-275, 2026-07-19): 타임아웃 600s→1200s 상향. s5e5(75만 행) 5-seed bagging이 기존 600s를 넘겨 매번 타임아웃으로 실패하던 문제 — eval과 동일한 값으로 통일.

## ADR-014 — Coder 컨트랙트는 class Patch + hook 분리
- 결정: 산출물은 `class Patch` 하나. action_type에 허용된 훅(hook)만 구현하고, 나머지는 현재 best pipeline이 fallback으로 제공한다. 훅은 `preprocess` / `feature_transform` / `param_candidates` / `build_model` / `postprocess_predictions` / `ensemble_spec` 6종. IO/k-fold 하니스/파라미터 선정은 Evaluator가 소유.
- 대안: 전체 스크립트 자유 생성 / `feature_fn`+`model_fn` 두 함수 분리 (이전 방식).
- 근거: hook 분리는 action_type 귀속을 코드 레벨로 강제하고(feature_transform만 바꾸는 게 feature_engineering), 1변경 규율을 컨트랙트로 보장한다. best pipeline을 base class로 두고 patch가 단일 훅만 override하면 cold-start seed 코드도 안전하게 재사용 가능. `validate_patch()` (AST 레벨)가 실행 전 위반을 차단한다.
- **[2026-06 BON-113]**: `feature_fn`+`model_fn` → `class Patch` with hooks로 전환. `materialize_best_pipeline()`이 이전 best와 신규 patch를 AST 레벨에서 병합해 누적 pipeline을 유지한다.
- **[2026-07-05 BON-268 amend]**: `validate_patch()`(AST 정적 검사)에 pandas-only API 금지(`.groupby`/`.map_dict`/`.take`/`.apply`/`.iterrows`/`.applymap`/`.get_dummies` — polars 1.41.2 실물에 `hasattr`로 대조해 확정, `value_counts`는 polars Series에 실존해 의도적으로 제외) + candidate patch 자신의 undefined-name 검사(실행 격리 모델과 동일 범위)를 추가. `agents/coder.py` 프롬프트에도 동일 금지 목록 반영.
- **[2026-07-22 #42 amend]**: 정적 검증은 코드 생성 *이후*에만 컨트랙트 위반("action_type=X may not implement hooks: [...]")을 잡아 재시도해도 같은 실수가 반복됐다(s6e7 실측: model_swap이 feature_transform을 구현하려는 시도 다수). `agents/coder.py.generate_code()`가 `evaluator/contract.py._ALLOWED_HOOKS`(source of truth)를 직접 import해, 매 호출 user 메시지에 이번 action_type이 허용하는 hook만 동적으로 강조하도록 변경 — 생성 이전 단계 가드 추가. 같은 커밋에서 multiclass 라벨 왕복(round-trip) 규칙도 컨트랙트에 명시: 타깃을 정수로 인코딩했으면 `postprocess_predictions`에서 원래 문자열 라벨로 되돌려야 한다(`ValueError: Mix of label input types` 방지, 실측 45건).
- **[2026-08 #74 amend]**: `ensemble` action_type의 자유형 wrapper 클래스(직접 fit/predict 구현)는 실행이 exec된 클래스 몸체 내부라 정적 검증도 런타임 안전망도 못 미치는 크래시(super() 오용, 생성자 stale kwarg, 자기 fit() 안에서 하위 모델 재구성)를 계속 냈다. 6번째 훅 `ensemble_spec(self, ctx) -> dict | None`을 추가해 Patch는 "무엇을 조합할지"(멤버 모델·파라미터·결합 방식)만 선언하고, harness(`evaluator/harness.py`)가 모델 생성·적합·결합을 전담한다. 자유형 `build_model` 기반 ensemble 훅은 병행 허용 — 크래시율 재비교 후 폐기 여부 결정. 상세는 ADR-023.

## ADR-015 — 인과 귀속은 상관 기반으로 시작, ablation 보류
- 결정: 자가 개선 효과는 `reflection_impact` 상관 + 1변경 규율 + 실제 채택 교훈 id로 추정. retrieval ON/OFF ablation은 도입하지 않는다.
- 대안: 인터리브 ablation / 별도 memory-OFF 대조 대회.
- 근거: 초기 단순성 우선. 단 이는 인과 증명이 아니라는 한계를 명시하고(`architecture.md` §4), 신호가 모호하면 ablation을 후속 과제로 승격한다.
- **[2026-07 amend]** BON-195: 상관 기반 귀속은 "그냥 인과가 약하다" 수준을 넘어 **자기강화(rich-get-richer) 편향**을 갖는다 — avg_gain 높은 교훈이 `_apply_impact_score`에서 부스팅되어 더 자주 검색되고, 그 결과 avg_gain이 계속 유지/상승하는 루프가 생긴다. 완화책: (1) z-score를 배치 로컬이 아닌 전역(`reflection_impact` 전체) 통계로 계산해 배치 구성에 따른 흔들림 제거(`memory/retriever._global_gain_stats`), (2) `_IMPACT_W` 0.25 → 0.15로 부스팅 강도 감쇠. attempt gain을 인용된 교훈에 균등 배분하는 근본 문제(Coder 변경분과 교훈 기여분 미분리)는 미해결 — ablation 도입 시 함께 재검토.
- **[2026-07-22 #43 amend]**: 전역 z-score 통계(위 amend)가 metric 스케일을 구분하지 않고 `avg_gain`을 pool한다는 별도 문제 발견 — rmse degenerate 예측(모델이 완전히 빗나감)이 만드는 원시 `gain_vs_best`(s6e1 실측 -105448, baseline rmse~8.75 대비)가 전역 std를 부풀려 auc/accuracy 스케일 교훈들의 z-score를 0으로 수렴시켰다. 근본 원인은 `evaluator/harness.py`에서 처리: 기존 "baseline보다 100배 좋으면 스케일 누수로 raise"(`_REGRESSION_IMPLAUSIBLE_BASELINE_RATIO`) 가드에 대칭으로 "100배 나쁘면 gain_vs_best를 하한 클립"을 추가(raise 아님 — label 판정은 클립 전 delta로 유지, DB에 저장되는 값만 스케일 폭주 차단). `_global_gain_stats`의 metric_class별 분리는 이 가드로 전역 std가 안정되는지 배포 후 재측정한 뒤 필요시 별도 판단(reflection_impact가 reflection_id 단위 집계라 분리가 비자명함 — 단순함 우선).
- **[2026-08 #97 amend]**: `_REGRESSION_IMPLAUSIBLE_BASELINE_RATIO`를 100 → 10으로 하향. s5e5가 이 가드보다 먼저 `_check_preprocess_target_leak`(ADR-024)에 잡혔어야 할 preprocess 누수였는데도 100배 문턱을 통과해(실측 gain 44배) 승격까지 갔던 사례가 계기 — 이 비율은 preprocess 누수 검사가 못 잡는 결과 기반 2차 방어선이라 문턱을 낮춰도 정상 개선을 오탐할 여지가 적다고 판단.
- **[2026-07-22 #58 amend]**: 위 가드(클립)로도 `reflection_impact` 전역 z-score 오염(mean=-4.22, std=139.19 실측)이 해소되지 않음을 확인 — 클립은 극단값만 완화할 뿐 metric 스케일 자체(rmse 원시 단위 vs auc 0~1)를 정규화하지 않아 근본 해결이 아니었다. `gain_vs_best_relative` 컬럼(regression_error는 `gain_vs_best/baseline_cv` 상대값, 나머지는 패스스루) 신설로 전역 통계를 metric 스케일 상대화로 교체하고, `reflection_impact` 뷰가 이 컬럼만 집계하도록 재정의(값이 없는 legacy row는 제외, raw `gain_vs_best`로 폴백하지 않음).

## ADR-016 — LLM 역할별 모델 배정 (Actor 분리 + Reflector 패밀리 다양성)
- 역할 매핑: Reflexion의 **Actor = Strategist(정책) + Coder(실행)**, **Self-Reflection = Reflector**, Evaluator는 결정적 코드(ADR-005).
- 결정: **처음부터 3모델 분리.**
  - **Strategist**(정책, 추론 모델) — `glm-5.2` (2026-06-24 `deepseek-v4-pro`에서 변경. 대안 `deepseek-v4-pro`, `kimi-k2.6`).
  - **Reflector**(성찰, 추론 모델) — `kimi-k2.6` (대안 `glm-5`). **Strategist와 다른 패밀리**로 고정 — glm(Strategist) ≠ kimi(Reflector) 유지.
  - **Coder**(실행, 코드 모델) — `gpt-oss:120b` (2026-07-02 `qwen3-coder-next`→`qwen3.5:397b`, 2026-07 `qwen3.5:397b`→`gpt-oss:120b` 재변경. 대안 `glm-4.7`).
- 근거: Coder 분리는 코드 특화 모델이 컨트랙트 준수에 유리(태스크 성격). Reflector를 다른 패밀리로 두는 건 **상관된 맹점** 완화 — 같은 모델이 가설을 내고 스스로 성찰하면 자기 추론을 합리화한다. ADR-005가 채점에서 LLM을 뺐어도 Reflector의 정성 진단·generality 라벨링엔 자기편향이 남으므로 교차 패밀리가 교훈 품질을 높인다.
- **[2026-07 amend]** BON-236: `qwen3-coder-next` deprecate 예정으로 `qwen3.5:397b`로 교체. 태그 확정 전 ops-vm에서 cloud `/api/tags` 실측 조회로 정확한 문자열 확인함(웹 검색은 `qwen3.5:397b-cloud`로 나왔으나 실제 API 응답은 bare `qwen3.5:397b` — BON-188 `glm-5.2:cloud` 접미사 오타 전례 재발 방지).
- **[2026-07 amend]** BON-240: `qwen3.5:397b`가 동일 프롬프트에서 `qwen3-coder-next` 대비 출력 토큰 9배(reasoning 과다)로 사이클당 지연·비용이 커짐. 같은 시기 코더 전문 라인(`qwen3-coder-next`/`480b`, `devstral` 계열)이 Ollama Cloud에서 전부 내려가 `gpt-oss:120b`로 교체. `MODEL_CODER_REASONING_EFFORT`(기본 `medium`)로 reasoning 강도 조절.
- 비용: 세 역할 모두 사이클당 1회라 분리해도 호출 수는 안 늘고 설정만 는다. 처음부터 교차 패밀리 critic을 확보하는 편이 교훈 품질에 유리하다고 보고 단계적 분리(2→3)는 두지 않는다. 단순 베이스라인이 필요하면 Reflector를 Strategist 모델로 잠시 묶을 수 있으나 기본은 3모델.
- 주의: 모델 ID는 변동성이 크다. 확정 전 `ollama.com/search?c=cloud`에서 현재 태그 재확인.

## ADR-017 — 무인 24/7 운용은 worker-vm + 단일 daemon (Phase 5)
- 결정: WSL2 로컬 → nexus-prime **worker-vm**(2 OCPU ARM64 / 12GB, always-on)으로 옮겨 무인 상주. 오케스트레이션은 cron이 아닌 **단일 장수 프로세스**(`bin/run_daemon.py` + systemd `Restart=always`). 추론만 Ollama Cloud, **임베딩은 Mac Ollama 유지**(ADR-004/008 재확인). 추적: 마일스톤 Phase 5, BON-67~70.
- 대안: (a) WSL2 유지 + 데몬화 — 데스크톱이라 진짜 24/7 아님(슬립/재부팅), dev 머신과 충돌. (b) mac-server — M1 10c/32GB로 강하지만 intermittent(가정 NAT)라 무인 호스트 부적합. (c) cron + 파일락 유지(ADR-011).
- 근거:
  - **호스트**: 이 루프의 wall-clock은 Ollama Cloud 추론 대기가 지배하고 처리량은 Cloud rate-limit(5h 세션 + 주간 cap)이 이미 throttle한다. 따라서 always-on 노드는 약해도(2 ARM/12GB) 충분 — "강한 CPU"보다 "진짜 24/7"이 우선. nexus-prime에 ML 전용 여유 노드는 없고 worker-vm이 유일한 여유 always-on 노드(airflow edge-worker만 상주).
  - **daemon > cron** (ADR-011 정련): 단일 24/7 워커에선 데몬이 cron 중첩/DuckDB 파일락(runbook §4) 문제를 제거하고, Ollama 페이싱 상태를 메모리에 들고 self-throttle 한다. ADR-011의 "Prefect 승격은 워커 ≥3"은 유지(BON-24) — 데몬화는 승격이 아니라 단일 워커의 단순화.
  - **임베딩 Mac 유지**: 임베딩은 매 사이클 retrieve+persist 2회로 빈번하다. Cloud로 보내면 추론 3역할에 써야 할 한도/과금을 갉아먹어 사이클 처리량 자체가 준다. Mac이 사실상 always-on이므로 ADR-004/008 분리를 그대로 둔다. 단 Mac 일시 불통(슬립)에 대비해 daemon은 임베딩 호출에 retry/backoff, 실패 시 해당 사이클만 스킵(크래시 금지).
  - **격리 = subprocess `os.unshare(CLONE_NEWNET)`** (ADR-013의 "컨테이너 vs nsjail" TBD 확정, BON-191): 초기 설계의 "Docker `--network none`"은 컨테이너 레벨을 의미했으나, eval/task 컨테이너 자체는 Postgres/MinIO/Ollama 접근에 네트워크가 필요하다. 격리 경계는 **생성 코드 subprocess**. Python 3.12 `os.unshare(CLONE_NEWNET)`을 preexec_fn에서 호출해 subprocess에게 격리된 network namespace를 부여 — subprocess에서 나가는 모든 연결 차단. 컨테이너에 `cap_add=["SYS_ADMIN"]` 필요(네트워크 namespace 생성용). eval 컨테이너는 시크릿 마운트 없음(secrets는 Airflow Variable env로만 주입, allowlist가 제거). mem/cpu/timeout 상한으로 OOM 리스크를 흡수. OOM·타임아웃은 워커 사망이 아니라 `error_trace`→교훈이 된다.
  - **Postgres 영속 + 백업**: Postgres raw 스키마의 competitions/attempts/reflections/pipelines가 누적 교훈이자 transfer 자산이다. 백업 대상은 DuckDB 파일이 아니라 ops-vm Postgres 데이터와 MinIO/로컬 code artifact다.
- 한계: 12GB는 대형 데이터셋에서 빠듯 — 격리 컨테이너 mem-limit로 OOM을 lesson화해 흡수하되, 빈발하면 mac-server 디스패치(하이브리드)를 재고한다.

## ADR-018 — 통합 웹은 aggregator 패턴 (각 워크로드 자체 API + ops-vm 통합 UI)
- 결정: kaggle.<your-domain> 공개 웹은 각 워크로드가 read/admin API를 자체 제공하고, ops-vm의 별도 aggregator 웹(신규 repo `aggregator-web`(이름 미정), 신규 Linear 프로젝트)이 두 API를 호출해 단일 UI로 렌더한다. 공유 publish layer는 두지 않는다. rondo daemon은 Postgres raw 스키마 위에 FastAPI 라우터를 제공하고, droid controller는 자체 API endpoint만 합의한다. 추적: BON-75~77 (rondo 측) + droid BON-72 재정의 + 신규 aggregator 프로젝트.
- 대안:
  - (a) **공유 Postgres publish layer + 통합 웹**: daemon이 attempt 요약을 Postgres에 push → ops-vm 웹이 직접 read. dual-write 일관성·schema 강제 결합·pgvector 등 새 컴포넌트 부담.
  - (b) **별개 웹 ×2, 같은 Postgres**: 도메인 단절을 인스턴스 단절로 표현. publish layer 부담은 (a)와 동일.
  - (c) **각 워크로드 API + aggregator** (선택): publish layer 폐기, 각 워크로드의 자체 store가 그대로 truth.
- 근거:
  - **도메인 단절 결정과 일관** (2026-06-04): lesson/work_unit을 두 시스템에 분리하기로 한 시점에 "공유 store"의 의미가 약해졌다. transfer가 불가능한 두 도메인(kaggle 노하우 ↛ 게임 조작)을 한 스키마에 묶을 이유 없음. API contract만 공통.
  - **자체 store truth 유지**: rondo는 Postgres raw 스키마를 truth로 유지한다. 별도 publish hook·마이그레이션 도구·dual-write 일관성 검증이 불필요한 것이 가장 큰 단순화.
  - **repo 자율성**: 각 워크로드가 스키마/저장 방식을 자유롭게 진화. contract 변경 시에만 aggregator 동기 업데이트.
  - **보안 모델 자연**: daemon API는 tailnet only, public 노출은 ops-vm aggregator 하나로 집중. worker-vm은 공인 IP 없음(infra-lookup 결과)이라 외부 노출 자체가 불가능하므로 이 분할이 강제이자 이득.
  - **자랑 의도 달성**: 한 페이지에서 두 시스템 표시.
- 한계/위험:
  - **API contract 합의 비용 1회**: 양 repo가 공통 응답 모양 합의. JSON schema 1장 + 명시적 버전.
  - **daemon에 HTTP 추가**: rondo daemon은 LLM 호출이 동기 블로킹이면 API 응답 지연. asyncio 또는 워커 스레드 분리로 흡수. ADR-017의 "단일 장수 프로세스" 문구는 깨지지 않음(같은 프로세스 안 라우터 추가).
  - **aggregator 가용성 의존**: 한 API down 시 부분 표시(degraded mode 처리 필요).
  - **인증 라우팅**: viewer 무인증 / admin path는 별도 internal 도메인 또는 Caddy의 path matcher로 tailnet only 분리. SSO 미도입(nexus-prime R6 트리거 미발동) 가정.

## ADR-019 — 외부 아이디어 채널: 분리된 게이트웨이 + 톰슨 샘플링 노출

> 현재 코드에는 `raw.external_ideas`, `external_idea_bandit`, Strategist 프롬프트 통합이 아직 없다. 아래는 채택된 설계 방향이다.

- 결정:
  - Kaggle 우승 writeup / pinned tips / 유사 fingerprint 대회 솔루션 스레드 등 외부 소스에서 추출한 ML 아이디어를 별도 테이블 `raw.external_ideas` 에 보관하고 Strategist 프롬프트에만 별도 섹션으로 노출. **reflections 풀 / `reflection_impact` 마트 / 검색 score 가중치에 일절 섞지 않는다.**
  - **외부 아이디어 자체는 `verified` / `promoted` 마킹 없이 영구 게이트웨이로 둔다.** 검증 가치는 채택된 사이클의 정상 reflection 이 lessons 풀에 들어가는 것으로만 살린다 — 시스템 기본 루프가 곧 승격 경로.
  - **복합 아이디어를 atomic action 으로 쪼개지 않는다.** `idea_text` 원문 보존, Strategist 가 읽은 후 자기 출력에서 `action_type` 을 결정. 외부 단에는 enum 강제 없음(추출 LLM 의 `probable_action_type` 은 nullable 추정값에 한함).
  - **노출은 stage 게이팅 + 톰슨 샘플링 (Beta-Bernoulli)** — `bootstrap`/`exploitation` 은 외부 idea 차단, `reflexion` 단계만 노출. `applies_when` fingerprint 1차 필터 → 각 후보 θᵢ ~ Beta(αᵢ, βᵢ) 샘플 → **top-3 노출**. 모든 idea 균일 Beta(1, 1) prior. 채택 + jump → α++, 채택 + regression → β++, **미채택 무변화** (Strategist 가 단지 다른 걸 선호했을 뿐, 실패 신호 아님).
  - **Archive 정책 = 자동 idea + 수동 source 분리**: idea 단위는 `trials ≥ 10 AND posterior_mean < 0.1` 자동 archive(sustained-bad 자연 감쇠). source 단위는 사람이 사후 평가해 화이트리스트 업데이트(자동화 안 함 — 좋은/나쁜 스레드 판단은 측정만으로 부족).
  - **추출 LLM 가드 4개 모두 적용**: (i) 실측 수치 인용 있는 글만, (ii) upvote/다수 동의 임계값 통과, (iii) 조건부 진술만(절대 추천 차단), (iv) `idea_text` 500자 상한 + 코드 블록 분리. 추출 단계 시스템 프롬프트에 포함.
- 대안:
  - (a) **lessons 풀에 `source='external'` + L3_general 직접 삽입**: 검색 단일 채널이라 우아하지만 earned knowledge 오염. `gain_vs_best` 가 null 이라 `reflection_impact` 계산 분기 필요. Strategist 가 "측정된 교훈"과 "남의 주장"을 구분 못함.
  - (b) **시드 코드/큐로 변환** (cold-start path 연장): LLM 이 아이디어를 `class Patch` 까지 만들어 큐잉. 검증 안 된 코드가 Coder/Evaluator 비용 부담. cold-start 는 유사 대회 `gain_vs_best > 0` 검증 코드라 외부 아이디어와 신뢰 수준이 다름.
  - (c) **분리된 게이트웨이 + Strategist 프롬프트 힌트** (선택).
  - 노출 정책 대안: **정적 top-K (최근 N일 + score 정렬)** — 외부 source 수가 주 N=5 수준이라 빈번한 후보 중복. 톰슨 샘플링이 적은 풀에서 exploration/exploitation 균형을 자동 학습.
- 근거:
  - **천장 상승의 외부 경로**: 현 시스템은 Strategist+Coder prior 안에서만 가설을 낸다(ADR-014). 누적 학습은 "알려진 도구를 이 데이터셋에 언제·어떻게 적용할지" 의 조건부 정책일 뿐, 새 기법 도입 경로가 없다. 외부 주입이 prior 천장을 들어 올리는 가장 직접적인 수단.
  - **분리 = ADR-005 정합**: LLM-as-judge 금지의 본질은 "측정 안 된 LLM 판정을 진실값으로 못 쓴다". 외부 아이디어는 정확히 측정 안 된 주장 — earned reflection(측정의 자연어 추상화)과 같은 풀에 두면 retrieval score 가중치(`sim * (1 + avg_gain)`)와 `reflection_impact` 인과 귀속(ADR-015)이 동시에 오염된다. 분리하면 두 자산 모두 손상 없음.
  - **Cold-start lessons(ADR-010) 와 구분**: cold-start lessons 는 *우리 시스템이 다른 대회에서 측정한* L2/L3 reflection(검증 자산). external_ideas 는 *외부인의 미검증 주장*. 둘 다 "다른 컨텍스트의 교훈을 현재 대회에 끌고 옴"이지만 신뢰 수준이 달라 풀·검색·프롬프트 섹션 모두 분리.
  - **노출 대상은 Strategist 만**: Coder 에 외부 텍스트(특히 코드 조각)를 노출하면 검증 안 된 코드 직접 카피 위험 — `prev_code` 위 1변경 컨트랙트(ADR-014)가 깨짐. Reflector 에 노출하면 자기 추론을 외부 주장으로 정당화 — ADR-016 의 cross-family critic 효과가 깎임. 외부 신호는 가설 생성 단계에만 영감으로 들어가고, 구현·성찰은 측정 결과에만 기반하도록 분리.
  - **자연 승격**: 외부 아이디어 → Strategist 채택 → Coder 실행 → Evaluator 결정적 신호 → Reflector 정상 reflection. 승격 경로가 시스템 기본 루프와 동일해서 별도 검증 로직 불필요.
  - **Atomic 분해 안 함의 함의**: Strategist 가 복합 아이디어를 단일 사이클에 다 적용 못함 — 1변경 규율(ADR-006)과 충돌하지 않음. 같은 idea 가 다음 사이클에 다시 톰슨 샘플링으로 뽑히면 다른 부분 / 다른 `action_type` 으로 채택 가능. 의도된 점진 분해.
  - **톰슨 샘플링 fit**: 외부 source 가 주 N=5 수준으로 적어 정적 top-K 는 빈번한 후보 중복. Beta-Bernoulli 가 α/β 만으로 사후 효과를 누적해 좋은 idea 자동 식별, 나쁜 idea 자연 감쇠. **α/β 자체가 모니터링 상태라 별도 채택률 마트 불필요** — 분석 뷰 `external_idea_bandit` 하나로 노출.
  - **source 품질이 결정적**: Kaggle Discussions hotness 는 일화 잡음이 커서 LLM 추출이 false signal 을 양산. 화이트리스트 — (i) 종료된 유사 fingerprint 대회 우승 writeup(ADR-010 fingerprint 매칭과 결합), (ii) 대회 pinned "Tips & Tricks", (iii) gold/silver solution 스레드. 양보다 질(주당 N=5 수준).
  - **추출 LLM 분리**: 같은 모델이 가설/성찰/외부 추출 셋을 다 하면 다양성이 0. Strategist/Reflector 와 다른 호출. 빈도가 주 1회 수준이라 비용은 무시 가능.
- 데이터 모델 (스키마 상세는 `spec.md`):
  - `raw.external_ideas(idea_id, fetched_at, source_url, source_kind, idea_text, probable_action_type NULLABLE, applies_when_json, confidence, alpha float default 1.0, beta float default 1.0, archived, adopted_attempt_ids)`
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

- 결정: daemon+task 이미지의 **빌드+registry push**는 airflow-stack의 `reflexion_rondo_deploy` DAG(ops 큐 docker.sock 재사용, airflow-stack decisions.md L29)가 담당한다. 이 repo의 `deploy/release.sh`는 더 이상 ops-vm에 SSH해서 빌드하지 않는다 — registry에 해당 태그가 이미 존재하는지 확인만 하고, daemon의 실제 컷오버(compose.yml 태그 bump+재시작)만 수행한다.
- **태그의 source of truth가 이미지별로 갈라진다**: daemon은 계속 git(`deploy/compose.yml`, 이 repo)이 진실이고 release.sh가 push한다. task는 **Airflow Variable**(`rondo_task_image_version`, airflow-stack 관리)이 진실이 되고, `reflexion_rondo_deploy` DAG가 빌드 직후 즉시 bump한다 — git push도 GitDagBundle의 60초 지연도 없다.
- 근거:
  - **여러 repo 재사용**: reflexion-rondo뿐 아니라 다른 repo도 같은 방식(ops 큐 docker.sock)으로 이미지 배포를 하게 될 예정 — repo마다 ops-vm 상주 체크아웃+전용 SSH 키를 만드는 대신, `dags/lib/image_deploy.py`(airflow-stack) 공용 헬퍼가 매 실행마다 임시 디렉터리로 clone→build→push 한다.
  - **신규 credential 불필요**: 이 repo가 public이라 clone에 인증이 없고, `registry.internal:5000`도 무인증(HTTP insecure, tailnet 경계로만 보호)이다. private repo가 이 메커니즘을 쓰게 되면 그때 공유 read-only PAT이 필요해진다(이 repo엔 해당 없음).
  - **daemon은 남겨둔 이유**: daemon의 "배포"는 compose.yml 태그 bump(git write, 이 repo)+ops-vm 재시작이라 airflow-stack의 credential 경계 밖 작업이다. Airflow DAG에 이 repo의 git write credential을 새로 심는 대신, 지금처럼 사용자 로컬(WSL) git credential로 release.sh가 처리하는 편이 새 credential 없이 끝난다.
- 트레이드오프:
  - **DAG Versioning과의 결합 약화** (airflow-stack ADR-L27 참조): task 이미지 태그가 Variable로 빠지면서, 특정 DagRun이 정확히 어떤 이미지로 돌았는지 이제 git log가 아니라 Airflow의 rendered-template을 봐야 안다.
  - **daemon/task 버전 불일치 창**: 두 이미지가 같은 커밋에서 함께 빌드되지만, task Variable은 즉시 반영되고 daemon은 release.sh를 별도로 돌릴 때까지 구버전으로 남을 수 있다. 둘 다 repo 전체를 COPY해 빌드하므로 코드 차이는 없지만 "지금 어느 버전이 떠있나"를 daemon/task 따로 확인해야 한다.
- cross-ref: issue #15(release.sh 사전검증 순서 수정, 이번 daemon 전용 버전에도 유지), issue #17(release.sh 축소), airflow-stack decisions.md L29/R2.

## ADR-023 — ensemble은 선언형 프리미티브(`ensemble_spec`), 자유형 wrapper와 병행

- 결정: `ensemble`/`bootstrap` action_type에 6번째 훅 `ensemble_spec(self, ctx) -> dict | None`을 추가한다. Patch는 `{"members": [{"model": <registry key>, "params": {...}}, ...], "method": "weighted_average"|"majority_vote", "weights": [...]}`만 반환하고, 모델 생성·적합·결합은 `evaluator/harness.py`가 고정 레지스트리(`lgbm`/`xgboost`/`catboost`/`hgb`/`random_forest`/`ridge`)로 전담한다. 기존 자유형 `build_model` 기반 ensemble(직접 wrapper 클래스 작성)은 병행 허용한다.
- 대안: (a) 몽키패치로 자주 나오는 wrapper 실수 패턴을 사후 교정 — 범위·리스크가 커서 보류. (b) 자유형을 전면 금지하고 `ensemble_spec` 강제 — 기존에 잘 동작하는 wrapper까지 막을 근거가 없어 기각.
- 근거: `ensemble` 크래시는 exec된 클래스 몸체 내부(super() 오용, 생성자 stale kwarg 하드코딩, 자기 fit() 안에서 하위 모델 재구성)에서 나서 정적 검증도 `_build_model_safe` 같은 런타임 안전망도 원천적으로 못 미쳤다 — harness가 그 코드를 볼 수 없기 때문. 선언형으로 바꾸면 멤버 구성·적합·결합이 전부 신뢰 코드 안에서 일어나 이 클래스의 크래시가 구조적으로 사라진다.
- 한계: 레지스트리에 없는 모델은 `ensemble_spec`으로 표현 불가 — 그런 경우는 자유형 경로가 여전히 유일한 선택지. 두 경로의 에러율을 배포 후 재비교해 자유형 폐기 여부를 판단한다.

## ADR-024 — audit holdout을 추론조건(dummy target)으로 재현하고 승격 차단 게이트로 승격

- 결정: `runtime/runner.py:_eval_holdout`이 holdout10의 타깃을 실제 추론(`bin/submit.py`)과 동일하게 dummy 상수로 치환한 뒤 preprocess/feature_transform을 태우고, 채점만 원본 타깃으로 한다. `ConfirmResult.holdout_regressed`를 신설해 `confirmed`에 AND 결합 — 후보 holdout이 현재 best(콜드스타트면 BasePipeline) 대비 악화되면 승격을 거부한다. baseline holdout을 측정 못 하면(에러) "정보 없음"으로 보고 막지 않는다(악화 확정과는 다름).
- 대안: holdout을 기록만 하고 게이트에 안 씀(기존 동작) — cross-seed confirm이 이미 승격 신뢰도를 담당한다고 봤으나, cross-seed는 seed만 바꾼 CV라 seed 불변 누수(ADR-025)에 장님이라는 게 드러났다.
- 근거: 기존 holdout은 타깃이 살아 있는 채로 파이프라인을 통과해 preprocess의 valid-target 의존 누수를 실제 추론 조건과 다르게(=누수를 그대로 재현하며) 측정했다. dummy target 치환으로 holdout이 실제 추론 조건의 복제가 되면서, 이 게이트 하나로 누수뿐 아니라 train/test skew 전반을 승격 전에 잡는다.
- 한계: holdout10 자체가 없는 대회(데이터가 90/10 split을 감당 못함)는 이 게이트가 작동하지 않는다.

## ADR-025 — 누수 파이프라인은 삭제 아닌 격리, baseline은 확정 파이프라인만(phantom-max 폐지)

- 결정: `raw.pipelines.invalid_reason`(text, nullable) 컬럼을 추가해 확정 후 누수로 밝혀진 행을 격리 표시한다(삭제하지 않음 — 이력 보존). 모든 baseline 조회 경로(`cycle/run.py:_prev_best`/`_prev_best_fold_scores`, `bin/blend.py`, `bin/submit.py`, `cycle/materialize.py`)는 `invalid_reason IS NULL` 필터를 공유한다. 확정 파이프라인이 하나도 없던 대회를 위해 "전체 attempt의 max(cv)로 폴백"하던 phantom-max 분기를 제거하고, 대신 (a) bootstrap 종료 시 최고 attempt를 `confirm_and_measure`로 검증해 자동 baseline을 세우는 경로(`cycle/run.py:establish_bootstrap_baseline`)와 (b) 기존 정체 대회를 위한 소급 스크립트(`bin/establish_baseline.py`, top-k 순회 + 첫 통과 승격)를 추가했다.
- 대안: 누수 확정 시 해당 행을 delete — attempt 이력·디버깅 근거가 사라져 기각. phantom-max를 그대로 두고 임계값만 조정 — max 순서통계량은 N이 늘수록 문턱이 같이 올라가 정직한 소폭 개선이 영원히 못 넘는 자기강화 데드락이라 근본 대응이 아니라고 판단.
- 근거: phantom-max는 확정 파이프라인 없는 대회의 콜드스타트를 풀기 위한 임시 봉합이었는데, 수백 draw의 상위 꼬리를 문턱으로 쓰다 보니 그 자체가 새로운 데드락을 만들었다(확정 파이프라인 0건인 대회에서 jump 라벨도 0건으로 실측 상관됨). baseline을 "확정된 것만"으로 좁히고 그 확정 절차 자체를 자동화(bootstrap 종료 시)·소급(기존 대회)으로 나눠 풀면 데드락과 phantom을 동시에 해소한다.
- 한계: `establish_baseline.py`의 top-k 폴백은 순위가 높은 candidate가 phantom(비정상적으로 좋은 CV)이면 cross-seed/holdout/scale-leak 가드에서 순차 탈락하며 다음 순위로 내려간다 — 후보 풀 자체가 얕으면(top-k 전부 phantom) 여전히 baseline을 못 세울 수 있다.

## ADR-026 — cv-LB 발산 트립와이어, 해제는 수동만

- 결정: 뷰 `cv_lb_calibration`(대회별 제출 시계열의 delta_cv/delta_lb/부호 일치)을 신설하고, `bin/api.py:refresh_submission_row`가 제출 결과를 받을 때마다 `_detect_cv_lb_divergence`로 판정한다. CV는 개선인데 LB가 악화된 제출이 나오면 원천 pipeline에 `invalid_reason='cv_lb_divergence'`를 표기하고 해당 대회의 `raw.competitions.auto_submit_paused_reason`을 세워 auto-submit을 중단한다. **자동 해제 없음** — 사람이 원인을 확인하고 `auto_submit_paused_reason`을 직접 NULL로 되돌려야 재개된다.
- 대안: 발산 감지 후 자동으로 이전 baseline으로 롤백 — 발산 원인이 다양해서(진짜 누수/우연한 shake-up/데이터 drift) 자동 판단이 오히려 위험하다고 판단해 기각.
- 근거: LB 회수가 자동화(ADR-003 amend)되면서 발산을 감지할 수 있게 됐지만, 감지만으로는 재발을 못 막는다 — auto-submit을 계속 돌리면 같은 문제로 반복 소모된다. 수동 해제는 의도적인 마찰이다: 자동 시스템이 "이 pipeline은 신뢰 못 함"이라고 판단했으면, 그 판단을 뒤집는 건 자동화가 아니라 사람의 확인이어야 한다.
- 한계: 해제를 잊으면 해당 대회는 무기한 자동 제출이 멈춘다 — 운영자가 주기적으로 `auto_submit_paused_reason IS NOT NULL`인 대회를 점검해야 한다(`docs/runbook.md`).

## ADR-027 — 격리 subprocess 메모리 상한은 RSS가 아닌 VSZ(RLIMIT_AS) 기준

- 결정: `runtime/isolate.py`의 attempt 격리 subprocess 메모리 상한은 `RLIMIT_AS`(가상 주소공간, VSZ)로 건다. 기본값 6GiB, `EVAL_MEM_LIMIT_BYTES`로 대회/큐별 override 가능(기존 메커니즘 유지).
- 대안: 더 낮은 값(1.5GiB)으로 tight하게 제한 — 실측으로 폐기됨(아래 근거).
- 근거: `RLIMIT_AS`는 물리 RSS가 아니라 가상 주소공간 상한이다. numpy/scipy/sklearn/lightgbm/catboost/xgboost 같은 라이브러리는 실제 쓰는 물리 메모리가 적어도 공유 라이브러리 mmap·BLAS 스레드풀 등으로 VSZ를 널찍하게 예약한다 — 1.5GiB는 이런 라이브러리를 import하는 것만으로 부족해서, 물리 메모리가 12GB나 남는 워커에서도 신규 대회 부트스트랩 attempt 전체가 실패하는 회귀를 냈다. 6GiB로 되돌린 뒤 Airflow 실측(super-cycle 3678건 전수)으로 재확인한 실제 동시성 기준 근소 초과분은 물리 RSS 여유로 흡수 가능하다고 판단.
- 한계: 이 특성 때문에 "물리 메모리가 충분한 환경"에서도 VSZ 예약이 큰 스택(예: WSL2의 다른 Python 배포판)에서는 동일 6GiB가 부족할 수 있다 — 운영 검증된 워커 환경(mac-server/worker-vm/ops-vm task 컨테이너) 밖에서 이 스크립트를 돌릴 땐 먼저 이 한계를 의심할 것.

## ADR-028 — eval CPU 상한은 커널 RLIMIT_CPU가 아니라 부모 폴링 워치독이 집행

- 결정: `runtime/isolate.py`의 attempt 격리 subprocess CPU 예산(기본 900초, `EVAL_CPU_BUDGET_SECS`로 override)은 기존 RSS 워치독과 같은 2초 폴링 루프가 `/proc/<pid>/stat`으로 직접 감시해 초과 시 명시적 `error_trace`("cpu budget exceeded: ...")를 남기고 선제 kill한다. 커널 `RLIMIT_CPU`는 폴링이 놓쳤을 때만 발동하는 soft<hard 백스톱(`budget+60`/`budget+120`)으로 강등한다. `cycle/run.py`의 eval 재시도 루프는 예산을 eval 회차가 아니라 **attempt 전체 기준**으로 집행한다 — 1회차가 예산을 다 쓰면 2회차는 아예 돌리지 않는다.
- 대안: (a) 기존처럼 `RLIMIT_CPU(soft=hard=900)`만으로 집행 — 이번 결정으로 폐기. (b) soft<hard로 켜되 백스톱이 아니라 주 집행 수단으로 유지(SIGXCPU를 자식이 직접 처리) — 자식이 LLM 생성 코드라 신호 핸들러를 신뢰할 수 없어 기각.
- 근거: 리눅스는 `RLIMIT_CPU`의 hard 한도를 soft보다 먼저 검사해서, soft==hard로 걸면 SIGXCPU 경고 단계 없이 곧장 SIGKILL(rc=-9)로 죽인다. 이는 커널 OOM killer 사망과 문자열이 완전히 동일해 원인 구분이 불가능했고, 2026-08-07 처리량 진단이 이걸 전부 OOM으로 오판해 RSS 워치독(#154)을 배포했지만 효과가 없었다(배포 후 2일간 RSS 워치독 발동 1회, 반면 rc=-9 kill은 113건/22.4h=계산의 40%, 성공 attempt 대비 peak RSS는 여유가 컸다 — 메모리가 아니라 CPU가 원인이었음을 실측으로 확인). 게다가 `cycle/run.py`가 이 무의미한 rc=-9 원문을 LLM 재생성 피드백으로 그대로 넘겨 2회차 eval도 같은 자리에서 또 죽었다(rc=-9 attempt 113건 전부가 예외 없이 이 경로) — attempt당 최대 소모가 ~1800초(16분+)까지 갔다. 부모 폴링으로 옮기면 원인이 명시된 에러를 남길 수 있고, attempt 단위 예산으로 재시도의 최악 소모를 절반으로 자르고, 재생성 피드백을 "더 싼 파이프라인을 써라" 같은 실행 가능한 지시로 바꿔 재시도가 낭비가 아니라 실제 성공 기회가 되게 한다. 예산 값 900은 그대로 유지한다 — 성공 attempt의 실측 wall time(p99=728초, max=1112초)이 이 근처라 낮추면 성공을 에러로 바꿀 위험이 있고, 신규 `peak_cpu_sec` 컬럼(성공/실패 무관 기록, `peak_rss_bytes`와 동일 계약)으로 다음 사이클에 재조정할 근거를 모은다.
- 한계: `/proc/<pid>/stat` 폴링은 2초 주기라 그 사이 짧게 폭증하는 CPU 소모는 최대 2초 지연 뒤에야 감지된다(RSS 워치독과 동일한 한계, 실무상 무시 가능). ADR-027(`RLIMIT_AS` 6GiB)은 이 변경과 무관하게 그대로 유지된다.

## ADR-029 — 생성 코드의 `n_jobs=-1` 등 무제한 병렬성은 정적 거부

- 결정: `evaluator/contract.py:validate_patch`에 `n_jobs`/`thread_count`/`num_threads`/`nthread`/`n_threads` 키워드 인자가 0 이하 리터럴(전부/거의 전부 코어 요청)이면 거부하는 AST 검사를 추가한다. 값이 변수·표현식이면 판정하지 않는다(기존 pandas-only 검사와 동일하게 과소탐지를 오탐보다 우선).
- 대안: (a) Docker/cgroup 레벨에서 하드 CPU quota로 강제 — airflow-stack `reflexion_rondo_cycle.py`의 `cpus=1.5`가 이미 이 의도였으나 docker provider가 `cpu_shares`(상대 가중치)로만 반영해(`CpuQuota=0` 실측 확인) 하드 캡이 아니다. 근본적으로는 더 맞는 위치지만 다른 repo(airflow-stack) 작업이고 fleet 처리량이 슬롯 대비 포화 상태가 아니라 급하지 않음 — 별도 후속. (b) 이 값 그대로 두고 무시 — 아래 실측 때문에 기각.
- 근거: `OMP_NUM_THREADS=2`/`OPENBLAS_NUM_THREADS=2`/`MKL_NUM_THREADS=2`(`deploy/Dockerfile`)는 BLAS/OpenMP 레벨 스레딩만 제한하고 이 파라미터들과는 무관하게 동작한다는 걸 이 세션에서 직접 재현: `OMP_NUM_THREADS=2` 환경에서 LightGBM `n_jobs=-1`은 20 threads/15.9x cores-equivalent, CatBoost `thread_count=-1`은 21 threads/15.0x, scikit-learn `RandomForestClassifier(n_jobs=-1)`은 43 threads/15.6x를 썼다. Airflow attempt 컨테이너의 CPU 상한이 위 대안(a)처럼 사실상 장식이라, LLM 생성 코드 하나가 이 파라미터를 쓰면 같은 호스트(특히 big 큐 슬롯 2개가 4vCPU를 공유하는 mac-server)의 sibling attempt를 실제로 굶길 수 있다. #159(eval CPU 예산 워치독)의 kill은 이 문제를 해결하지 못한다 — attempt 자신의 CPU-초 소진을 더 빨리 채워 자신은 더 빨리 죽지만, sibling이 그 사이 굶는 것 자체는 막지 못한다.
- 한계: 이름 기반 정적 lint라 `getattr`/문자열 조합/변수 경유 등으로 우회 가능하다(파일 상단 docstring — 보안 경계 아님, 정직한 실수를 값싸게 재생성으로 돌려보내는 soft guard). 실제 강제 경계는 여전히 Docker/cgroup 레벨이어야 하며, 그 후속(대안 a)이 남아 있다.

## ADR-030 — bandit/lesson 보상 신호는 attempt-time label이 아니라 confirm 결과를 반영

- 결정: `cycle/promotion.py`에 순수 함수 `effective_label(original_label, confirm)`을 추가한다. `label=="jump"`인데 `confirm.confirmed is False`(cross-seed 미재현 또는 holdout 악화)면 하류 학습 신호(`update_bandit` 호출의 `label`, `reflect()`에 넘기는 `AttemptContext.label`)에는 `"regression"`으로 다운그레이드한다. `raw.attempts.label`(attempt 생성 시점 DB 값)은 건드리지 않는다 — 그 시점의 잠정 판정은 그대로 사실이고, 다운그레이드 대상은 오직 나중에 계산되는 보상/lesson 신호뿐이다. `cycle/run.py`(직접모드, `run_attempt_core`)와 `bin/run_promote_task.py`(프로덕션 airflow 모드) 양쪽에 적용한다.
- 대안: (a) 같은 아이디어의 재생성을 코드 내용(hash) 기준으로 캐싱해 재검증을 건너뛰기 — 증상(반복되는 22분짜리 confirm)만 가리고 원인(왜 같은 아이디어가 계속 최고 후보로 뽑히는가)은 안 건드린다. 바이트 단위로 동일한 코드만 잡아 사소한 변형에는 무력하다 — 기각. (b) `update_bandit`을 confirm 완료까지 지연 — 프로덕션은 retrieve/attempt×3/promote가 별도 Airflow task/컨테이너라 "승자가 될지" 자체가 attempt 생성 시점엔 미정이라 구조적으로 불가능. 대신 승자에 한해 confirm 이후 보정 delta를 추가하는 현재 설계를 택함.
- 근거: `cycle/action_optimizer.py:update_bandit`은 `label=="jump"`에 α+=1.0(최강 보상)을 준다. `cycle/run.py`가 `defer_promotion` 여부와 무관하게 이 함수를 무조건 호출하는데, confirm(cross-seed+holdout, `bin/run_promote_task.py`에서 승자만 별도로 나중에 실행)은 이 시점에 아직 모른다 — 실제로 `bin/run_promote_task.py`는 `update_bandit`을 아예 호출하지 않았다(grep 확인, #164 전). 그 α/β를 소비하는 `assign_super_cycle_actions`는 `get_action_prior`(advisory, LLM이 최종 결정)와 달리 LLM 개입 없는 결정론적 top-N 배정이라 안전판도 없다. 실측(2026-08-09~10): s6e1의 `preprocessing` 후보가 cv_score 소수점 10자리까지 동일한 채로 32회 재생성됐고 매번 holdout에서 거부됐다 — jump→α+=1.0→다음 cycle 당첨 확률↑→재생성→다시 jump→... 자기강화 루프가 confirm 결과와 무관하게 돌아간 것으로 설명된다. `reflect()`도 같은 결함 — confirm 이전 raw label을 그대로 lesson에 반영해 strategist가 "이 방향 성공했다"고 계속 학습했다.
- 한계: bandit은 decay(0.95)가 있는 노이즈 추정기라, 이 보정은 이전에 쐈던 α+=1.0을 수학적으로 정확히 되돌리는 게 아니라 β 쪽에 새 delta를 더하는 것이다(반대 방향 신호를 추가하는 것) — 여러 cycle에 걸쳐 수렴하는 설계고, 단발 보정으로 즉시 상쇄되진 않는다. 효과(재생성 빈도 감소)는 배포 후 며칠 관찰이 필요하며 즉시 검증 가능한 항목이 아니다.

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
