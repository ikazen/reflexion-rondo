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

## ADR-013 — 생성 코드는 컨테이너/nsjail로 격리 실행
- 결정: Coder가 생성한 `class Patch`는 격리 런타임에서 실행. 시간/메모리 상한, 네트워크 차단, FS 화이트리스트.
- 대안: timeout만 적용 / 신뢰 후 직접 실행.
- 근거: cron 무인 루프에서 LLM 생성 코드를 실행하므로 OOM·행·우발적 네트워크 접근이 워커를 죽이거나 환경을 오염시킬 수 있다. 격리 경계가 안정성과 재현성을 보장한다.
- **현재 구현(BON-191):** `runtime/isolate.py`가 `runtime/runner.py`를 subprocess로 실행. preexec_fn에서 `os.unshare(CLONE_NEWNET)`으로 network namespace를 분리해 subprocess egress 차단(DockerOperator `cap_add=["SYS_ADMIN"]` 필요). CAP_SYS_ADMIN 없는 환경(로컬 mac 등)에선 조용히 스킵. rlimit(AS/CPU) + 600s timeout 병행. env allowlist가 시크릿 env 제거.

## ADR-014 — Coder 컨트랙트는 class Patch + hook 분리
- 결정: 산출물은 `class Patch` 하나. action_type에 허용된 훅(hook)만 구현하고, 나머지는 현재 best pipeline이 fallback으로 제공한다. 훅은 `preprocess` / `feature_transform` / `param_candidates` / `build_model` / `postprocess_predictions` 5종. IO/k-fold 하니스/파라미터 선정은 Evaluator가 소유.
- 대안: 전체 스크립트 자유 생성 / `feature_fn`+`model_fn` 두 함수 분리 (이전 방식).
- 근거: hook 분리는 action_type 귀속을 코드 레벨로 강제하고(feature_transform만 바꾸는 게 feature_engineering), 1변경 규율을 컨트랙트로 보장한다. best pipeline을 base class로 두고 patch가 단일 훅만 override하면 cold-start seed 코드도 안전하게 재사용 가능. `validate_patch()` (AST 레벨)가 실행 전 위반을 차단한다.
- **[2026-06 BON-113]**: `feature_fn`+`model_fn` → `class Patch` with hooks로 전환. `materialize_best_pipeline()`이 이전 best와 신규 patch를 AST 레벨에서 병합해 누적 pipeline을 유지한다.

## ADR-015 — 인과 귀속은 상관 기반으로 시작, ablation 보류
- 결정: 자가 개선 효과는 `reflection_impact` 상관 + 1변경 규율 + 실제 채택 교훈 id로 추정. retrieval ON/OFF ablation은 도입하지 않는다.
- 대안: 인터리브 ablation / 별도 memory-OFF 대조 대회.
- 근거: 초기 단순성 우선. 단 이는 인과 증명이 아니라는 한계를 명시하고(`architecture.md` §4), 신호가 모호하면 ablation을 후속 과제로 승격한다.

## ADR-016 — LLM 역할별 모델 배정 (Actor 분리 + Reflector 패밀리 다양성)
- 역할 매핑: Reflexion의 **Actor = Strategist(정책) + Coder(실행)**, **Self-Reflection = Reflector**, Evaluator는 결정적 코드(ADR-005).
- 결정: **처음부터 3모델 분리.**
  - **Strategist**(정책, 추론 모델) — `glm-5.2` (2026-06-24 `deepseek-v4-pro`에서 변경. 대안 `deepseek-v4-pro`, `kimi-k2.6`).
  - **Reflector**(성찰, 추론 모델) — `kimi-k2.6` (대안 `glm-5`). **Strategist와 다른 패밀리**로 고정 — glm(Strategist) ≠ kimi(Reflector) 유지.
  - **Coder**(실행, 코드 모델) — `qwen3-coder-next` (대안 `glm-4.7`, 저비용 `devstral-small-2:24b`).
- 근거: Coder 분리는 코드 특화 모델이 컨트랙트 준수에 유리(태스크 성격). Reflector를 다른 패밀리로 두는 건 **상관된 맹점** 완화 — 같은 모델이 가설을 내고 스스로 성찰하면 자기 추론을 합리화한다. ADR-005가 채점에서 LLM을 뺐어도 Reflector의 정성 진단·generality 라벨링엔 자기편향이 남으므로 교차 패밀리가 교훈 품질을 높인다.
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

---

## 미정 항목 (TBD)

| 항목 | 제안 | 상태 |
|---|---|---|
| Strategist 모델 | deepseek-v4-pro (대안 kimi-k2.6) | ADR-016 |
| Reflector 모델 | kimi-k2.6 (Strategist와 다른 패밀리) | ADR-016 |
| Coder 모델 | qwen3-coder-next (대안 glm-4.7/devstral-small-2) | ADR-016 |
| 스토어 (검색+분석) | Postgres + pgvector (벡터 컬럼) | 확정, ADR-007 amend (BON-98) |
| 벡터 인덱스 | 브루트포스 → 필요 시 pgvector HNSW | 승격 조건부 |
| Ollama Cloud 요금제 | Pro($20, 동시 3) | 시작값 |
| 시작 대회 | playground-series-s4e1 (Bank Churn, 이진/AUC, ~16.5만 행) | 확정 |
| 임베딩 모델 | qwen3-embedding:8b(로컬, 1024d) | 확정, ADR-008 |
| 격리 런타임 | `os.unshare(CLONE_NEWNET)` preexec + rlimit + timeout | 구현 완료, ADR-017 (BON-191) |
| 운용 호스트 | worker-vm + 단일 daemon(systemd) | 확정, ADR-017 (Phase 5) |
| label 임계값 z | fold_std 배수(기본 1.0) | TBD (캘리브레이션 필요) |
| fingerprint 거리 가중치 | task/metric 큼·size 중간·기타 작음 | TBD (대회 누적 후 캘리브레이션) |
| 외부 아이디어 source 화이트리스트 | 우승 writeup / pinned tips / gold·silver solution | TBD, ADR-019 (큐레이션 필요) |
| 외부 아이디어 추출 LLM | Strategist/Reflector 와 다른 호출 | TBD, ADR-019 |
| 외부 아이디어 주입 빈도·수량 | 주 1회 N=5 (제안) | TBD, ADR-019 |
