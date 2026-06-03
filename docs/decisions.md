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

## ADR-012 — label은 결정적 임계값으로 계산, Reflector 판정은 참고용
- 결정: `label`(jump/neutral/regression)·`gain_vs_best`는 **Evaluator가 CV 델타와 fold 분산으로 결정적 계산**한다. Reflector(LLM)의 정성 판정은 `reflector_label`로 별도 기록하되, 마트·검색의 진실값으로는 쓰지 않는다.
- 대안: Reflector가 label을 직접 부여 (v2 초안).
- 근거: jump/regression 판정은 점수 움직임에 대한 채점이므로 ADR-005(LLM-as-judge 금지)에 귀속된다. Playground는 fold 노이즈 수준의 델타 싸움이라 "노이즈 vs 진짜 점프" 경계를 임계값으로 명시해야 한다. 정성 판정은 디버깅 참고로 가치가 있어 폐기하지 않고 분리 보관.

## ADR-013 — 생성 코드는 컨테이너/nsjail로 격리 실행
- 결정: Coder가 생성한 `feature_fn`/`model_fn`은 격리 런타임에서 실행. 시간/메모리 상한, 네트워크 차단, FS 화이트리스트.
- 대안: timeout만 적용 / 신뢰 후 직접 실행.
- 근거: cron 무인 루프에서 LLM 생성 코드를 실행하므로 OOM·행·우발적 네트워크 접근이 워커를 죽이거나 환경을 오염시킬 수 있다. 격리 경계가 안정성과 재현성을 보장한다.

## ADR-014 — Coder 컨트랙트는 feature_fn + model_fn 분리
- 결정: 산출물을 단일 스크립트가 아닌 두 컨트랙트로 분리. `feature_fn(train, valid, target) -> (Xtr, Xval)`, `model_fn(params) -> estimator`. IO/k-fold 하니스는 Evaluator가 소유.
- 대안: 전체 스크립트 자유 생성 / 단일 `fit_predict`.
- 근거: 누수 방지(폴드 내 feature fit)를 컨트랙트로 강제하고, `action_type` 귀속을 깔끔히 하며(피처 변경 vs 모델 변경 분리), cold-start에서 `raw.pipelines` 코드를 안전하게 재사용한다.

## ADR-015 — 인과 귀속은 상관 기반으로 시작, ablation 보류
- 결정: 자가 개선 효과는 `reflection_impact` 상관 + 1변경 규율 + 실제 채택 교훈 id로 추정. retrieval ON/OFF ablation은 도입하지 않는다.
- 대안: 인터리브 ablation / 별도 memory-OFF 대조 대회.
- 근거: 초기 단순성 우선. 단 이는 인과 증명이 아니라는 한계를 명시하고(`architecture.md` §4), 신호가 모호하면 ablation을 후속 과제로 승격한다.

## ADR-016 — LLM 역할별 모델 배정 (Actor 분리 + Reflector 패밀리 다양성)
- 역할 매핑: Reflexion의 **Actor = Strategist(정책) + Coder(실행)**, **Self-Reflection = Reflector**, Evaluator는 결정적 코드(ADR-005).
- 결정: **처음부터 3모델 분리.**
  - **Strategist**(정책, 추론 모델) — `deepseek-v4-pro` 시작값 (대안 `kimi-k2.6`).
  - **Reflector**(성찰, 추론 모델) — `glm-5` 시작값 (대안 `kimi-k2.6`). **Strategist와 다른 패밀리**로 고정.
  - **Coder**(실행, 코드 모델) — `qwen3-coder-next` 시작값 (대안 `glm-4.7`, 저비용 `devstral-small-2:24b`).
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
  - **격리 = Docker `--network none`** (ADR-013의 "컨테이너 vs nsjail" TBD 확정): nexus-prime이 Docker는 제공하나 nsjail 표준은 없다. 생성 코드는 순수 compute라 네트워크 차단이 깔끔히 맞고, mem/cpu/timeout 상한으로 12GB ARM의 OOM 리스크를 흡수한다. OOM·타임아웃은 워커 사망이 아니라 `error_trace`→교훈이 된다.
  - **DuckDB 영속 + 백업**: DuckDB 파일 = 누적 교훈 = transfer의 가치. nexus-prime L7("백업 안 함")의 재고 트리거("stateful 신규 서비스 추가")에 정확히 해당하므로, worker-vm 로컬 디스크 + MinIO 야간 스냅샷(systemd timer)으로 백업한다. nexus-prime decisions.md L7에 BON-70 cross-reference.
- 한계: 12GB는 대형 데이터셋에서 빠듯 — 격리 컨테이너 mem-limit로 OOM을 lesson화해 흡수하되, 빈발하면 mac-server 디스패치(하이브리드)를 재고한다.

---

## 미정 항목 (TBD)

| 항목 | 제안 | 상태 |
|---|---|---|
| Strategist 모델 | deepseek-v4-pro (대안 kimi-k2.6) | 시작값, ADR-016 |
| Reflector 모델 | glm-5 (Strategist와 다른 패밀리) | 시작값, ADR-016 |
| Coder 모델 | qwen3-coder-next (대안 glm-4.7/devstral-small-2) | 시작값, ADR-016 |
| 스토어 (검색+분석) | DuckDB 단일 (벡터 컬럼) | 권장(시작), ADR-007 |
| 벡터 인덱스 | 브루트포스 → 필요 시 vss(HNSW) | 승격 조건부 |
| Ollama Cloud 요금제 | Pro($20, 동시 3) | 시작값 |
| 시작 대회 | playground-series-s4e1 (Bank Churn, 이진/AUC, ~16.5만 행) | 확정 |
| 임베딩 모델 | qwen3-embedding:8b(로컬, 1024d) | 확정, ADR-008 |
| 격리 런타임 | Docker `--network none` + mem/cpu/timeout | 확정, ADR-017 (BON-67) |
| 운용 호스트 | worker-vm + 단일 daemon(systemd) | 확정, ADR-017 (Phase 5) |
| label 임계값 z | fold_std 배수(기본 1.0) | TBD (캘리브레이션 필요) |
| fingerprint 거리 가중치 | task/metric 큼·size 중간·기타 작음 | TBD (대회 누적 후 캘리브레이션) |
