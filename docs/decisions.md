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

## ADR-004 — 추론 역할은 Ollama Cloud, 나머지는 로컬
- 결정: Strategist/Coder/Reflector만 클라우드. 오케스트레이션·저장소·임베딩·CV·dbt는 로컬.
- 근거: 로컬 부담 제거 + 드롭인. v1의 "Cloud A/B" 추상은 조기 분산화로 판단해 제거.

## ADR-005 — Evaluator는 결정적 코드
- 결정: 채점은 코드로만. LLM-as-judge 금지.
- 근거: 피드백 객관성. 성찰 오염 방지.

## ADR-006 — `reflexion` 단계에 한해 시도당 변경 1개
- 결정: `reflexion` stage의 attempt만 단일 변경 강제. `bootstrap`/`exploitation`은 예외.
- 근거: 인과 귀속은 유지하되, cold-start 비효율을 피한다.

## ADR-007 — 이중 기록(dual-write)
- 결정: 교훈을 벡터DB + DuckDB 양쪽에 기록.
- 근거: 검색(벡터)과 분석(SQL) 역할 분리. 디버깅 용이.

## ADR-008 — 임베딩은 로컬
- 결정: nomic-embed-text 로컬 실행.
- 근거: 클라우드 키가 임베딩 미인가 가능 + 고지능 불필요 + 비용 절감.

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

---

## 미정 항목 (TBD)

| 항목 | 제안 | 상태 |
|---|---|---|
| Strategist/Reflector 모델 | gpt-oss:120b / glm-5.1:cloud / deepseek 대형 | TBD |
| Coder 모델 | 시작은 동일 모델, 추후 코드 특화 분리 | TBD |
| 벡터DB | Chroma | 권장(시작) |
| Warehouse | DuckDB | 권장(시작) |
| Ollama Cloud 요금제 | Pro($20, 동시 3) | 시작값 |
| 시작 대회 | 현재 열린 Playground Series 1개 | TBD |
| 임베딩 모델 | nomic-embed-text(로컬) | 권장(시작) |
| 격리 런타임 | 컨테이너 vs nsjail 구체 선정 | TBD |
| label 임계값 z | fold_std 배수(기본 1.0) | TBD (캘리브레이션 필요) |
| fingerprint 거리 가중치 | task/metric 큼·size 중간·기타 작음 | TBD (대회 누적 후 캘리브레이션) |
