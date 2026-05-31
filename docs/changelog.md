# 변경 이력

## LLM/임베딩 모델 배정 (2026-05-31)
- ADR-016 신설: 역할별 모델 배정. Reflexion Actor = Strategist(정책) + Coder(실행), Reflector = self-reflection.
  - Strategist `deepseek-v4-pro` / Coder `qwen3-coder-next` 시작값.
  - Reflector는 Strategist와 추론 모델 공유로 시작(2모델) → 메타 루프 데이터 후 **다른 패밀리**로 분리(3모델). 근거: 상관된 맹점 완화.
- ADR-008 개정: 임베딩 `nomic-embed-text`(768d) → `qwen3-embedding:0.6b`(1024d, MRL). 2026 MTEB v2 오픈웨이트 최상위. 스키마 `reflections.embedding` → `float[1024]`.
- 모델 ID 변동성 주의: 확정 전 ollama.com/search?c=cloud 재확인.

## DuckDB 단일 스토어로 통합 (2026-05-31)
- ADR-007 개정: 별도 벡터DB(Chroma) + dual-write 폐기. DuckDB 하나가 기록·검색·분석을 모두 담당.
- 임베딩은 `reflections.embedding` 벡터 컬럼에 저장, 검색은 `array_cosine_similarity` 브루트포스 + 메타필터.
- 근거: 누적 1만~수만 벡터 규모에선 브루트포스가 수십 ms로 충분 → ANN 인덱스는 조기 최적화. dual-write 정합성 부담 제거 + zero-server(ADR-011) 유지.
- 승격 트리거: 수십만 건 초과 시 DuckDB `vss`(HNSW) 인덱스 추가.
- 영향: `memory/vector_store.py` → `memory/retriever.py`, setup의 Chroma 초기화 제거.

## 설계 문서 구조화 (2026-05-31)
- `init.md`(v2 설계서)를 표준 문서 구조로 분산: README, architecture, decisions, spec, tasks, setup, runbook. `init.md` 제거.
- 결정 추가:
  - ADR-012 label은 결정적 임계값(Evaluator)으로 계산, Reflector 판정은 `reflector_label` 참고 컬럼으로 분리.
  - ADR-013 생성 코드는 컨테이너/nsjail 격리 실행.
  - ADR-014 Coder 컨트랙트를 `feature_fn` + `model_fn`으로 분리.
  - ADR-015 인과 귀속은 `reflection_impact` 상관 기반으로 시작, ablation 보류(한계 명시).
- spec 보강: 지표 레지스트리, label·gain 결정 규칙, Coder 컨트랙트 시그니처, 실행 전 검증 게이트, `reflector_label` 컬럼.

## v1 → v2 (설계)
- 대회 간 경험 전이(transfer)를 1등 시민 모듈로 격상.
- 새 대회 cold-start 절차 정의, "시도당 1변경" 규율의 예외(bootstrap/exploitation) 명시.
- 운영 스택 단순화: Airflow → 단순 Python 러너 + cron, MLflow 제거, "Cloud A/B" 추상 제거.
- Polars/Optuna 본문 편입, 일반 전략 노트는 `docs/strategy.md`로 분리.
