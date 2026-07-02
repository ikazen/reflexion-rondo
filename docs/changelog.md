# 변경 이력

## Coder 모델 교체 (2026-07-02)
- `qwen3-coder-next` deprecate 예정으로 `qwen3.5:397b`로 교체(ADR-016 amend, BON-236).
- 태그 확정 전 ops-vm에서 cloud `/api/tags` 실측 조회로 정확한 문자열 확인(bare
  `qwen3.5:397b` — 웹 검색 결과와 다르게 `-cloud` 접미사 없음).

## 문서 정리 (2026-07-01)
- `CLAUDE.md`를 제거하고 정확한 내용을 `README.md`로 통합 — 정본 진입 문서를 README+docs 하나로 유지.
- README 구조 트리·진입점·배포 절차를 현재 코드에 맞춰 재작성 (DuckDB/평면 모듈/`runs/code` 등 stale 서술 제거).
- `docs/setup.md`, `docs/spec.md`, `docs/decisions.md`의 잔여 모델명(`deepseek-v4-pro`/`glm-5`)을 현재 기본값(`glm-5.2`/`kimi-k2.6`)으로 정정.

## 사이클 skip-continue + circuit breaker (2026-06-18)
- dagrun(사이클) 1개 실패 시 큐 전체를 즉시 failed 처리하던 fail-fast를 폐지.
  실패 사이클은 건너뛰고 다음 사이클을 계속 실행한다 (skip-and-continue).
- 연속 `RONDO_MAX_CONSECUTIVE_FAILURES`(기본 5)회 실패 시 큐를 failed로 중단하는
  circuit breaker 추가. 성공 1회로 연속 카운터 리셋.
- 큐 최종 상태: 1개라도 성공 → `done`, 전부 실패/스킵 → `failed`, 연속 한도 초과 → `failed`.
- direct 모드 generic exception도 동일 처리로 통일. `EmbeddingUnavailableError` skip은
  연속 카운터를 건드리지 않는다.

## 현재 구현 정렬 (2026-06-08)
- Store/검색/분석은 Postgres + pgvector 기준. DuckDB 관련 항목은 과거 결정 이력으로만 남아 있다.
- 임베딩 기본값은 `qwen3-embedding:8b`(1024d), Strategist/Reflector/Coder는 Ollama Cloud 모델 상수(`config/settings.py`)를 사용한다.
- 운영 경로는 daemon → Airflow `reflexion_rondo_cycle` super-cycle. direct daemon mode는 로컬 smoke/test용 단일 attempt fallback이다.
- Cross-competition transfer의 fingerprint 검색, cold-start JSON, `raw.pipelines`, `cold_start_progression` 뷰는 부분 구현 상태다.
- ADR-019 external ideas 채널과 제출 예산 자동 게이트는 설계/스키마 일부만 있고 운영 코드 통합은 미구현이다.

## LLM/임베딩 모델 배정 (2026-05-31)
- ADR-016 신설: 역할별 모델 배정. Reflexion Actor = Strategist(정책) + Coder(실행), Reflector = self-reflection.
  - 처음부터 3모델 분리: Strategist `deepseek-v4-pro` / Reflector `kimi-k2.6`(다른 패밀리, glm-5에서 변경) / Coder `qwen3-coder-next`.
  - Reflector를 Strategist와 다른 패밀리로 고정 — 근거: 상관된 맹점 완화(자기 가설을 스스로 합리화하는 편향).
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
