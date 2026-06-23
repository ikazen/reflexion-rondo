# reflexion-rondo

Kaggle 정형 대회용 자기 개선 Reflexion 루프. LLM 3역할(Strategist, Coder, Reflector)이 사이클마다 가설 생성 → 코드 실행 → CV 평가 → 교훈 저장을 반복하고, 저장된 교훈으로 다음 사이클을 개선한다.

## 환경

- Python 패키지 관리: `uv` (`uv run python ...` 형태로 실행)
- 로컬 venv: `.venv/` (repo 내부)
- DB: ops-vm Postgres + pgvector (`raw` 스키마). `RONDO_DB_URL` 환경변수.
- 객체 스토리지: MinIO (생성 코드 `.py` 저장). `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`.
- LLM 추론: Ollama Cloud Pro (`OLLAMA_CLOUD_BASE_URL`, `OLLAMA_API_KEY`).
- 임베딩: Mac Ollama 서버 로컬 (`OLLAMA_BASE_URL`, 키 없음).
- 운영 호스트: worker-vm (Airflow DockerOperator 경유). 로컬에선 direct 모드로 실행.

## 모델 배정

| 역할 | 모델 | 환경변수 |
|------|------|---------|
| Strategist | `glm-5.2:cloud` | `OLLAMA_CLOUD_BASE_URL` + `OLLAMA_API_KEY` |
| Reflector | `kimi-k2.6` (Strategist와 다른 패밀리 — ADR-016) | 동일 |
| Coder | `qwen3-coder-next` | 동일 |
| Embedding | `qwen3-embedding:8b` | `OLLAMA_BASE_URL` (키 없음) |

모델명 단일 소스 = `config/settings.py` 기본값 (평문, 비밀 아님). `MODEL_*` env override는 실험용으로만 남기고 SOPS·Airflow Variable엔 넣지 않는다. 모델 변경 = settings.py 편집 후 두 이미지(daemon, task) 재빌드·재배포.

## 주요 진입점

```bash
# daemon 시작 (큐 폴링)
uv run python -m bin.run_daemon

# 새 대회 등록 + 첫 루프
uv run python -m bin.start_competition --id <slug> --name "<명>" --task binary --metric auc --target <col>
uv run python -m bin.run_reflexion --competition <slug> --stage bootstrap --cycles 5

# 큐 조작 (daemon 실행 중)
curl -X POST http://localhost:8000/api/queue \
  -H 'Content-Type: application/json' \
  -d '{"competition": "<slug>", "stage": "reflexion", "n_cycles": 30}'

# 대시보드
uv run streamlit run dashboard.py

# DB 초기화 (스키마 유지)
uv run python bin/reset.py --competition <slug>
```

## 프로젝트 구조

```
agents/          LLM 역할 (strategist, coder, reflector)
bin/             실행 진입점:
                   run_daemon.py (큐 폴링 + FastAPI + Airflow trigger)
                   run_retrieve_task.py / run_attempt_task.py / run_promote_task.py (Airflow super-cycle 3태스크)
                   run_cycle_task.py (Airflow single-cycle DockerOperator 태스크)
                   run_reflexion.py (로컬/수동 러너)
                   start_competition.py (대회 등록)
                   archive_lessons.py (저효율 교훈 자동 archive)
                   healthcheck.py (의존성 헬스체크 + /api/health 재사용)
                   api.py (FastAPI 앱 팩토리)
                   airflow_client.py (Airflow REST API 클라이언트)
                   reset.py / run_cycle.py / submit.py
config/          settings.py + competitions/<slug>.py (대회별 설정)
cycle/           사이클 로직:
                   run.py (단일 사이클 오케스트레이션)
                   super_cycle.py (슈퍼사이클 오케스트레이션)
                   stagnation.py (정체 감지)
                   action_optimizer.py (action_bandit Thompson sampling)
                   error_pitfalls.py (에러 시그니처 정규화 + top pitfall 조회)
                   materialize.py (AST 레벨 파이프라인 누적 병합)
evaluator/       결정적 k-fold CV (contract, harness, metrics)
memory/          retriever (pgvector 검색 + MMR), transfer (cross-competition, 부분 구현)
runtime/         격리 실행 (isolate.py → subprocess runner.py, 타임아웃 600s)
store/           db.py (psycopg2 풀), s3_code.py (MinIO), fingerprint.py, schema.sql
deploy/          Dockerfile, build.sh (registry.internal 푸시)
dashboard.py     Streamlit 모니터링
```

## 수퍼사이클 구조 (1회 실행 단위)

1. **Retrieve** (`bin/run_retrieve_task.py`): pgvector 코사인 검색으로 교훈 top-k + `action_bandit` Thompson sample로 attempt 3개에 서로 다른 `action_type` 배정 → `raw.super_cycle_context` upsert.
2. **Attempt × 3** 병렬 (`bin/run_attempt_task.py`): Strategize → Generate → Evaluate(k-fold CV) → Persist.
3. **Promote** (`bin/run_promote_task.py`): `gain_vs_best` 최대값 winner 선정 → `was_promoted` 플래그 → Reflect 호출 (winner: jump/regression/error 시만, loser: 전부).

## Stage 규칙

| stage | 1변경 규율 | 비고 |
|-------|-----------|------|
| `bootstrap` | 예외 (큰 변경 허용) | 새 대회 진입 첫 N=3~5회 |
| `reflexion` | **강제** — `prev_code` 주입 후 "한 군데만 수정" | 정상 루프 |
| `exploitation` | 예외 | best 안정화 |

## Coder 컨트랙트 (ADR-014)

Coder 출력은 반드시 `class Patch` 하나. action_type별 허용 훅:

| action_type | 허용 훅 |
|---|---|
| `feature_engineering` | `feature_transform` |
| `model_swap` | `build_model` |
| `preprocessing` | `preprocess` |
| `hyperparam_search` | `param_candidates` |
| `ensemble` | 모든 훅 (fallback, `compound` 별칭) |
| `bootstrap` | 모든 훅 (from-scratch) |

훅 이름: `preprocess` / `feature_transform` / `param_candidates` / `build_model` / `postprocess_predictions`.
`Patch.action_type` 속성이 배정된 action_type과 일치해야 함. IO·네트워크·eval 금지.
Polars API 사용 (pandas 스타일 혼용 금지). 컨트랙트 위반 코드는 `evaluator/contract.py`가 실행 전 차단.

## DB 스키마 핵심

- `raw.attempts` — 모든 시도 기록. `was_promoted`, `super_cycle_id`, `code_path`(MinIO) 포함.
- `raw.reflections` — 교훈 + `embedding vector(1024)`. 검색은 `store/db.py`가 아닌 `memory/retriever.py`.
- `raw.action_bandit` — `(scope, scope_key, action_type)` Beta-Bernoulli 밴딧.
- `raw.super_cycle_context` — retrieve → attempt 상태 전달용 임시 테이블.
- `raw.competitions` — 대회 메타 + fingerprint JSON.
- `raw.kaggle_submissions` — Kaggle 제출 추적 (submit_id, status, lb_score, polling 상태). `bin/api.py`의 `/api/submissions` 엔드포인트가 관리.

스키마 변경 시: `store/schema.sql` 수정 후 `uv run python bin/reset.py --hard`.

## 이미지 빌드 & 배포

```bash
# mac-server에서 빌드 (ARM64 이미지 → registry.internal 푸시)
ssh mac-server "cd ~/Projects/reflexion-rondo && git pull && bash deploy/build.sh"
```

이미지 2개: `reflexion-rondo/daemon` (오케스트레이션+LLM호출), `reflexion-rondo/eval` (격리 실행).

## 테스트

```bash
uv run pytest
```

## 주요 설계 결정 위치

- `docs/architecture.md` — 컴포넌트 배치, 데이터 흐름, 슈퍼사이클 상세
- `docs/decisions.md` — ADR-001 ~ ADR-019 (DuckDB→Postgres 전환, 3모델 분리 등)
- `docs/spec.md` — DB 스키마, enum 정의, 분석 뷰 SQL
- `docs/runbook.md` — 새 대회 등록 절차, 페이싱, 모니터링
