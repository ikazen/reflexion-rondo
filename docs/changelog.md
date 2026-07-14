# 변경 이력

## 이미지 빌드를 Airflow DAG로 이관, release.sh는 daemon 전용으로 축소 (2026-07-14)
- ADR-022: daemon+task 이미지 빌드+push를 airflow-stack의 `reflexion_rondo_deploy` DAG(ops 큐 docker.sock 재사용)로 이관. 여러 repo가 재사용할 공용 헬퍼(`dags/lib/image_deploy.py`, airflow-stack)라 신규 credential이 필요 없다(public repo clone 무인증, registry 무인증).
- task 이미지 태그의 source of truth가 git(DAG 파일)에서 Airflow Variable(`rondo_task_image_version`)로 이동 — DAG가 빌드 직후 즉시 bump, git push/GitDagBundle 지연 없음.
- `deploy/release.sh`(issue #17)가 build 단계를 잃고 registry 태그 존재 확인 + daemon 사전검증(issue #15 순서 유지) + compose.yml bump+재시작만 남김.
- `deploy/build.sh` 주석 정정(release.sh가 더 이상 정본 빌드 경로가 아님).

## release.sh 사전검증 순서 수정 (2026-07-14)
- issue #15: 스모크가 태그 bump(compose.yml+DAG, 양쪽 repo)/daemon 재시작보다 늦게 실행되던 순서 버그 수정. 이제 daemon+task 이미지를 일회성 컨테이너로 먼저 검증하고, 통과한 뒤에만 태그 bump+재시작이 진행된다.
- task 이미지는 이전엔 검증 대상이 아니었다(스모크는 daemon 컨테이너 exec뿐) — import 스모크로 신규 편입.
- `docs/runbook.md` §2 "이미지 배포" 절을 실제 흐름과 일치하도록 갱신, GitDagBundle 60초 반영 지연 사실 추가.
- `deploy/build.sh`의 존재하지 않는 `promote.sh` 참조 주석 정정.

## 문서 로직 서술 재싱크 — spec/architecture/runbook (2026-07-14)
- spec.md §3 지표 레지스트리: `qwk`(TBD→구현됨, metric_class ordinal→classification), `balanced_accuracy`(BON-273) 신규 행 추가.
- spec.md §4 label 규칙: `z` 기본값 1.0→실제 `LABEL_Z=2.0`(BON-194) 정정. jump 판정이 harness 절대-마진의 잠정값이며 `cycle/run.py`가 `is_significant_gain`(paired per-fold t-test, BON-247/267)으로 재확정한다는 사실 반영. 완벽점수·회귀 trivial-baseline 누수 가드 2종과 `is_noop_tie`(BON-239) 추가.
- spec.md §5 / architecture.md §5 / runbook.md §6: "네트워크 sandbox 미구현" 서술이 stale — 프로덕션에서 `os.unshare(CLONE_NEWNET)` egress 차단 + rlimit이 이미 구현됨(ADR-017)으로 정정, 파일시스템 sandbox만 미구현 유지. 코드생성 재생성 횟수도 1회→실제 2회로 정정.

## 문서·docstring 코드 싱크 정리 (2026-07-13)
- BON-240 반영: Coder 모델 문서 표기를 `qwen3.5:397b`(출력 토큰 과다) → `gpt-oss:120b`로 전 문서 정정(README, architecture, spec, setup, decisions ADR-016 amend).
- `lb_score`/제출 추적 상태 정정: `raw.kaggle_submissions` 폴링(`/api/submissions/*`)이 이미 lb_score를 attempts까지 기록하는데 "미구현"으로 서술되어 있던 architecture.md·runbook.md 정정. `submission_budget` 일일 상한 enforcement만 여전히 미구현.
- spec.md 스키마 누락 보강: `raw.kaggle_submissions`(§1.11 신설), `holdout_cv_gap_trend` 뷰, `raw.attempts.{holdout_score,confirm_seed_gains,fold_scores}`, `raw.pipelines.{pipeline_sha256,oof_preds}`, `raw.super_cycle_context` PK가 `queue_id`→`run_id`(BON-237)로 변경된 사실, `/api/health`+`/api/submissions/*` 엔드포인트 5종.
- README 프로젝트 구조 트리를 실제 파일과 일치시킴 — 존재하지 않는 `cycle/super_cycle.py` 참조 제거, 신규 파일 6개(`bin/blend.py`, `bin/export_results.py`, `bin/rebuild_best_pipeline.py`, `bin/seed_competition_data.py`, `cycle/promotion.py`, `store/train_data.py`) 추가.
- 죽은 코드 삭제: `main.py`(uv-init 스텁, 미참조), `bin/run_cycle.py`(Phase-0 PoC, `cycle/run.py`+`run_cycle_task.py`로 대체됨).
- `api.md`(관측 API 설계, ~30 엔드포인트 대부분 미구현) — 코드가 아니라 gitignore된 미추적 로컬 스크래치 파일이었음이 드러나 GitHub Issue #11로 이관 후 삭제.

## 학습 신호 회복 — jump 라벨 붕괴 수정 + 코드생성 정적 가드 (2026-07-05)
- ADR-012 amend(BON-267): label의 jump 판정을 harness 절대-마진에서 promotion과 동일한 paired 유의성 검정(`is_significant_gain`)으로 통일. 전체 7447건 attempt 중 jump 0건이던 근본원인 수정 — bandit 보상·stagnation 감지·reflection 게이트가 전부 이 label에 의존해 함께 정상화됨.
- BON-268: `evaluator/contract.py`의 `validate_patch`에 정적 검사 2종 추가 — pandas-only API(`.groupby`/`.map_dict`/`.take`/`.apply`/`.iterrows`/`.applymap`/`.get_dummies`) 금지, candidate patch 자신의 undefined-name 검사(실행 격리 모델과 일치하는 범위로 검증). `agents/coder.py` contract 프롬프트에도 동일 금지 목록 반영.
- BON-269: reflection이 실패에서만 학습한다는 문제 제기는 재검토 결과 BON-267로 이미 구조적으로 해결됨을 확인(게이트 자체엔 버그 없었음, jump가 0건이라 죽어 있던 코드였을 뿐) — 코드 변경 없이 종료.

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
