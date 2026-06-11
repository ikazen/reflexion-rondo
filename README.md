# reflexion-rondo

가중치 학습(fine-tuning) 없이 `경험 → 성찰(Reflection) → 지식화(RAG) → 재적용` 루프로,
시간이 지날수록 Kaggle 정형 대회 성능이 스스로 향상되는 자가 개선형 시스템.

핵심 가설: **약한 모델 + 누적된 구체적 교훈 ≥ 강한 모델 단독.**

## 시스템 목표

1. **Intra-competition gain**: 한 대회 안에서 attempt가 누적될수록 CV가 개선된다.
2. **Inter-competition transfer**: 대회 N의 경험이 대회 N+1의 cold-start 효율을 끌어올린다.

성공 조건: 객관적·자동 검증 가능한 피드백 신호(CV + LB)가 있는 도메인
→ Kaggle CSV 제출형 정형 대회 (Playground Series).

## 진입점

- `bin/run_daemon.py` — 운영 daemon. `raw.cycle_queue`를 폴링하고, `AIRFLOW_URL`이 있으면 Airflow super-cycle DAG를 트리거한다. `AIRFLOW_URL`이 없으면 로컬 smoke/test용 단일 attempt만 실행한다.
- `bin/run_reflexion.py` — 로컬/수동 Reflexion 사이클 러너 (`cycle/run.py` 호출: retrieve → strategize → generate → evaluate → reflect → persist + 생성 코드를 `runs/code/`에 저장).
- `bin/start_competition.py` — 대회 등록 (fingerprint 계산 → `raw.competitions` insert). 유사 대회 검색·cold-start lessons/seed pipeline 정보를 `runs/cold_start/`에 저장한다.
- `bin/run_cycle.py` — Phase 0 PoC (LLM 없는 LightGBM 베이스라인, 참고용).
- `bin/submit.py` — Kaggle 제출.  `dashboard.py` — Streamlit 모니터링.

## 레포 구조

```text
reflexion-rondo/
  config/settings.py      # 모델/엔드포인트/자격증명 로딩
  agents/                 # 평면 모듈 (서브디렉토리 아님)
    strategist.py         # 프롬프트 + StrategyDecision (action_type enum)
    coder.py              # class Patch 생성 (action_type별 hook)
    reflector.py          # 성찰 → 교훈 + generality enum (참고 판정 포함)
  cycle/run.py            # Reflexion 1 사이클 오케스트레이션 (7단계)
  evaluator/
    harness.py            # 결정적 k-fold CV + label 계산
    metrics.py            # 지표 + metric_sign
    contract.py           # 생성 코드 검증 + smoke test
  memory/retriever.py     # Postgres/pgvector 검색 + embed() + 메타필터 + MMR
  memory/transfer.py      # fingerprint 거리 · 유사 대회 검색 · cold-start lessons/seed 추출
  store/
    schema.sql            # competitions, attempts, reflections, submission_budget + 분석 뷰
    db.py                 # connect / ensure_competition / insert_attempt
    fingerprint.py        # 결정적 메타피처(14개) 계산기
  bin/
    run_daemon.py         # queue daemon + FastAPI + Airflow trigger
    run_reflexion.py      # 로컬/수동 Reflexion 사이클 러너
    run_retrieve_task.py  # Airflow super-cycle retrieve task
    run_attempt_task.py   # Airflow super-cycle attempt task
    run_promote_task.py   # Airflow super-cycle promote task
    start_competition.py  # 대회 등록 (fingerprint → competitions insert)
    run_cycle.py          # Phase 0 PoC (LLM 없는 베이스라인, 참고용)
    submit.py             # Kaggle 제출
  dashboard.py            # Streamlit 모니터링
  runs/                   # 생성 코드(runs/code/) · cold-start JSON · 제출 CSV  (gitignore)
  docs/                   # 아래 문서

분석 뷰(dbt 아님, schema.sql 내 SQL view): score_progression, stg_attempts,
stg_attempts_reflexion_only, reflection_impact, action_bandit_posterior,
cold_start_progression

계획(미구현):
  external ideas 채널      # raw.external_ideas / Strategist 노출 / 사후 bandit — ADR-019 설계만
  제출 예산 자동 게이트    # submission_budget 스키마는 있으나 submit.py enforcement는 미구현
  강화된 생성 코드 sandbox # 현재 subprocess + env allowlist, 네트워크 격리 미구현
```

## 문서

- `docs/architecture.md` — 컴포넌트·데이터 흐름·transfer 메커니즘
- `docs/decisions.md` — 기술 결정 이력 (ADR)
- `docs/spec.md` — DB 스키마·분석 뷰·LLM API·코드 컨트랙트
- `docs/setup.md` — 초기 셋업
- `docs/runbook.md` — 운영 절차·관측·디버깅
- `docs/strategy.md` — 정형 대회 일반 전략 노트 (Strategist/Reflector 컨텍스트)
- `docs/changelog.md` — 변경 이력
