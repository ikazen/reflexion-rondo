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

- `bin/start_competition.py` — 새 대회 cold-start (fingerprint → warm-start 시드 → bootstrap)
- `bin/run_cycle.py` — Reflexion 1 사이클 (retrieve → strategize → generate → evaluate → submit? → reflect → persist)
- cron이 `run_cycle.py`를 주기 호출.

## 레포 구조

```text
reflexion-rondo/
  config/                 # 대회 설정, 모델/엔드포인트, 자격증명 로딩
  agents/
    strategist/           # 프롬프트 + 출력 스키마 (action_type enum)
    coder/                # 출력 스키마 (feature_fn / model_fn 컨트랙트)
    reflector/            # 출력 스키마 (generality enum, 참고 판정)
    client.py             # Ollama Cloud / 로컬 클라이언트 래퍼
  evaluator/              # 결정적 k-fold + Optuna 캡슐화 + 지표 + label 계산
  runtime/                # 생성 코드 격리 실행 (컨테이너/nsjail)
  memory/
    embed.py              # 로컬 임베딩
    vector_store.py       # Chroma add/query + 재순위
    transfer.py           # fingerprint 거리, 유사 대회 검색, warm-start 시드
  store/
    schema.sql            # competitions, attempts, reflections, pipelines
    ingest.py             # dual-write
    fingerprint.py        # 결정적 메타피처 계산기
  dbt/
    models/staging/       # stg_attempts (+ reflexion_only view)
    models/marts/         # score_progression, reflection_impact, cold_start_progression
  bin/
    run_cycle.py          # 1 사이클
    start_competition.py  # cold-start (fingerprint → seed → bootstrap)
  pipelines/              # 후보 파이프라인 산출물
  runs/                   # 로그, 아티팩트
  docs/                   # 아래 문서
```

## 문서

- `docs/architecture.md` — 컴포넌트·데이터 흐름·transfer 메커니즘
- `docs/decisions.md` — 기술 결정 이력 (ADR)
- `docs/spec.md` — DB 스키마·dbt 모델·LLM API·코드 컨트랙트
- `docs/tasks.md` — 개발 로드맵·진행 상태
- `docs/setup.md` — 초기 셋업
- `docs/runbook.md` — 운영 절차·관측·디버깅
- `docs/strategy.md` — 정형 대회 일반 전략 노트 (Strategist/Reflector 컨텍스트)
- `docs/changelog.md` — 변경 이력
