# 운영 절차·관측·디버깅

## 0. 상태 초기화 (reset)

개발 중 모델 교체·스키마 변경 등으로 기존 이력을 버릴 때 사용한다.

```bash
# 전체 초기화 (DB + 생성 코드 + 제출 CSV)
uv run python bin/reset.py

# 특정 대회만 초기화 (다른 대회 이력 보존)
uv run python bin/reset.py --competition playground-series-s4e1

# 확인 프롬프트 생략 (-y)
uv run python bin/reset.py -c playground-series-s4e1 -y
```

초기화 대상:
- `runs/reflexion.duckdb` — 전체 attempts / reflections / competitions 기록
- `runs/code/{competition_id}/` — 생성된 Python 코드 파일
- `runs/submission_*.csv` — 제출 파일 (전체 초기화 시)

대회별 초기화는 DB 행만 삭제하고 다른 대회 데이터는 건드리지 않는다.

## 1. 오케스트레이션

- `bin/run_cycle.py`: 1 사이클 (retrieve → strategize → generate → evaluate → submit? → reflect → persist).
- `bin/start_competition.py`: 새 대회 cold-start (fingerprint → warm-start 시드 → bootstrap 큐잉).
- cron이 `run_cycle.py`를 주기 호출.

승격 트리거(언제 Prefect 도입): 워커 ≥ 3, 또는 단계별 재시도/타임아웃 로직이 복잡해질 때.
그 전에는 Python + cron이 가장 적은 디버깅 비용.

## 2. 페이싱 (Ollama Cloud)

- 5시간 세션 + 주간 한도를 cron schedule에 반영. 상한 도달 시 그날 스킵.
- 동시 실행: 요금제별 상한(Pro 3)과 로컬 워커 수(시작 ≤ 2)를 균형 맞춤.

## 3. 제출 예산 게이트

- `submission_budget` 테이블에 일별 카운트.
- `Submit` 단계가 SELECT-then-UPDATE로 일일 상한(보통 5) 초과 방지.
- best 후보만 LB로. CV-LB 상관·shake를 함께 기록.

## 4. 동시성

- DuckDB는 단일 writer. cron 중첩 실행 시 충돌하므로 파일 락(또는 DuckDB row lock)으로 직렬화.
- 워커 ≤ 2 규모에서는 파일 락으로 충분.

## 5. 생성 코드 격리 실행

- `runtime/`가 컨테이너/nsjail에서 `feature_fn`/`model_fn`을 실행 (ADR-013).
- 시간/메모리 상한, 네트워크 차단, FS 화이트리스트.
- 타임아웃·OOM·격리 위반 → 강제 종료 후 `error_trace` 기록 → Reflect 단계가 실패에서 교훈 추출.

## 6. 모니터링 대시보드

```bash
uv run streamlit run dashboard.py
# → http://localhost:8501
```

- CV score 진행 곡선, label/action_type 분포, reflection_impact 상위 교훈, 최근 attempt 테이블 제공
- DB 마이그레이션을 자동 적용하므로 스키마 변경 후에도 별도 작업 불필요
- 사이클 실행 중에도 새로고침으로 실시간 확인 가능

## 7. 관측·디버깅

- `reflections.embedded_text`/`full_lesson`을 함께 저장 → 검색 결과(`spec.md §1.6` 쿼리)를 사람이 SQL로 바로 읽음.
- 기록·검색·분석이 한 DuckDB라 SQL 한 곳에서 디버깅.
- `raw.attempts.reflection_ids` + `retrieval_scores`로 검색 단계까지 역추적 ("전략가가 왜 엉뚱한 조언을 받았나").
- 실패 attempt도 기록되어 분석 대상.
- transfer 점검: `cold_start_progression`의 `warm_start_ratio` 추세. 우상향 아니면 fingerprint 가중치/generality 라벨링/검색 메타필터 점검.
- 노이즈 점검: `cv_fold_var`가 큰 attempt의 label은 `neutral`로 빠지는지 확인 (spec §4).

## 8. 교훈 위생 (메타 루프, Phase 4)

- `archived=true` 또는 `reflection_impact.avg_gain ≤ 0` 교훈은 검색 제외/가중치 하향.
- `L1_local`은 transfer에서 자동 제외.
- bootstrap 단계 attempt는 `reflection_impact` 집계에서 제외 (1변경 규율 위배 데이터).
