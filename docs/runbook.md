# 운영 절차·관측·디버깅

## 0. 상태 초기화 (reset)

개발 중 모델 교체·스키마 변경 등으로 기존 이력을 버릴 때 사용한다.

```bash
# 전체 초기화 (Postgres 데이터 삭제, 스키마 유지)
uv run python bin/reset.py

# 스키마까지 drop 후 재생성 (스키마 변경 시)
uv run python bin/reset.py --hard

# 특정 대회만 초기화 (다른 대회 이력 보존)
uv run python bin/reset.py --competition playground-series-s4e1

# 확인 프롬프트 생략
uv run python bin/reset.py --yes
```

초기화 대상:
- Postgres `raw` 스키마 테이블 — attempts / reflections / competitions 등
- `runs/code/{competition_id}/` — 생성된 Python 코드 파일
- `runs/submission_*.csv` — 제출 파일 (전체 초기화 시)

## 1. 새 대회 등록 절차

### 1-1. 데이터 다운로드
```bash
# Kaggle 규칙 동의 먼저 (403 뜨면 브라우저에서 동의)
! kaggle competitions download -c <competition-id> -p data/<competition-id>
cd data/<competition-id> && unzip -q *.zip
```

### 1-2. config/competitions/<slug>.py 작성
`config/competitions/s5e3.py` 참고. 필수 항목:
- `COMPETITION_ID`, `NAME`, `TARGET`, `METRIC`, `TASK_TYPE`, `METRIC_SIGN`
- `TASK_TYPE` 값은 `binary` / `multiclass` / `regression` 중 하나
- `IS_CLASSIFICATION`, `DROP_COLS` (id 계열 컬럼)
- `DATA_DIR`, `S3_DATA_PATH`
- `EDA_CARD` — feature별 dtype 명시 필수 (pl.String 컬럼 인코딩 방법 포함)

### 1-3. cold-start 등록
```bash
uv run python -m bin.start_competition \
    --id <competition-id> \
    --name "<대회명>" \
    --task binary --metric auc --target <타깃컬럼>
# 출력: 유사 대회 목록, 교훈 수, 시드 파이프라인 수 확인
```

### 1-4. 큐에 등록하고 daemon 실행
```bash
# daemon이 떠있으면 API로 큐잉
curl -X POST http://localhost:8000/api/queue \
  -H 'Content-Type: application/json' \
  -d '{"competition": "<slug>", "stage": "bootstrap", "n_cycles": 5}'

# daemon 없이 직접 실행
uv run python -m bin.run_reflexion \
    --competition <slug> --stage bootstrap --cycles 5 --cold-start
```

### 1-5. reflexion 루프
```bash
curl -X POST http://localhost:8000/api/queue \
  -d '{"competition": "<slug>", "stage": "reflexion", "n_cycles": 30}'
```

### 1-6. 교훈 위생 (30사이클마다)
```bash
uv run python -m bin.archive_lessons
```

## 2. 오케스트레이션

daemon(`bin/run_daemon.py`)이 `raw.cycle_queue`를 10초 간격으로 폴링해서 순차 실행한다.

```bash
# daemon 시작
uv run python -m bin.run_daemon

# 상태 확인
curl http://localhost:8000/api/heartbeat

# 큐 조회
curl http://localhost:8000/api/queue

# 큐 취소
curl -X PATCH http://localhost:8000/api/queue/<queue_id> \
  -d '{"status": "cancelled"}'
```

실행 모드:
- **airflow 모드 (운영)**: `AIRFLOW_URL` 환경변수가 있으면 Airflow DAG `reflexion_rondo_cycle` 트리거. 1 DAG run = 1 슈퍼사이클 (retrieve → attempt_0/1/2 병렬 → promote). retrieve/promote는 default 큐, attempt는 big 큐.
- **direct 모드 (로컬 테스트)**: `AIRFLOW_URL` 없으면 daemon 프로세스 안에서 단일 `run_cycle()` attempt만 실행한다. forced action 배정, 3-way 병렬 attempt, promote/loser reflection은 실행하지 않는다.

**이미지 업데이트 절차:**
```bash
# mac-server에서 빌드 및 push
ssh mac-server "cd ~/Projects/reflexion-rondo && git pull && bash deploy/build.sh"
# airflow-stack DAG의 IMAGE SHA 업데이트 후 push
```

## 3. 페이싱 (Ollama Cloud)

환경변수로 제어. 미설정 시 비활성.

| 환경변수 | 설명 |
|---|---|
| `OLLAMA_CLOUD_SESSION_HOURS` | 세션 윈도우 길이 (기본 5.0h) |
| `OLLAMA_CLOUD_SESSION_CYCLES` | 세션당 최대 사이클 수 (0=비활성) |
| `OLLAMA_CLOUD_WEEKLY_CYCLES` | 주간 최대 사이클 수 (0=비활성) |

한도 초과 시 다음 윈도우까지 sleep 후 재개 (스킵 아님). daemon 재시작 시 DB의 실제 attempt 수로 카운터 복원.

## 4. 제출 예산 게이트

- `submission_budget` 테이블은 스키마에 존재한다.
- 현재 `bin/submit.py`는 CSV 생성과 Kaggle 제출 호출을 수행하지만, 일일 제출 상한 자동 enforcement와 `lb_score` 업데이트는 아직 구현하지 않았다.
- 운영 시에는 best 후보만 수동 제출하고, LB score는 별도 기록 절차가 추가될 때까지 수동 관리한다.

## 5. 동시성

Postgres가 concurrent read를 처리한다. daemon은 단일 프로세스로 순차 실행이므로 write 충돌이 없다. 여러 daemon을 띄우는 구성은 현재 미지원.

## 6. 생성 코드 격리 실행

`runtime/isolate.py`가 tmpdir를 생성하고 `runner.py`를 subprocess로 실행한다.

- 타임아웃: 600초 (기본값, `DEFAULT_TIMEOUT`)
- 타임아웃·에러 → `error_trace` 기록 → Reflect 단계가 실패에서 교훈 추출
- 격리 수준: subprocess 분리 + env allowlist 필터링 (BON-104). 네트워크 격리 미구현.
- `OMP_NUM_THREADS=2` 등 스레드 제한은 Dockerfile ENV + subprocess allowlist 양쪽에 설정.

## 7. 모니터링

**Streamlit 대시보드 (로컬)**
```bash
uv run streamlit run dashboard.py
# → http://localhost:8501
```
CV score 진행 곡선, label/action_type 분포, reflection_impact 상위 교훈, 최근 attempt 테이블 제공.

**Daemon API**
```bash
curl http://localhost:8000/api/heartbeat   # 현재 상태
curl http://localhost:8000/api/attempts    # 최근 attempt 목록
curl http://localhost:8000/api/lessons     # 교훈 목록
```

**Prometheus (BON-90, 배포 후)**
- `GET /metrics` — `rondo_cycles_total`, `rondo_daemon_last_cycle_timestamp_seconds`, `rondo_queue_pending`
- Grafana alert: `time() - rondo_daemon_last_cycle_timestamp_seconds > 7200` → Discord

## 8. 관측·디버깅

- `reflections.embedded_text`/`full_lesson`을 함께 저장 → 검색 결과를 사람이 SQL로 바로 읽음.
- `raw.attempts.reflection_ids` + `retrieval_scores`로 검색 단계까지 역추적.
- 실패 attempt도 기록되어 분석 대상.
- transfer 점검: `cold_start_progression`의 `warm_start_ratio` 추세. 우상향 아니면 fingerprint 가중치/generality 라벨링/검색 메타필터 점검.
- 노이즈 점검: `cv_fold_var`가 큰 attempt의 label은 `neutral`로 빠지는지 확인.

## 9. 교훈 위생 (메타 루프)

- `archived=true` 또는 `reflection_impact.avg_gain ≤ 0` 교훈은 검색 제외/가중치 하향.
- `L1_local`은 transfer에서 자동 제외.
- bootstrap 단계 attempt는 `reflection_impact` 집계에서 제외 (1변경 규율 위배 데이터).
