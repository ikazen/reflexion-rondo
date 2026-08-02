# 변경 이력

## 미배포 (main, 다음 task 이미지 빌드에 포함 예정)
- #120: `bin/quarantine_leaks.py`가 코드 exec/데이터 로드 실패(판정 불가)를 누수 확정과 동일하게 처리해 격리 대상으로 잘못 집계하던 문제 수정 — `--dry-run` 프로덕션 실측 중 s4e1/s5e3 정상 파이프라인 27개가 로컬 데이터 캐시 부재만으로 부당하게 격리될 뻔한 사고를 계기로 발견.

## v1.4.14 — daemon 크래시루프 수정: naive/aware datetime (2026-08-02)
- #118: `bin/run_daemon.py:_submission_refresh_due`가 DB에서 읽은 naive `timestamp`(`raw.kaggle_submissions.submitted_at`/`checked_at`)를 `datetime.now(timezone.utc)`(aware)와 직접 빼서 `TypeError`로 daemon이 크래시루프에 빠짐 — v1.4.13 배포 직후 프로덕션에서 실제 발생. 로컬 mock 테스트는 양쪽을 전부 aware로 구성해 이 조합을 못 잡았다.
- 세 값(`now`/`submitted_at`/`checked_at`) 모두 naive로 정규화 후 뺄셈하도록 수정, DB round-trip을 실제로 거치는 회귀 테스트 2종 추가.

## v1.4.13 — 학습 정체 근본 원인 수정 (측정 정합성 + 승격 래칫 + LB 되먹임 + 컴퓨트 회수) (2026-08-02)
"reflexion 상한선이 아닌데 성능 향상이 없다"는 진단에서 나온 4갈래 구조적 결함을 한 번에 배포. Milestone 5개(M1~M5), 이슈 12개.

**M1 측정 정합성**
- #97: `preprocess` 훅이 valid split의 타깃 컬럼을 직접 읽는 누수를 fold0 동등성 검사(마스킹 전/후 결과 비교)로 실측 차단 + AST 정적 가드. `_REGRESSION_IMPLAUSIBLE_BASELINE_RATIO` 100→10 하향.
- #98: audit holdout을 `bin/submit.py`와 동일한 dummy target 치환으로 재현해 실제 추론 조건과 일치시키고, `holdout_regressed`를 승격 조건에 AND 결합(기록용→차단 게이트).
- #99: `raw.pipelines.invalid_reason` 컬럼 + `bin/quarantine_leaks.py` 신설 — 확정 후 누수로 밝혀진 파이프라인을 격리(삭제 아님). 모든 baseline 조회 경로에 `invalid_reason IS NULL` 필터 적용.

**M2 승격 래칫 복구**
- #100: bootstrap 종료 시 최고 attempt를 `confirm_and_measure`로 검증해 확정 baseline 자동 확립(`cycle/run.py:establish_bootstrap_baseline`).
- #101: 기존에 확정 파이프라인이 없던 대회를 위한 소급 스크립트 `bin/establish_baseline.py`(top-k 순회, `--dry-run` 지원).
- #102: "확정 파이프라인 없으면 전체 attempt의 max(cv)로 폴백"하던 phantom-max 분기 제거 — N이 늘수록 문턱이 같이 올라가는 자기강화 데드락의 근본 원인.

**M3 LB 되먹임 연결**
- #103: daemon 유휴 틱마다 `status IN ('submitted','pending')` 제출을 지수 백오프로 재폴링 — 기존엔 업로드 직후 1회뿐이라 pending이면 수동 refresh 전까지 영구 방치됐다.
- #104: `cv_lb_calibration` 뷰 + 발산 트립와이어 — CV는 개선인데 LB가 악화된 제출이 나오면 원천 pipeline 격리 + 해당 대회 auto-submit 중단(`auto_submit_paused_reason`, 자동 해제 없음).

**M4 컴퓨트 회수**
- #84: `comp.MAX_TRAIN_ROWS` opt-in + 층화 샘플링(`store/train_data.py`) — s4e7(11.5M행)이 100-cycle 큐 전량 OOM되던 문제, 1.5M으로 상한 적용.
- #74: `ensemble_spec` 선언형 앙상블 프리미티브 — 자유형 wrapper 클래스 크래시(70%→55%에서 정체)의 구조적 원인(exec된 클래스 몸체 내부는 harness가 볼 수 없음)을 해소. Patch는 멤버·결합방식만 선언, harness가 생성·적합·결합 전담.

**M5 탐색 능력**
- #75: `raw.blend_weights` 테이블 신설 + `bin/blend.py:compute_and_store_blend`를 양쪽 승격 경로에 배선 — 계산·저장까지만(`bin/submit.py`는 의도적으로 미연결, 범위 밖).
- #76: `bin/archive_lessons.py`를 daemon 24시간 스윕으로 배선 — 이전엔 CLI 수동 실행뿐이라 사실상 죽은 레버였다.

## v1.4.12 — submit build_model 안전망 (2026-07-25)
- #94: `bin/submit.py`의 `build_model()` 생성자 stale kwarg 문제에 평가 경로(#79)와 동일한 런타임 안전망 적용.

## v1.4.11 — MinIO 텍스트 다운로드 mojibake 수정 (2026-07-25)
- #92: charset 없는 `text/plain` 응답을 latin-1로 잘못 디코드하던 문제 — UTF-8 명시 디코드로 수정.

## v1.4.10 — materialized_code 스냅샷 + 백필 (2026-07-24)
- #89: 승격 시점 병합본 스냅샷을 `raw.pipelines.materialized_code`에 저장. attempt 제출 base를 replay 대신 이 스냅샷으로 로드(replay 폴백은 sha 불일치 시 중단). 기존 승격 행을 위한 백필 스크립트 신설.

## v1.4.9 — auto-submit 재제출 유의성 검정 (2026-07-24)
- #87/#88: fold noise 수준의 변화로도 재제출되던 auto-submit 게이트에 유의성 검정 추가.

## v1.4.8 — ensemble materialize 유실 수정 (2026-07-24)
- #83: `materialize`가 Patch 안 중첩 클래스를 유실해 ensemble 승격이 merge-verify 단계에서 크래시하던 문제 수정.

## v1.4.7 — 콜드스타트 데드락 + confirm 실패 원인 소실 수정 (2026-07-23)
- #73: 확정 승격 콜드스타트 데드락과 confirm 실패 시 원인이 로그에서 사라지던 문제 수정.

## v1.4.6 — attempt_only 제출 누적 base 유실 수정 (2026-07-23)
- #80: `attempt_only` 제출이 누적 base pipeline을 버리고 vanilla 모델로 떨어지던 문제 수정.

## v1.4.3~5 — ensemble action_type 크래시 완화 3라운드 (2026-07-22~23)
- #74/#77: Coder 컨트랙트의 잘못된 예시·조언이 ensemble action_type 70% 크래시를 유발하던 문제 1차 수정.
- #74 후속(#78): 정석 예시의 `sorted(unique())`가 null 섞이면 크래시 — `drop_nulls()` 추가.
- #74 후속(#79): `build_model()` 생성자 stale kwarg 런타임 안전망 추가 — 프롬프트 경고만으론 재발을 못 막았다. (이 3라운드로도 크래시율은 70%→55%까지만 떨어졌고, 근본 해결은 v1.4.13의 #74 선언형 프리미티브로 이관.)

## v1.4.2 — submit 생성자 early-stopping 파라미터 크래시 수정 (2026-07-22)
- #72: 생성자 early-stopping 파라미터가 submit full-train fit을 크래시시키던 문제 수정.

## v1.4.1 — reflection_impact DROP VIEW CASCADE (2026-07-22)
- #69: `lesson_dead`가 `reflection_impact`에 의존해 CASCADE 없이는 재적용이 실패하던 문제 수정.

## v1.4.0 — 관측 API 24개 + 파생 뷰 6개 + 대회 온보딩 14개 (2026-07-22)
- #66/#67: 관측 계측 P1/P2(`retrieved_ids`/`error_signature` 영속화) + 파생 뷰 6종(funnel/dead/duplicates/calibration/error_recurrence/transfer) + 엔드포인트 24개(score/timeline, bandit/posteriors, lessons, errors, transfer 등).
- #39: 대회 온보딩 14개 — binary AUC 6개, accuracy/balanced_accuracy 3개, regression 5개.

## v1.3.1 — auto-submit 캐시 트리거 수정 (2026-07-22)
- #62/#63: auto-submit 제출 CSV 캐시가 실제로 채워지지 않던 트리거·키 문제 수정.
- #61: 코드베이스 전체 이슈 번호 태그 정리 + 문서 최신화.

## v1.3.0 — gain_vs_best_relative: metric 스케일 정규화 (2026-07-22)
- `reflection_impact` 전역 z-score가 metric 스케일 혼합(rmse 원시 단위 vs auc 0~1)으로 오염돼 있던 문제(mean=-4.22, std=139.19 실측) — DB wipe로는 해결 안 되는 코드 자체의 스케일 정규화 부재가 근본 원인으로 확인됨.
- `gain_vs_best_relative` 컬럼 신설(regression_error는 `gain_vs_best / baseline_cv` 상대값, 나머지 metric_class는 `gain_vs_best` 패스스루) — `evaluator/harness.py` → `runtime/isolate.py`/`runtime/runner.py`(subprocess JSON 경계) → `cycle/run.py` → `store/schema.sql`까지 전체 파이프라인에 배관.
- `reflection_impact` 뷰가 이 컬럼만 집계하도록 재정의 — `gain_vs_best_relative IS NULL`인 legacy row는 집계에서 제외(raw `gain_vs_best`로 폴백하지 않음).
- 프로덕션 실측으로 정규화 검증: raw `gain_vs_best=-110.92` → `gain_vs_best_relative=-0.0014`.

## v1.2.32 — Kaggle 자동 제출 gap 진단 및 수정 (2026-07-22)
- s5e7/s6e6 자동 제출 누락 재진단: (a) `bin/submit.py` dummy target 생성이 문자열 타깃 대회에서 크래시하던 버그, (b) `raw.kaggle_submissions.status='submitted'`가 폴링 데드라인을 지나도 안 풀려 재제출이 영구 스킵되던 버그.
- 두 경로 모두 수정 후 실제 Kaggle 제출로 종단 검증 — s5e7 LB 0.973279, s6e6 LB 0.96408로 `complete` 상태 도달 확인.

## v1.2.31 — subprocess 고아 프로세스 kill + degenerate 회귀 gain 클립 + s6e7 코드생성 harness 버그 (2026-07-22)
- #37: `bin/api.py`의 `subprocess.run` 타임아웃 처리가 `uv run`이 spawn한 손자 python 프로세스를 못 죽여 고아로 남던 문제(타임아웃 기록 후에도 CPU 85%+ 점유 실측) — `_run_in_pgroup` 헬퍼로 교체, `start_new_session=True` + 타임아웃 시 `os.killpg`로 프로세스 그룹 전체 종료.
- #43: rmse degenerate 예측(모델이 완전히 빗나감)이 만드는 극단적 `gain_vs_best`(s6e1 실측 -105448)가 `reflection_impact` 전역 z-score(BON-195)를 오염시키던 문제 — `evaluator/harness.py`에 대칭 가드 추가(issue #4의 "100배 좋으면 raise"에 대칭으로 "100배 나쁘면 gain_vs_best 하한 클립", label 판정은 영향 없음).
- #42: s6e7(multiclass) 에러율 65% 재진단 결과 다수가 Coder 코드 문제가 아니라 harness 자체 버그(`_encode_residual_categoricals`가 null 섞인 문자열 컬럼 정렬 시 크래시, ADR-014 amend)로 확인 — null 제외 정렬로 수정. 추가로 multiclass 라벨 왕복(round-trip) 컨트랙트 규칙과 action_type별 hook 동적 강조를 `agents/coder.py`에 반영.

## v1.2.28~30 — 대회 온보딩 배치 3건 + 운영 버그 수정 (2026-07-17~19)
- #24 (2026-07-17): 신규 대회 3개 추가 — s6e7(multiclass), s6e1(regression), s5e9(regression).
- #25/#26 (2026-07-17): `release.sh`의 registry 태그 존재 확인이 OCI 매니페스트 타입 미인식 + repo path에 태그가 남는 버그로 늘 실패하던 문제 수정.
- #27/#28 (2026-07-17): attempt eval `RLIMIT_AS` 기본값을 1.5GiB→6GiB로 복원(#20이 낮췄던 값 — mac-server Colima VM을 8→16GiB로 증설해 근본 해결, 신규 대회 bootstrap 전체가 SIGKILL(rc=-9)되던 원인).
- BON-275 (2026-07-19): 제출(`bin.submit`) 타임아웃 600s→1200s 상향 — s5e5(75만 행) 5-seed bagging이 기존 값을 넘겨 매번 타임아웃.
- #29 (2026-07-18): README에 "지원 대회 조건" 섹션 추가.
- #31 (2026-07-18): promote 시점 submission CSV 캐싱 — 매일 auto-submit이 daemon(ops-vm, 2 OCPU)에서 매번 fit하며 CPU를 포화시키던 경로 대체.
- #33 (2026-07-19): `release.sh` post-restart heartbeat 체크가 노출 안 된 `localhost:8000` 대신 `rondo-api.internal` 사용하도록 수정.
- #35 (2026-07-19): `attempt_only` 재구성이 훅 메서드만 옮겨 붙이는 lossy transplant라 Patch의 훅 밖 클래스 속성이 소실되며 s6e7 auto-submit이 크래시하던 문제 수정.
- #40 (2026-07-19): 대회 4건 온보딩 — s4e2(multiclass)/s5e7(binary)/s4e9(regression)/s6e2(binary), accuracy 메트릭 최초 실사용.

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
