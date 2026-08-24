# 변경 이력

## v1.5.14 — 진짜 stacking(ensemble_spec method="stack"), bin/blend.py 폐기 (2026-08-25)
- #231(Milestone v1.6.0, ADR-036): `ensemble_spec`에 `"method": "stack"` + `"meta"` 모델 키 추가
  (`evaluator/harness.py:_fit_predict_stack`/`_oof_member_predictions`). 멤버는 outer CV fold의
  train을 inner K-fold(기본 5)로 나눠 만든 out-of-fold(OOF) 예측으로 meta를 학습하고(누수 방지 —
  meta가 멤버의 train-fit 성능을 절대 못 봄), 실제 검증/제출 예측 시엔 멤버를 outer train 전체로
  재적합해서 쓴다(표준 stacking 관례). meta는 `ctx.is_classification`과 무관하게 항상 회귀 변형으로
  생성 — 멤버 출력이 연속값(확률/예측값) 공간이라 RidgeClassifier 등 분류 변형이 predict_proba
  자체가 없거나 불필요한 이산화 손실을 낳는 함정(#229/#239와 동일 클래스)을 피한다. discrete label
  채점(`metric_class == "classification"`)은 지원 안 함 — `majority_vote`를 쓸 것.
- `bin/blend.py`(파이프라인 밖 사후 Ridge blend) + `raw.blend_weights` 테이블 + dashboard 패널 +
  `bin/backfill_oof_preds.py` 전부 삭제 — `bin/submit.py`가 그 값을 전혀 소비하지 않던 죽은 코드였음을
  코드로 재확인(`docs/spec.md` §1.13 기존 인지 사항). `oof_preds` 컬럼 자체(수집 로직)는 향후 분석
  재료로 유지, 현재 소비처는 없음.
- `agents/coder.py` 컨트랙트 프롬프트에 `method="stack"` 섹션 추가(ensemble_spec 기존 섹션과 동일 구조).
- **merge 전 백테스트(완료 기준)**: deep tier confirmed pipeline 2개(s4e10/s6e8)에 실제 원본
  preprocess/feature_transform을 보존한 채 동일 멤버 조합(lgbm+random_forest)을
  weighted_average vs stack으로 비교 — 둘 다 stack이 실측 개선(s4e10 auc 0.955991→0.960543,
  s6e8 auc 0.946023→0.949325, 80k 다운샘플).
- 회귀 테스트 9개 추가(빈 members/meta 누락/discrete metric_class 거부, 회귀·binary_proba
  end-to-end, yva=None 제출 경로, OOF 누수 검증, evaluate_pipeline 전체 경로).
- 부수 발견: s5e4 confirmed pipeline이 weighted_average 백테스트 중 "Input y contains NaN"으로
  크래시(#245로 이슈화, 이 PR과 무관 — 프로덕션에서도 이미 반복 관측되던 별개 문제).

## v1.5.13 — Optuna 튜닝 레인, 900s attempt 예산 밖 별도 DAG (2026-08-24)
- #230(Milestone v1.6.0, ADR-035): 신규 `evaluator/tuner.py` + `evaluator/search_spaces.py` —
  confirmed pipeline(raw.pipelines)의 `model_spec`/`ensemble_spec`(#229) 멤버 params를 Optuna로
  수십~수백 trial 탐색한다. trial마다 model_spec/ensemble_spec 멤버 하나만 오버라이드하는 얇은
  어댑터(`_SingleModelTrialPipeline`/`_EnsembleMemberTrialPipeline`)로 confirmed pipeline을 감싸고
  `evaluate_pipeline`을 그대로 재사용 — attempt 평가와 동일한 CV 방법론(is_original 인지 분할,
  leak 가드) 보장. `PipelineContext`에 X/y/scoring은 추가하지 않는다(#230 배경이 명시한 실패
  패턴 재발 방지).
- `MODEL_REGISTRY`(evaluator/models.py, #229) 8개 모델 전부에 대응하는 검색공간 정의, 각 공간을
  실제 생성자로 검증하는 회귀 테스트(파라미터 키 어긋남을 build 시점이 아니라 여기서 잡음).
- 신규 테이블 `raw.tuned_params`(tuning_run_id로 한 번의 튜닝 실행 묶음), `PipelineContext.tuned_params`
  추가 필드(advisory, `ctx.best_params`와 동일 계약) — `cycle/run.py:_latest_tuned_params`가 가장
  최근 튜닝 실행의 개선된(improved=True) 항목만 다음 attempt의 model_spec/build_model 훅에
  advisory로 흘려보낸다. `agents/coder.py` 컨트랙트 프롬프트에 사용법 명시.
- 신규 `bin/tune_pipeline.py` CLI — 별도 Airflow DAG(`reflexion_rondo_tune`, airflow-stack repo)
  전용 진입점. attempt와 달리 `runtime/isolate.py`의 subprocess+RSS 워치독 격리를 거치지 않고
  in-process로 돈다(trial마다 subprocess를 새로 띄우면 튜닝의 효율 이점이 사라짐) — 컨테이너
  mem_limit이 유일한 메모리 백스톱(ADR-035 트레이드오프로 명시).
- **merge 전 백테스트(완료 기준)**: deep tier confirmed pipeline 2개(s4e10/s4e11)에 실제 원본
  preprocess/feature_transform을 보존한 채 lgbm model_spec을 얹어 25-trial 튜닝 — 둘 다 실측
  개선 확인(s4e10 auc +0.0020, s4e11 accuracy +0.0257, 둘 다 `improved=True`).
- 부수 발견: 완전히 빈(최초 부트스트랩) Postgres에 `apply_schema`를 1회만 돌리면 statement
  순서 문제로 실패하는 사전 존재 버그 발견(#243로 이슈화, 프로덕션 무영향 — 이미 마이그레이션된
  DB는 재현 안 됨).

## v1.5.12 — 선언형 단일 모델(model_spec, ADR-034), 모델 레지스트리 harness에서 분리 (2026-08-24)
- #229(Milestone v1.6.0, keystone): `model_swap`에 7번째 훅 `model_spec(self, ctx) -> dict | None`
  추가 — `ensemble_spec`(ADR-023)의 단일 모델 버전. `{"model": <registry key>, "params": {...}}`만
  선언하면 harness가 레지스트리에서 직접 생성자를 호출해, LLM이 손으로 쓴 `build_model`의
  super() 오용·stale kwarg·오탈자 클래스명 문제가 구조적으로 발생하지 않는다.
- 모델 레지스트리(`_ENSEMBLE_MODEL_REGISTRY`)를 `evaluator/harness.py`에서 신설
  `evaluator/models.py`(`MODEL_REGISTRY`)로 승격 — `ensemble_spec`과 `model_spec`이 동일
  레지스트리·동일 생성자 헬퍼(`build_registry_model`/`construct_with_kwarg_retry`)를 공유해
  드리프트가 구조적으로 불가능해졌다. `extra_trees`/`elastic_net` 두 모델 추가.
- `PatchedPipeline.ensemble_spec`/`model_spec`이 서로의 존재(및 `build_model`)를 상속 억제
  조건에 넣도록 대칭 확장 — #239가 고친 `ensemble_spec`↔`build_model`/`param_candidates` 상호
  억제를 `model_spec`까지 반영하지 않으면, base가 `ensemble_spec`/`model_spec` 중 하나를 갖고
  patch가 다른 쪽을 선택했을 때 상속이 patch의 실제 의도를 가릴 수 있었다.
- `fit_predict()`(#239 공유 진입점)가 `ensemble_spec` → `model_spec` → 자유형 `build_model`
  순으로 라우팅하도록 확장, `preselect_params`도 `model_spec` 존재 시 탐색을 건너뛴다.
- `EvalResult.model_type`(model_spec이 선언한 레지스트리 이름, 그 외는 None) 신설 —
  `runtime/isolate.py:IsolatedResult` → `cycle/run.py` → `raw.attempts.model_type`(기존
  스키마 컬럼, 지금까지 항상 NULL이었음)까지 배선해 모델 다양성을 추적할 수 있게 했다.
- `agents/coder.py` 컨트랙트 프롬프트에 model_swap 전용 model_spec 섹션 추가(ensemble_spec
  섹션과 동일 구조), `evaluator/contract.py`(`_ALL_HOOKS`/`_ALLOWED_HOOKS['model_swap']`/
  `_HOOK_ARITY`) 갱신.
- 회귀 테스트 20여 개 추가: model_spec 상속/억제 대칭성, preselect_params 우회,
  fit_predict 라우팅, contract 레벨 arity/allowed-hooks 검증(ensemble_spec 테스트 세트를
  그대로 미러링).

## v1.5.11 — 원본 데이터 병합: deep tier 5개 전부 배선 (2026-08-24)
- #228(Milestone v1.6.0): `EXTRA_TRAIN_PATHS` 병합 메커니즘(`store/train_data.py`)은
  이미 있었지만 27개 대회 전부 빈 배열로 방치돼 있던 것을, deep tier 5개(`s4e10`/
  `s4e11`/`s5e4`/`s6e8`) 중 Kaggle에서 컬럼 일치가 확인된 4개에 실제 원본 데이터셋을
  연결(`s4e12`는 신뢰할 만한 원본을 못 찾아 보류). MinIO `kaggle/{slug}/data/
  original.csv`에 업로드 + 로컬 `data/playground-series-*/`에도 배치.
  - `s4e10`(Loan Approval): `laotse/credit-risk-dataset`, 컬럼 완전 일치.
    `#225` 스파이크 실험에서 이미 실측(5-fold 전부 개선, +0.0018 AUC).
  - `s4e11`(Depression): `hopesb/student-depression-dataset`, 19개 중 17개 컬럼 일치
    (학생 전용 서브셋이라 `Name`/`Working Professional or Student` 없음 — 자동 null 채움).
  - `s5e4`(Podcast): `sangampaudel530/original-podcast-dataset`, 컬럼 완전 일치.
  - `s6e8`(Smartphone Addiction): `zahranusratt/smartphone-usage-and-addiction-
    analysis-dataset`, 핵심 피처 전부 일치(`transaction_id`/`user_id`/`addiction_level`은
    대회에 없는 컬럼이라 자동 배제).
  - **`evaluator/harness.py`에 `is_original`-aware 분할 신설**: `_make_folds`/
    `split_audit_holdout`/`preselect_params`가 원본(실제) 데이터 행을 모든 fold·holdout·
    inner-split의 **train 쪽에만** 넣고 validation/holdout 쪽에는 절대 안 넣도록 수정
    — CV/holdout이 재는 대상이 실제 Kaggle 제출 조건(100% synthetic)이어야 하는데
    원본 행이 섞이면 그 프록시가 오염된다(leakage는 아니지만 측정 대상 자체가 달라짐).
    `runtime/runner.py`/`bin/submit.py`는 Patch 훅에 넘기기 전 `is_original` bookkeeping
    컬럼을 제거하도록 배선.
  - CV_SCHEME(Group/TimeSeries 오버라이드)은 deep tier 5개 전부 iid 데이터라 불필요 —
    도입 안 함.
  - 회귀 테스트 8개 추가(fold/holdout/preselect 분할 검증 + Patch 훅에 컬럼 미노출 검증).

## v1.5.10 — #226 adversarial review 후속: ensemble_spec 상속 무력화 + 조기종료 재시도 (2026-08-24)
- #239: `/code-review max`로 c8a116b..HEAD(#221~#227) 전체를 adversarial review한 결과,
  #226 자체는 올바르지만 그 수정이 정확히 동작하기 시작하면서 새로 노출되는 문제 2건 발견.
  - **critical**: `PatchedPipeline.ensemble_spec()`이 patch가 직접 정의 안 하면 base로
    무조건 폴백해, 확정 best가 ensemble이면 `model_swap`/`hyperparam_search`
    (`evaluator/contract.py:_ALLOWED_HOOKS`가 이 둘엔 `ensemble_spec` 정의 자체를 금지)
    attempt가 매 사이클 base와 완전히 동일한 cv_score만 재현하며 영구 무력화됐다 —
    #226 이전엔 base가 애초에 ensemble_spec을 못 가져와 불가능했던 시나리오. patch가
    `build_model`/`param_candidates`를 직접 정의하면(단일모델 의도가 명확하면) 상속을
    끊도록 수정.
  - `_fit_predict_ensemble`의 `yva=None`(제출·holdout) 분기가 `_fit_full_train`(#71)의
    조기종료 kwarg 재시도 안전망 없이 그냥 `fit()`을 호출해, ensemble 멤버 params에
    조기종료 키가 있으면(코더 컨트랙트가 명시적으로 허용) 제출이 죽거나(에러) holdout
    게이트가 조용히 무력화될 수 있었다.
  - 세 호출부(`evaluate_pipeline`/`_eval_holdout`/`_bagged_predict`)가 ensemble 분기
    로직을 개별 복제하던 것(#226 PR에서 통일하겠다고 했으나 미완이었음)을
    `evaluator/harness.py:fit_predict()` 공유 헬퍼로 실제 통일 — `bin/submit.py`의
    `_fit_full_train`/`_predict_raw`/`_EARLY_STOPPING_KEYS`는 이제 죽은 코드라 삭제.
  - 부수: `run_daemon.py` ACTIVE 필터의 오해의 소지 있는 코멘트 정정(동결 대회는
    auto-submit 24h 윈도우로 자연 배제됨), `importlib.import_module` 실패 시 슬러그
    하나만 건너뛰도록 가드(#223과 같은 클래스의 크래시 방지), 테스트 fixture의
    로컬 타임존 의존성(#223을 막으려던 테스트에서 재발할 뻔함) 제거.
  - 회귀 테스트: ensemble base 위 model_swap/hyperparam_search가 실제로 자기 훅을
    호출하는지, `PatchedPipeline`이 `evaluator/contract.py:_ALL_HOOKS`와 계속
    동기화되는지 검증하는 영구 테스트 추가.

## v1.5.9 — fleet 동결: 27개 동시 운영 → deep tier 5개 (2026-08-24)
- #227(ADR-032, Milestone v1.6.0): `config/competitions/*.py`에 `ACTIVE` bool 추가.
  `s6e8`/`s4e12`/`s4e10`/`s5e4`/`s4e11` 5개만 `ACTIVE=True`로 남기고 나머지 22개는 동결
  (attempts 이력 보존, 삭제 아님). `bin/run_daemon.py:_sweep_queue_refill`이 `ACTIVE=False`
  대회를 idle 재보급 대상에서 제외하도록 배선. 선정 기준: 최근 confirmed 갱신(아직 얕은 과실
  있음) + `#225` 스파이크 실험으로 헤드룸 검증(s4e10) + task_type/metric 4종 다양성. 배경은
  ADR-032 — attempts 최다 3개 대회가 정확히 가장 오래 정체된 대회였다는 실측(컴퓨트 문제
  아님)과 breadth 전략의 근거였던 transfer 가설이 `#76`으로 이미 반증됨.

## v1.5.8 — ensemble_spec이 제출·CV 재평가 경로에서 누락되던 3중 버그 수정 (2026-08-23)
- #226(v1.6.0 "천장 돌파" 진단 중 발견): `bin/submit.py`가 `ensemble_spec`을 호출하는
  코드가 아예 없어, 확정된 ensemble pipeline이 제출 시점엔 조용히 단일 `build_model`로
  대체됐다. 더 심각하게는 `runtime/runner.py`의 `_load_best_pipeline_class`가 고정 훅
  이름 목록(`_HOOK_NAMES`)으로 base pipeline을 재구성하면서 `ensemble_spec`을 빠뜨려,
  ensemble이 한 번 승격되면 그 다음 사이클부터 모든 attempt의 CV 재평가 base가 조용히
  non-ensemble로 퇴화했다(`prev_best_cv`는 ensemble 기준인데 이후 비교는 퇴화한 base
  기준이라 영구 정체를 만드는 메커니즘). audit holdout도 동일 사각지대 공유.
  `bin/submit.py`가 이미 겪고 고친 것과 같은 클래스의 버그(#83, `type(...)` 훅 복사가
  훅 밖 클래스 속성/nested class를 유실)라 `runtime/runner.py`도 동일하게
  `PatchedPipeline(BasePipeline(), patch_cls())`로 통일. `_fit_predict_ensemble`에
  `yva=None`(조기종료 없는 전체학습) 분기를 추가해 submit/holdout 양쪽이 실제로 이
  경로를 탈 수 있게 배선. `score_progression`/`cold_start_progression`의 `best_so_far`가
  확정 여부와 무관한 raw max라 `label`(confirmed 기준, ADR-025)과 괴리될 수 있다는
  점도 발견해 스키마 주석으로 명시(s6e1 실측: 30일 jump 라벨 43건, best_so_far는
  463회 무변화 — 별도 버그 아니라 의도된 정의 차이).

## v1.5.7 — daemon 크래시루프: _sweep_queue_refill naive/aware datetime 비교 수정 (2026-08-23)
- #223: `#196`의 `_sweep_queue_refill`이 timezone 없는 `raw.attempts.run_ts`(naive)를
  aware `idle_cutoff`와 Python에서 직접 비교해 `TypeError`. `raw.cycle_queue`가
  하나라도 pending/running이면 건드리지 않는 가드가 있어 지금까지 발화 안 하다가,
  v1.5.6 컷오버 시점에 큐가 처음으로 완전히 드레인되며 매 daemon startup마다 즉시
  크래시루프. `idle_cutoff`를 naive UTC로 맞춰서 비교하도록 수정. 기존
  `tests/test_daemon_queue_refill.py`도 aware `run_ts`를 mock해 이 버그를 못 잡고
  있었던 것을 naive로 정정.

## v1.5.6 — auto-submit AmbiguousColumn 500 수정 (2026-08-23)
- #221: `#178`에서 `_best_attempt()`에 `raw.pipelines` JOIN을 추가하면서 그 아래
  `join raw.competitions c using (competition_id)`가 그대로 남아 `raw.attempts`/
  `raw.pipelines` 양쪽 `competition_id`가 충돌, `psycopg2.errors.AmbiguousColumn`으로
  매 호출이 500을 냈다. 배포 시점(2026-08-18 22:40) 이후 auto-submit이 한 번도
  성공하지 못해 Kaggle 제출이 3일간 0건이었다. `using` → `on c.competition_id =
  a.competition_id`로 명시.

## v1.4.29 — s6e8 재가동: auto-submit confirmed-only + 트립와이어 데드밴드 + CPU 상한 (2026-08-19)
- #178(근본원인): `bin/api.py:_best_attempt()`가 raw.attempts 전체 max cv_score를 확정
  여부와 무관하게 골라, `--attempt-id`(원래 사람이 미확정 attempt를 수동 지정하는
  escape hatch)로 그대로 제출되고 있었다. 완료 Kaggle 제출 74건 중 61%가 cross-seed
  +holdout 확정을 한 번도 거치지 않은 코드였다. `_best_attempt()`를 raw.pipelines
  JOIN으로 제한(ADR-031).
- #175: cv-LB 발산 트립와이어(ADR-026)가 배포 이후 정지시킨 5개 대회 전부 노이즈
  오탐(`|delta_lb|`가 `|prev_lb|`의 0.05% 미만)이었다. `|prev_lb|`의 0.1% 데드밴드 +
  최근 3개 delta 중 2개 이상일 때만 정지하도록 완화.
- #176: s6e8 CPU 예산(900s) kill률 35%(직전 활동일 80%), 성공 attempt p99=841s로
  벽에 붙어 있었다. `CycleConfig.cpu_budget_secs`(comp.CPU_BUDGET_SECS 오버라이드)
  배선 후 s6e8을 3600s로 상향 — 검열된 측정을 풀어 실제 분포 확보(#182에서 관측 후
  영구값 결정).
- #174/#177: 정지 해제 + 확정 pipeline 교정 제출(cv 0.9638775, lb 0.96501 — CV↓ LB도
  같이 소폭↓, 역상관 아님 확인) 후 30 cycle 재적재.

## v1.4.28 — raw.confirm_memo.fold_scores 타입 정정 (2026-08-16)
- #171: `raw.confirm_memo.fold_scores`가 float[] 배열 컬럼으로 생성됐는데
  `PromotionCache.put_confirm_memo`는 JSON 문자열을 넘겨 타입 불일치로 저장이
  실패하고 있었다. jsonb로 정정.

## v1.4.27 — confirm 게이트 캐시(negative memo + baseline eval) (2026-08-11)
- #166/#167/#168: 15분+ 걸리는 promote 케이스가 confirm 게이트 중복 재계산 때문임을
  실측(s6e1 confirm 39회가 (cv_score, fold_scores) 행동 지문 기준 3그룹으로 붕괴 —
  35회가 이미 확정된 거부 판정을 재계산). 소스 해시 기반 dedupe는 안 통한다(같은
  cv_score를 내는 34개 후보가 서로 다른 AST) — 행동 지문(cv+fold_scores)만 유효.
  `raw.confirm_memo`(negative-only) + `raw.baseline_eval_cache`(변하지 않는 best
  pipeline 재평가 방지) 신설, 승격 경로 4곳에 배선. promote p95 20배 개선 확인.

## v1.4.26 — bandit/lesson 보상 신호를 confirm 결과와 연동 (2026-08-11)
- #164: promote task가 가끔 22분(cross-seed 3 + holdout 2 = 순차 풀 CV 5회) 걸리는
  이유를 추적하다가 발견 — s6e1의 `preprocessing` 후보가 cv_score 소수점 10자리까지
  동일한 채로 32회(2026-08-09~10, 하루 이상) 재생성돼 매번 22분짜리 confirm을
  돌고 매번 holdout 악화로 거부당했다.
- 원인: `cycle/action_optimizer.py:update_bandit`이 attempt 생성 시점의 잠정
  label(`is_significant_gain` 통과 시 "jump")에 α+=1.0(최강 보상)을 주는데, 이
  호출이 `defer_promotion` 여부와 무관하게 무조건 실행된다. 프로덕션(airflow
  모드)에서 confirm(cross-seed+holdout)은 별도 task(`bin/run_promote_task.py`)가
  승자만 나중에 도는데, 그 결과가 bandit에 전혀 반영 안 됐다(이 파일이
  `update_bandit`을 아예 호출 안 함). jump→α+=1.0→다음 cycle 당첨 확률↑→비슷한
  아이디어 재생성→다시 jump→다시 α+=1.0→... 자기강화 루프가 confirm 결과와
  무관하게 돌아간 구조. `reflect()`도 동일 결함으로 confirm 이전 raw label을
  그대로 lesson에 반영했다.
- `cycle/promotion.py`에 `effective_label(label, confirm)` 신설 — confirm이
  jump를 거부하면 regression으로 다운그레이드. `cycle/run.py`(직접모드)와
  `bin/run_promote_task.py`(프로덕션 경로) 양쪽의 bandit 보상·reflect lesson에
  적용. `raw.attempts.label`(DB 원본)은 건드리지 않고 하류 학습 신호만 보정.
  ADR-030.

## v1.4.25 — 생성 코드의 무제한 병렬성(n_jobs=-1 등) 정적 거부 (2026-08-10)
- #162: #159 CPU 예산 워치독 배포 후 리소스 상황을 점검하다가 발견 — Airflow
  attempt 컨테이너(`cpus=1.5`)는 실제로는 `cpu_shares`(상대 가중치)로만 반영돼
  하드 CPU quota가 아니고(`CpuQuota=0` 실측 확인), `OMP_NUM_THREADS=2`류 env var는
  LightGBM/CatBoost/XGBoost/scikit-learn의 `n_jobs`/`thread_count` 파라미터와
  무관하게 동작한다(실측: `n_jobs=-1` LightGBM 20 threads/15.9x cores, CatBoost
  `thread_count=-1` 21 threads/15.0x, sklearn RandomForest `n_jobs=-1` 43
  threads/15.6x). mac-server는 big 큐 슬롯 2개가 4vCPU를 공유해 이런 생성 코드
  하나가 sibling attempt를 실제로 굶길 수 있었다. `evaluator/contract.py`에
  `n_jobs`/`thread_count`/`num_threads`/`nthread`/`n_threads` 0 이하 리터럴을
  정적 거부하는 검사 추가. ADR-029.

## v1.4.23 — eval CPU 예산 워치독: rc=-9 오진과 재시도 이중 소모 제거 (2026-08-09)
- #159: RSS 워치독(#154, v1.4.21) 배포 후 2일 실측 — 전체 attempt 계산시간 56.3h 중
  40%(22.4h)가 `RLIMIT_CPU(soft=hard=900)`의 흔적 없는 SIGKILL(rc=-9)이었다. 커널이
  hard 한도를 먼저 검사해 SIGXCPU 경고 없이 곧장 죽여 OOM killer 사망과 문자열이
  동일했고(오진의 근본원인), `cycle/run.py`가 이 무의미한 원문을 LLM 재생성 피드백에
  그대로 넘겨 2회차 eval도 같은 자리에서 또 죽었다(rc=-9 attempt 113건 전부 재시도
  경로, attempt당 최대 ~1800초). ADR-028.
- `runtime/isolate.py`: 기존 RSS 폴링 루프(2초 주기)가 CPU 시간도 함께 감시해
  `"cpu budget exceeded: ..."`로 선제 kill. `RLIMIT_CPU`는 폴링이 놓쳤을 때만
  발동하는 soft<hard 백스톱(`budget+60`/`budget+120`, rc=-24)으로 강등. 신규
  `peak_cpu_sec` 컬럼(`peak_rss_bytes`와 동일 계약)으로 예산 900의 적정성을 계측.
- `cycle/run.py`: CPU 예산을 eval 회차가 아니라 attempt 전체 기준으로 집행 — 1회차가
  예산을 다 쓰면 2회차를 아예 돌리지 않아 최악 소모가 절반으로 준다. 재생성 피드백도
  리소스 kill 원문 대신 "n_estimators/n_splits/탐색 후보 수를 줄여라" 같은 실행 가능한
  지시로 교체.
- `cycle/error_pitfalls.py`: `normalize_error`의 `\d+` 스크럽이 `rc=-9`/`rc=-11`/
  `rc=-24`를 전부 같은 시그니처로 뭉개던 것을 수정 — `runner exited without
  output.json (rc=-N)` 패턴은 신호 번호를 보존해 원인별 집계가 가능하게 함.

## v1.4.19 — 대시보드 Fleet Overview (2026-08-03)
- #143: 대시보드 최상단에 전역 Fleet Overview 신설 — 대회별 큐 상태·confirmed/quarantined pipeline 수·14일 attempt/jump/error/OOM
카운트·`auto_submit_paused_reason`을 벌크 쿼리 5개(N+1 아님, 실측 0.085s)로 모아 attention 배지(🔴/🟡/🟢)로 정렬해 어디부터 볼지 여기서 고르게 함. 대회 선택 종속
섹션으로 Submissions·Quarantine·Blend 신설 — `cv_lb_calibration` 제출 이력(발산 경고), 격리 pipeline 목록, `auto_submit_paused_reason` 경고,
blend_cv_score vs 단일 best pipeline 비교. daemon API 미경유, 전부 Postgres 직접 쿼리(GH #65 설계 그대로).

## v1.4.18 — 대시보드 콘솔 경고·느림 수정 (2026-08-03)
- #141: 캐싱 부재로 `_query_df`가 rerun마다 재조회하던 문제에 `@st.cache_data(ttl=60)`(`bin/api.py` 60초 TTL 관례와 동일) 적용. Health
Signals·상세 섹션이 같은 쿼리(lesson_funnel/bandit_calibration/error_recurrence)를 중복 호출하던 것 제거. CV Progression 차트
`.interactive()` 제거 + 명시 타입으로 Scale binding 콘솔 경고 해소, Bandit 차트 null 열 드롭으로 Infinite extent 경고 해소.
- 부수 수정: `lesson_duplicates` 뷰의 O(n²) 자기조인을 대회당 최근 250건으로 캡핑 — s4e1 12.18s → 0.29s.

## v1.4.17 — 대시보드 건강 신호등 + 관측 패널 6종 (2026-08-03)
- #65: `dashboard.py`(Streamlit, Postgres 직결, daemon API 미경유)에 Health 신호등 4칸(`bin/api.py:/api/reflexion-health`와 동일 임계값을
API 호출 없이 재현) + CV progression jump 마커/정체 경고 오버레이 + Lesson Funnel/Dead/Duplicates + Bandit Calibration + Error
Recurrence + Transfer Matrix 히트맵(pandas Styler) 추가. `docs/decisions.md` 등 별도 ADR 없이 기존 GH #65 설계(뷰 직접 소비) 그대로 구현.
- 부수 수정: `_rows_df`가 polars 기본 `infer_schema_length=100`으로 앞쪽 100행이 전부 null인 컬럼(예: holdout_score)에서 타입을 오추론해 뒷행 실측값에서
크래시하던 문제 수정(`infer_schema_length=None`) — s5e10 등 attempt가 많은 대회에서 대시보드 자체가 안 뜨던 원인.

## v1.4.16 — submit.py nested class 소실 버그 수정 (2026-08-03)
- #137: `bin/submit.py:_load_pipeline`의 기본(non-attempt_only) 경로가 Patch 훅 메서드만 `type(...)`으로 새 클래스에 옮겨 붙여, 훅 밖 nested
class(자유형 ensemble wrapper의 `_EnsembleRegressor` 등)를 소실시켜 `AttributeError`로 제출이 크래시하던 문제 — s5e10 신규 clean baseline 재제출
실제 프로덕션 크래시로 발견. `attempt_only` 경로는 이미 `PatchedPipeline`으로 고쳐져 있었는데 기본 경로만 별도 구현이라 놓쳐 있었음. 두 경로 다
`PatchedPipeline(BasePipeline(), patch_cls())`로 통일.

## v1.4.15 — Ollama Cloud 재시도 + quarantine_leaks 오판 수정 (2026-08-02)
- #131: attempt task 60개 중 40개(67%)가 Ollama Cloud "model temporarily overloaded" 503에 크래시하던 문제 — `agents/llm_retry.py`
신규(지수 백오프 1/4/16초, `memory/retriever.embed`와 동일 패턴), `coder.py`/`strategist.py`/`reflector.py` 호출부 3곳 적용. 배포 후 실측 60→0%
근접까지 개선 확인.
- #120: `bin/quarantine_leaks.py`가 코드 exec/데이터 로드 실패(판정 불가)를 누수 확정과 동일하게 처리해 격리 대상으로 잘못 집계하던 문제 수정 — `--dry-run` 프로덕션 실측
중 s4e1/s5e3 정상 파이프라인 27개가 로컬 데이터 캐시 부재만으로 부당하게 격리될 뻔한 사고를 계기로 발견.
- #122~125: 문서·주석 싱크 정리 Milestone — ADR-023~027 신설, spec.md/README `compound`(실재한 적 없던 action_type) 삭제 및 `ensemble_spec`
반영, architecture.md/runbook.md 운영 정합성, 17개 파일 480줄 규칙 위반 주석 정리.

## v1.4.14 — daemon 크래시루프 수정: naive/aware datetime (2026-08-02)
- #118: `bin/run_daemon.py:_submission_refresh_due`가 DB에서 읽은 naive
`timestamp`(`raw.kaggle_submissions.submitted_at`/`checked_at`)를 `datetime.now(timezone.utc)`(aware)와 직접 빼서 `TypeError`로
daemon이 크래시루프에 빠짐 — v1.4.13 배포 직후 프로덕션에서 실제 발생. 로컬 mock 테스트는 양쪽을 전부 aware로 구성해 이 조합을 못 잡았다.
- 세 값(`now`/`submitted_at`/`checked_at`) 모두 naive로 정규화 후 뺄셈하도록 수정, DB round-trip을 실제로 거치는 회귀 테스트 2종 추가.

## v1.4.13 — 학습 정체 근본 원인 수정 (측정 정합성 + 승격 래칫 + LB 되먹임 + 컴퓨트 회수) (2026-08-02)
"reflexion 상한선이 아닌데 성능 향상이 없다"는 진단에서 나온 4갈래 구조적 결함을 한 번에 배포. Milestone 5개(M1~M5), 이슈 12개.

**M1 측정 정합성**
- #97: `preprocess` 훅이 valid split의 타깃 컬럼을 직접 읽는 누수를 fold0 동등성 검사(마스킹 전/후 결과 비교)로 실측 차단 + AST 정적 가드.
`_REGRESSION_IMPLAUSIBLE_BASELINE_RATIO` 100→10 하향.
- #98: audit holdout을 `bin/submit.py`와 동일한 dummy target 치환으로 재현해 실제 추론 조건과 일치시키고, `holdout_regressed`를 승격 조건에 AND 결합(기록용→차단 게이트).
- #99: `raw.pipelines.invalid_reason` 컬럼 + `bin/quarantine_leaks.py` 신설 — 확정 후 누수로 밝혀진 파이프라인을 격리(삭제 아님). 모든 baseline 조회
경로에 `invalid_reason IS NULL` 필터 적용.

**M2 승격 래칫 복구**
- #100: bootstrap 종료 시 최고 attempt를 `confirm_and_measure`로 검증해 확정 baseline 자동 확립(`cycle/run.py:establish_bootstrap_baseline`).
- #101: 기존에 확정 파이프라인이 없던 대회를 위한 소급 스크립트 `bin/establish_baseline.py`(top-k 순회, `--dry-run` 지원).
- #102: "확정 파이프라인 없으면 전체 attempt의 max(cv)로 폴백"하던 phantom-max 분기 제거 — N이 늘수록 문턱이 같이 올라가는 자기강화 데드락의 근본 원인.

**M3 LB 되먹임 연결**
- #103: daemon 유휴 틱마다 `status IN ('submitted','pending')` 제출을 지수 백오프로 재폴링 — 기존엔 업로드 직후 1회뿐이라 pending이면 수동 refresh 전까지 영구 방치됐다.
- #104: `cv_lb_calibration` 뷰 + 발산 트립와이어 — CV는 개선인데 LB가 악화된 제출이 나오면 원천 pipeline 격리 + 해당 대회 auto-submit
중단(`auto_submit_paused_reason`, 자동 해제 없음).

**M4 컴퓨트 회수**
- #84: `comp.MAX_TRAIN_ROWS` opt-in + 층화 샘플링(`store/train_data.py`) — s4e7(11.5M행)이 100-cycle 큐 전량 OOM되던 문제, 1.5M으로 상한 적용.
- #74: `ensemble_spec` 선언형 앙상블 프리미티브 — 자유형 wrapper 클래스 크래시(70%→55%에서 정체)의 구조적 원인(exec된 클래스 몸체 내부는 harness가 볼 수 없음)을 해소.
Patch는 멤버·결합방식만 선언, harness가 생성·적합·결합 전담.

**M5 탐색 능력**
- #75: `raw.blend_weights` 테이블 신설 + `bin/blend.py:compute_and_store_blend`를 양쪽 승격 경로에 배선 — 계산·저장까지만(`bin/submit.py`는 의도적으로 미연결, 범위 밖).
- #76: `bin/archive_lessons.py`를 daemon 24시간 스윕으로 배선 — 이전엔 CLI 수동 실행뿐이라 사실상 죽은 레버였다.

## v1.4.12 — submit build_model 안전망 (2026-07-25)
- #94: `bin/submit.py`의 `build_model()` 생성자 stale kwarg 문제에 평가 경로(#79)와 동일한 런타임 안전망 적용.

## v1.4.11 — MinIO 텍스트 다운로드 mojibake 수정 (2026-07-25)
- #92: charset 없는 `text/plain` 응답을 latin-1로 잘못 디코드하던 문제 — UTF-8 명시 디코드로 수정.

## v1.4.10 — materialized_code 스냅샷 + 백필 (2026-07-24)
- #89: 승격 시점 병합본 스냅샷을 `raw.pipelines.materialized_code`에 저장. attempt 제출 base를 replay 대신 이 스냅샷으로 로드(replay 폴백은 sha 불일치 시
중단). 기존 승격 행을 위한 백필 스크립트 신설.

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
- #74 후속(#79): `build_model()` 생성자 stale kwarg 런타임 안전망 추가 — 프롬프트 경고만으론 재발을 못 막았다. (이 3라운드로도 크래시율은 70%→55%까지만 떨어졌고, 근본
해결은 v1.4.13의 #74 선언형 프리미티브로 이관.)

## v1.4.2 — submit 생성자 early-stopping 파라미터 크래시 수정 (2026-07-22)
- #72: 생성자 early-stopping 파라미터가 submit full-train fit을 크래시시키던 문제 수정.

## v1.4.1 — reflection_impact DROP VIEW CASCADE (2026-07-22)
- #69: `lesson_dead`가 `reflection_impact`에 의존해 CASCADE 없이는 재적용이 실패하던 문제 수정.

## v1.4.0 — 관측 API 24개 + 파생 뷰 6개 + 대회 온보딩 14개 (2026-07-22)
- #66/#67: 관측 계측 P1/P2(`retrieved_ids`/`error_signature` 영속화) + 파생 뷰
6종(funnel/dead/duplicates/calibration/error_recurrence/transfer) + 엔드포인트 24개(score/timeline, bandit/posteriors, lessons,
errors, transfer 등).
- #39: 대회 온보딩 14개 — binary AUC 6개, accuracy/balanced_accuracy 3개, regression 5개.

## v1.3.1 — auto-submit 캐시 트리거 수정 (2026-07-22)
- #62/#63: auto-submit 제출 CSV 캐시가 실제로 채워지지 않던 트리거·키 문제 수정.
- #61: 코드베이스 전체 이슈 번호 태그 정리 + 문서 최신화.

## v1.3.0 — gain_vs_best_relative: metric 스케일 정규화 (2026-07-22)
- `reflection_impact` 전역 z-score가 metric 스케일 혼합(rmse 원시 단위 vs auc 0~1)으로 오염돼 있던 문제(mean=-4.22, std=139.19 실측) — DB
wipe로는 해결 안 되는 코드 자체의 스케일 정규화 부재가 근본 원인으로 확인됨.
- `gain_vs_best_relative` 컬럼 신설(regression_error는 `gain_vs_best / baseline_cv` 상대값, 나머지 metric_class는 `gain_vs_best`
패스스루) — `evaluator/harness.py` → `runtime/isolate.py`/`runtime/runner.py`(subprocess JSON 경계) → `cycle/run.py` →
`store/schema.sql`까지 전체 파이프라인에 배관.
- `reflection_impact` 뷰가 이 컬럼만 집계하도록 재정의 — `gain_vs_best_relative IS NULL`인 legacy row는 집계에서 제외(raw `gain_vs_best`로 폴백하지 않음).
- 프로덕션 실측으로 정규화 검증: raw `gain_vs_best=-110.92` → `gain_vs_best_relative=-0.0014`.

## v1.2.32 — Kaggle 자동 제출 gap 진단 및 수정 (2026-07-22)
- s5e7/s6e6 자동 제출 누락 재진단: (a) `bin/submit.py` dummy target 생성이 문자열 타깃 대회에서 크래시하던 버그, (b)
`raw.kaggle_submissions.status='submitted'`가 폴링 데드라인을 지나도 안 풀려 재제출이 영구 스킵되던 버그.
- 두 경로 모두 수정 후 실제 Kaggle 제출로 종단 검증 — s5e7 LB 0.973279, s6e6 LB 0.96408로 `complete` 상태 도달 확인.

## v1.2.31 — subprocess 고아 프로세스 kill + degenerate 회귀 gain 클립 + s6e7 코드생성 harness 버그 (2026-07-22)
- #37: `bin/api.py`의 `subprocess.run` 타임아웃 처리가 `uv run`이 spawn한 손자 python 프로세스를 못 죽여 고아로 남던 문제(타임아웃 기록 후에도 CPU 85%+ 점유
실측) — `_run_in_pgroup` 헬퍼로 교체, `start_new_session=True` + 타임아웃 시 `os.killpg`로 프로세스 그룹 전체 종료.
- #43: rmse degenerate 예측(모델이 완전히 빗나감)이 만드는 극단적 `gain_vs_best`(s6e1 실측 -105448)가 `reflection_impact` 전역
z-score(BON-195)를 오염시키던 문제 — `evaluator/harness.py`에 대칭 가드 추가(issue #4의 "100배 좋으면 raise"에 대칭으로 "100배 나쁘면 gain_vs_best 하한
클립", label 판정은 영향 없음).
- #42: s6e7(multiclass) 에러율 65% 재진단 결과 다수가 Coder 코드 문제가 아니라 harness 자체 버그(`_encode_residual_categoricals`가 null 섞인 문자열
컬럼 정렬 시 크래시, ADR-014 amend)로 확인 — null 제외 정렬로 수정. 추가로 multiclass 라벨 왕복(round-trip) 컨트랙트 규칙과 action_type별 hook 동적 강조를
`agents/coder.py`에 반영.

## v1.2.28~30 — 대회 온보딩 배치 3건 + 운영 버그 수정 (2026-07-17~19)
- #24 (2026-07-17): 신규 대회 3개 추가 — s6e7(multiclass), s6e1(regression), s5e9(regression).
- #25/#26 (2026-07-17): `release.sh`의 registry 태그 존재 확인이 OCI 매니페스트 타입 미인식 + repo path에 태그가 남는 버그로 늘 실패하던 문제 수정.
- #27/#28 (2026-07-17): attempt eval `RLIMIT_AS` 기본값을 1.5GiB→6GiB로 복원(#20이 낮췄던 값 — mac-server Colima VM을 8→16GiB로 증설해 근본
해결, 신규 대회 bootstrap 전체가 SIGKILL(rc=-9)되던 원인).
- BON-275 (2026-07-19): 제출(`bin.submit`) 타임아웃 600s→1200s 상향 — s5e5(75만 행) 5-seed bagging이 기존 값을 넘겨 매번 타임아웃.
- #29 (2026-07-18): README에 "지원 대회 조건" 섹션 추가.
- #31 (2026-07-18): promote 시점 submission CSV 캐싱 — 매일 auto-submit이 daemon(ops-vm, 2 OCPU)에서 매번 fit하며 CPU를 포화시키던 경로 대체.
- #33 (2026-07-19): `release.sh` post-restart heartbeat 체크가 노출 안 된 `localhost:8000` 대신 `rondo-api.internal` 사용하도록 수정.
- #35 (2026-07-19): `attempt_only` 재구성이 훅 메서드만 옮겨 붙이는 lossy transplant라 Patch의 훅 밖 클래스 속성이 소실되며 s6e7 auto-submit이 크래시하던 문제 수정.
- #40 (2026-07-19): 대회 4건 온보딩 — s4e2(multiclass)/s5e7(binary)/s4e9(regression)/s6e2(binary), accuracy 메트릭 최초 실사용.

## 이미지 빌드를 Airflow DAG로 이관, release.sh는 daemon 전용으로 축소 (2026-07-14)
- ADR-022: daemon+task 이미지 빌드+push를 airflow-stack의 `reflexion_rondo_deploy` DAG(ops 큐 docker.sock 재사용)로 이관. 여러 repo가
재사용할 공용 헬퍼(`dags/lib/image_deploy.py`, airflow-stack)라 신규 credential이 필요 없다(public repo clone 무인증, registry 무인증).
- task 이미지 태그의 source of truth가 git(DAG 파일)에서 Airflow Variable(`rondo_task_image_version`)로 이동 — DAG가 빌드 직후 즉시 bump, git
push/GitDagBundle 지연 없음.
- `deploy/release.sh`(issue #17)가 build 단계를 잃고 registry 태그 존재 확인 + daemon 사전검증(issue #15 순서 유지) + compose.yml bump+재시작만 남김.
- `deploy/build.sh` 주석 정정(release.sh가 더 이상 정본 빌드 경로가 아님).

## release.sh 사전검증 순서 수정 (2026-07-14)
- issue #15: 스모크가 태그 bump(compose.yml+DAG, 양쪽 repo)/daemon 재시작보다 늦게 실행되던 순서 버그 수정. 이제 daemon+task 이미지를 일회성 컨테이너로 먼저
검증하고, 통과한 뒤에만 태그 bump+재시작이 진행된다.
- task 이미지는 이전엔 검증 대상이 아니었다(스모크는 daemon 컨테이너 exec뿐) — import 스모크로 신규 편입.
- `docs/runbook.md` §2 "이미지 배포" 절을 실제 흐름과 일치하도록 갱신, GitDagBundle 60초 반영 지연 사실 추가.
- `deploy/build.sh`의 존재하지 않는 `promote.sh` 참조 주석 정정.

## 문서 로직 서술 재싱크 — spec/architecture/runbook (2026-07-14)
- spec.md §3 지표 레지스트리: `qwk`(TBD→구현됨, metric_class ordinal→classification), `balanced_accuracy`(BON-273) 신규 행 추가.
- spec.md §4 label 규칙: `z` 기본값 1.0→실제 `LABEL_Z=2.0`(BON-194) 정정. jump 판정이 harness 절대-마진의 잠정값이며 `cycle/run.py`가
`is_significant_gain`(paired per-fold t-test, BON-247/267)으로 재확정한다는 사실 반영. 완벽점수·회귀 trivial-baseline 누수 가드 2종과
`is_noop_tie`(BON-239) 추가.
- spec.md §5 / architecture.md §5 / runbook.md §6: "네트워크 sandbox 미구현" 서술이 stale — 프로덕션에서 `os.unshare(CLONE_NEWNET)`
egress 차단 + rlimit이 이미 구현됨(ADR-017)으로 정정, 파일시스템 sandbox만 미구현 유지. 코드생성 재생성 횟수도 1회→실제 2회로 정정.

## 문서·docstring 코드 싱크 정리 (2026-07-13)
- BON-240 반영: Coder 모델 문서 표기를 `qwen3.5:397b`(출력 토큰 과다) → `gpt-oss:120b`로 전 문서 정정(README, architecture, spec, setup,
decisions ADR-016 amend).
- `lb_score`/제출 추적 상태 정정: `raw.kaggle_submissions` 폴링(`/api/submissions/*`)이 이미 lb_score를 attempts까지 기록하는데 "미구현"으로 서술되어
있던 architecture.md·runbook.md 정정. `submission_budget` 일일 상한 enforcement만 여전히 미구현.
- spec.md 스키마 누락 보강: `raw.kaggle_submissions`(§1.11 신설), `holdout_cv_gap_trend` 뷰,
`raw.attempts.{holdout_score,confirm_seed_gains,fold_scores}`, `raw.pipelines.{pipeline_sha256,oof_preds}`,
`raw.super_cycle_context` PK가 `queue_id`→`run_id`(BON-237)로 변경된 사실, `/api/health`+`/api/submissions/*` 엔드포인트 5종.
- README 프로젝트 구조 트리를 실제 파일과 일치시킴 — 존재하지 않는 `cycle/super_cycle.py` 참조 제거, 신규 파일 6개(`bin/blend.py`,
`bin/export_results.py`, `bin/rebuild_best_pipeline.py`, `bin/seed_competition_data.py`, `cycle/promotion.py`,
`store/train_data.py`) 추가.
- 죽은 코드 삭제: `main.py`(uv-init 스텁, 미참조), `bin/run_cycle.py`(Phase-0 PoC, `cycle/run.py`+`run_cycle_task.py`로 대체됨).
- `api.md`(관측 API 설계, ~30 엔드포인트 대부분 미구현) — 코드가 아니라 gitignore된 미추적 로컬 스크래치 파일이었음이 드러나 GitHub Issue #11로 이관 후 삭제.

## 학습 신호 회복 — jump 라벨 붕괴 수정 + 코드생성 정적 가드 (2026-07-05)
- ADR-012 amend(BON-267): label의 jump 판정을 harness 절대-마진에서 promotion과 동일한 paired 유의성 검정(`is_significant_gain`)으로 통일. 전체
7447건 attempt 중 jump 0건이던 근본원인 수정 — bandit 보상·stagnation 감지·reflection 게이트가 전부 이 label에 의존해 함께 정상화됨.
- BON-268: `evaluator/contract.py`의 `validate_patch`에 정적 검사 2종 추가 — pandas-only
API(`.groupby`/`.map_dict`/`.take`/`.apply`/`.iterrows`/`.applymap`/`.get_dummies`) 금지, candidate patch 자신의
undefined-name 검사(실행 격리 모델과 일치하는 범위로 검증). `agents/coder.py` contract 프롬프트에도 동일 금지 목록 반영.
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
- ADR-008 개정: 임베딩 `nomic-embed-text`(768d) → `qwen3-embedding:0.6b`(1024d, MRL). 2026 MTEB v2 오픈웨이트 최상위. 스키마
`reflections.embedding` → `float[1024]`.
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
