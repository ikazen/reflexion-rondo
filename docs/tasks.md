# 작업 단계·진행 상태

현재 상태: **설계 단계.** 코드 없음. `init.md` 설계서를 표준 문서 구조로 정리 완료.

각 phase는 독립적으로 동작 가능한 단위. 앞 phase가 돌아야 다음 phase 의미 있음.

## Phase 0 — PoC (배관만) · 완료
시작 대회 `playground-series-s4e1` (Bank Churn, 이진/AUC). closed라 late submission으로 LB 채점.
LightGBM 5-fold + DuckDB 로깅 + 수동 제출. LLM/검색/transfer 모두 OFF. 한 사이클이 도는지만 확인.
- [x] 시작 대회 선정 → `playground-series-s4e1`
- [x] Kaggle API 다운로드 + Evaluator k-fold 하니스 (CV AUC 0.88885, fold std 0.00175)
- [x] `raw.competitions`/`raw.attempts` 스키마 + 단일 스토어 기록 골격
- [x] 수동 제출 1회로 CV-LB 연결 확인 (CV 0.88885 / LB public 0.88668 / private 0.88821)

전이 코퍼스 후보(Phase 3, 착수 시 지표 재확인): AUC 클러스터 `s4e1`/`s5e3`/`s5e8`, 회귀 `s4e4`/`s5e5`/`s5e9`, accuracy `s4e11`/`s5e7`, 다중분류 `s4e2`.

## Phase 1 — Fingerprint + 베이스라인 자동화 · 완료
같은 대회에 여러 attempt를 자동 누적.
- [x] `store/fingerprint.py` 결정적 메타피처 계산기 (14개 메타피처, s4e1: mid/21.2% minority)
- [x] `score_progression` DuckDB 뷰 (dbt는 Phase 3 마트 3개 이상 시 도입)
- [x] Evaluator `label`/`gain_vs_best` 결정 규칙 (Phase 0에서 구현, fold_std 기반)
- [x] `runtime/` 스캐폴드 (Phase 2에서 컨테이너/nsjail 구현)

## Phase 2 — Reflexion (한 대회) · 미착수
Strategist + Reflector + DuckDB 벡터 검색 + reflexion 1변경 규율.
- [ ] Coder 컨트랙트(`feature_fn`/`model_fn`) + 실행 전 검증 게이트
- [ ] `reflections.embedding` 컬럼 + 브루트포스 코사인 검색 + 메타필터
- [ ] `reflection_impact` 마트
- [ ] Strategist의 "실제 채택 교훈 id" 출력

## Phase 3 — Transfer (두 번째 대회) · 미착수
cold-start 절차 가동. `cold_start_progression`으로 효과 측정. 효과 없으면 fingerprint/generality 튜닝.
- [ ] `memory/transfer.py` 유사 대회 검색 (가중 거리)
- [ ] cold-start 시드 큐잉 (`raw.pipelines` 재사용)
- [ ] `cold_start_progression` 마트 + `warm_start_ratio` 추세 확인

## Phase 4 — 메타 루프 · 미착수
- [ ] `reflection_impact`로 저효율 교훈 archive
- [ ] 검색 재순위 가중치 적용
- [ ] RAG 품질 점검

## 후속 과제 (조건부)
- ablation: `reflection_impact` 신호가 모호하면 retrieval ON/OFF 도입 (ADR-015)
- label `z`·fingerprint 거리 가중치 캘리브레이션 (대회 누적 후)
- 워커 ≥ 3 시 Prefect 승격 검토
