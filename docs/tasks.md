# 작업 단계·진행 상태

현재 상태: **설계 단계.** 코드 없음. `init.md` 설계서를 표준 문서 구조로 정리 완료.

각 phase는 독립적으로 동작 가능한 단위. 앞 phase가 돌아야 다음 phase 의미 있음.

## Phase 0 — PoC (배관만) · 미착수
선택한 대회 1개. LightGBM 5-fold + DuckDB 로깅 + 수동 제출. LLM/검색/transfer 모두 OFF.
한 사이클이 도는지만 확인.
- [ ] 시작 대회 선정 (열린 Playground Series 1개)
- [ ] Kaggle API 다운로드 + Evaluator k-fold 하니스
- [ ] `raw.competitions`/`raw.attempts` 스키마 + 단일 스토어 기록 골격
- [ ] 수동 제출 1회로 CV-LB 연결 확인

## Phase 1 — Fingerprint + 베이스라인 자동화 · 미착수
같은 대회에 여러 attempt를 자동 누적.
- [ ] `store/fingerprint.py` 결정적 메타피처 계산기
- [ ] `score_progression` 마트
- [ ] Evaluator `label`/`gain_vs_best` 결정 규칙 (spec §4)
- [ ] 생성 코드 격리 런타임 (`runtime/`, 컨테이너 vs nsjail 선정)

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
