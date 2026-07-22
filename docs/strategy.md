# 정형 데이터 경진대회 일반 전략 노트

> 설계 문서(`architecture.md`)와 분리된 실전 팁 모음. 시스템 설계가 아니라 "어떤 가설을 던지면 좋은가"의 baseline 지식.
> Strategist 프롬프트의 시스템 컨텍스트 / Reflector의 검토 체크리스트로도 활용.

정형 데이터 대회 상위권(Silver~Gold) 진입 전략의 핵심: *"어떻게 가장 효율적으로, 가장 신뢰할 수 있는 실험을 많이 할 것인가."*

---

## 1. 툴 스택 (2026 Toolbelt)

| 단계 | 필수 툴 | 이유 |
|---|---|---|
| 데이터 처리 | Polars | Pandas보다 빠름, 적은 램으로 대용량 처리 가능 |
| 모델 (GBDT) | LightGBM, CatBoost | 속도는 LightGBM, 범주형/성능은 CatBoost |
| 모델 (SOTA) | TabPFN 3 | 작은 데이터셋에서 튜닝 없이 강력한 베이스라인 |
| 하이퍼파라미터 | Optuna | 자동 탐색. 수동 튜닝 지양 |
| 브레인스토밍 | LLM | 파생 변수 아이디어, 코드 초안 |
| 코드 관리 | Kaggle Notebooks | 무료 GPU/TPU + 데이터셋 연결 편함 |

---

## 2. 접근 방법 (Winning Workflow)

### 2.1 검증 전략 (가장 중요)

리더보드에서 떨어지는 가장 흔한 원인은 검증 전략 실패.

- **K-Fold**: 데이터를 K개로 나눠 번갈아 검증.
- **Stratified**: 타깃 비율을 fold마다 유지(분류).
- **Group**: 같은 환자/날짜 등 그룹이 train/val에 섞이지 않게 분리.
- **Time-aware**: 시간 누수 방지(시계열).

### 2.2 피처 엔지니어링 (Secret Sauce)

모델 가중치보다 "무엇을 주느냐"가 점수를 결정.

- **Target Encoding**: 타깃의 평균을 변수로. 누수 방지(폴드 내 계산) 필수.
- **Interaction Features**: 두 변수의 곱/비 (예: `키 / 몸무게 = BMI`).
- **LLM-assisted**: 도메인 설명을 주고 파생 변수 10개 요청 → 코드로 구현.
- **Binning / Frequency encoding / Aggregations** (group by 기반).

### 2.3 앙상블 / 스태킹

- **Weighted Averaging**: 성능 다른 모델 결과에 가중치.
- **Stacking**: 1단계 예측을 2단계 모델(주로 Ridge/Lasso) 입력으로.

### 2.4 실험 자동화 (Agentic Workflow)

하나의 파이프라인을 짜두고, 하이퍼파라미터·변수 조합을 자동으로 수백 번 돌린 뒤 베스트만 취합. (본 시스템 `architecture.md`의 Reflexion 루프가 이 단계에 해당.)

---

## 3. 핵심 마인드셋: "CV를 믿어라"

```
Strong CV + Correlation with Public LB = Gold Medal
```

- **CV**: 로컬 테스트 점수.
- **Public LB**: 대회 중 실시간 점수 (공개 일부 데이터).
- **결론**: Public LB가 낮아도 CV가 꾸준히 올라가면 그 방향이 맞다. 등수에 일희일비 금지.

→ 본 시스템의 ADR-003 (CV 주 신호, LB 확인용)과 일치.

---

## 4. 입문 로드맵

1. 지난 대회(Closed) 중 정형 대회 하나 골라 상위권 Notebook을 Copy & Edit으로 돌려본다.
2. Polars 익히기: Pandas 코드를 Polars로 변환 연습. 처리 시간 10배 단축.
3. 본인만의 견고한 K-Fold 검증 코드를 만들어둔다. 평생 쓴다.

---

## 5. 본 시스템에서의 활용

- **action_type 배정**: 정상 사이클은 밴딧(`cycle/action_optimizer.py`) posterior가 Strategist 프롬프트에 advisory hint로만 주입되고 최종 결정은 LLM 자유(ADR-020). `super_cycle`은 `assign_super_cycle_actions`가 attempt별로 강제 배정. §2의 액션 카탈로그는 두 경로 모두에서 Strategist가 참고하는 선택지 설명으로 쓰인다.
- **Reflector 체크리스트**: §2.1 검증 누수, §2.2 피처 누수, §3 CV-LB 상관 — 매 reflection마다 점검.
- **Bootstrap 시드 후보**: §1의 LightGBM/CatBoost + §2.1 Stratified K-Fold + 기본 target encoding을 "도메인 무관 안전 베이스라인"으로 고정.
