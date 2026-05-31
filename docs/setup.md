# 초기 셋업

로컬 1대 기준. 모든 자격증명은 `.env` / 로컬 secrets에만 둔다 (repo 평문 금지).

## 1. 사전 요구
- Python 3.10+
- 로컬 Ollama (임베딩용) — `http://localhost:11434`
- 격리 런타임: 컨테이너(예: Docker) 또는 nsjail (ADR-013, 구체 선정 TBD)
- Kaggle 계정 + API 토큰
- Ollama Cloud 계정 (Pro 시작값)

## 2. 자격증명 (.env)

```dotenv
OLLAMA_API_KEY=<your-ollama-cloud-key>
KAGGLE_USERNAME=<your-kaggle-username>
KAGGLE_KEY=<your-kaggle-key>
```

`.env`는 `.gitignore`에 포함. Kaggle은 `~/.kaggle/kaggle.json`도 허용.

## 3. 로컬 모델

```bash
ollama pull qwen3-embedding:0.6b   # 임베딩 (로컬, 1024d, ADR-008)
# Strategist/Coder/Reflector 모델은 Ollama Cloud (모델 배정 ADR-016)
```

## 4. 데이터 스토어 초기화
- DuckDB(단일 스토어): `store/schema.sql` 적용 (competitions / attempts / reflections[embedding 컬럼 포함] / pipelines / submission_budget)
- dbt(dbt-duckdb): `dbt deps && dbt run` 으로 staging + marts 빌드
- 별도 벡터DB 불필요 (검색은 DuckDB 벡터 컬럼 브루트포스, ADR-007)

## 5. 동작 확인 (Phase 0)
- `bin/start_competition.py`로 대회 1개 등록 → fingerprint insert
- `bin/run_cycle.py` 1회 실행 → `raw.attempts`에 row 1개 + CV 기록 확인

> 미구현: 대부분 Phase별로 채워진다(`tasks.md`). 이 문서는 환경/자격증명 셋업 기준만 정의.
