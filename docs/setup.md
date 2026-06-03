# 초기 셋업

Mac(M1 Pro) = Ollama 추론 서버, WSL2 = 저장·실행 환경 기준.

---

## 1. Mac — Ollama 외부 접속 허용

Ollama는 기본적으로 `localhost`만 수신한다. WSL2에서 접속하려면 `0.0.0.0`으로 변경해야 한다.

```bash
# Mac 터미널에서 실행
launchctl setenv OLLAMA_HOST "0.0.0.0"
```

그 다음 메뉴바 Ollama 아이콘 → Quit → 다시 실행.

확인:
```bash
# Mac에서
curl http://localhost:11434/api/tags
# JSON 응답이 오면 정상
```

---

## 2. Mac — 모델 저장 경로 변경 (선택)

기본 경로는 `~/.ollama/models`. 변경하려면 **절대경로**로 지정해야 한다 (`~` 확장 안 됨).

```bash
# Mac 터미널 — 디렉토리 먼저 생성
mkdir -p /Users/<your-username>/mnt/ollama-model

# 경로 등록 후 Ollama 재시작
launchctl setenv OLLAMA_MODELS "/Users/<your-username>/mnt/ollama-model"
# 메뉴바 → Quit → 다시 실행
```

---

## 3. Mac — 모델 pull

```bash
# 임베딩 (필수)
ollama pull qwen3-embedding:0.6b

# Strategist / Reflector (테스트용 기본값)
ollama pull qwen3.5:14b

# Coder (테스트용 기본값)
ollama pull devstral-small-2
```

> 모델 태그는 바뀔 수 있다. 설치 전 [ollama.com/search](https://ollama.com/search) 에서 현재 태그 확인.
> 프로덕션 모델(deepseek-v4-pro / glm-5 / qwen3-coder-next)은 Ollama Cloud 전환 시 `.env`만 수정하면 된다 (ADR-004).

---

## 4. WSL2 — 호스트명 설정 (Tailscale)

모든 인스턴스가 Tailscale 아래에 있으므로 Tailscale IP 또는 MagicDNS 호스트명을 사용한다.

```bash
# MagicDNS resolve 확인 (WSL2에서)
ping mac-server.<tailnet>.ts.net
```

resolve 되면 MagicDNS 호스트명을 사용한다. 안 되면 Tailscale IP(`tailscale status`에서 확인)를 사용한다. Tailscale IP는 `100.x.x.x` 대역으로 네트워크가 바뀌어도 고정된다.

---

## 5. WSL2 — .env 작성

프로젝트 루트에 `.env` 파일 생성:

```dotenv
# MagicDNS 사용 시
OLLAMA_BASE_URL=http://mac-server.<tailnet>.ts.net:11434
# 또는 Tailscale IP 직접 사용
# OLLAMA_BASE_URL=http://100.x.x.x:11434

# 모델 (기본값과 동일하면 생략 가능)
MODEL_STRATEGIST=qwen3.5:14b
MODEL_REFLECTOR=qwen3.5:14b
MODEL_CODER=devstral-small-2
MODEL_EMBEDDING=qwen3-embedding:0.6b
```

`.env`는 `.gitignore`에 포함되어 있으므로 커밋되지 않는다. 실제 tailnet 이름과 IP는 `.env`에만 보관한다.

---

## 6. WSL2 — 연결 확인

```bash
# .env의 OLLAMA_BASE_URL로 직접 테스트
source .env
curl ${OLLAMA_BASE_URL}/api/tags
```

JSON 응답에 pull한 모델들이 보이면 정상.

임베딩 동작 확인:
```bash
uv run python - <<'EOF'
from config.settings import OLLAMA_BASE_URL, MODEL_EMBEDDING
from ollama import Client
c = Client(host=OLLAMA_BASE_URL)
r = c.embed(model=MODEL_EMBEDDING, input="test")
print(f"embedding dim: {len(r.embeddings[0])}")  # 1024 이어야 함
EOF
```

---

## 7. WSL2 — Kaggle 데이터 세팅

```bash
# ~/.kaggle/kaggle.json 없으면 먼저 발급 (kaggle.com → Account → API)
mkdir -p data/playground-series-s4e1
cd data/playground-series-s4e1
kaggle competitions download -c playground-series-s4e4
unzip "*.zip"
```

---

## 8. DuckDB 초기화 확인

```bash
uv run python - <<'EOF'
from store.db import connect
conn = connect()
print(conn.execute("show tables").fetchall())
EOF
```

`raw.competitions`, `raw.attempts`, `raw.reflections` 등이 보이면 정상.

---

## 9. 한 사이클 실행 (연결 전체 테스트)

대회를 먼저 등록한 뒤 사이클을 돌린다.

```bash
uv run python bin/start_competition.py
```

이후 `cycle/run.py`를 직접 호출하는 스크립트를 만들거나 `bin/run_cycle.py`를 업데이트해서 사용한다 (현재 `bin/run_cycle.py`는 Phase 0 PoC — LLM 없이 고정 코드 실행).
