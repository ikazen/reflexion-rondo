# 초기 셋업

Mac(M1 Pro) = 로컬 임베딩 Ollama 서버, worker/WSL2 = 실행 환경, ops-vm Postgres = 저장소 기준.

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
# 임베딩 (필수, 로컬 Mac Ollama)
ollama pull qwen3-embedding:8b
```

> 현재 코드의 기본 역할 모델은 Strategist=`glm-5.2`, Reflector=`kimi-k2.6`,
> Coder=`qwen3-coder-next`, Embedding=`qwen3-embedding:8b`다.
> Strategist/Reflector/Coder는 Ollama Cloud를 사용하고, 임베딩만 `OLLAMA_BASE_URL`의
> 로컬 Ollama 서버를 사용한다.
>
> 모델 태그는 바뀔 수 있다. 설치 전 [ollama.com/library](https://ollama.com/library) 에서 현재 태그 확인.
> 모델 ID를 바꾸려면 현재는 `config/settings.py`의 상수를 수정해야 한다.

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

# Ollama Cloud
OLLAMA_CLOUD_BASE_URL=https://ollama.com
OLLAMA_API_KEY=<your-ollama-cloud-key>

# 참고: config/settings.py의 모델 기본값이 단일 소스다.
# MODEL_STRATEGIST / MODEL_REFLECTOR / MODEL_CODER / MODEL_EMBEDDING 환경변수로 override 가능하지만 실험용으로만 쓰고 SOPS·Airflow Variable엔 넣지 않는다.
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
kaggle competitions download -c playground-series-s4e1
unzip "*.zip"
```

---

## 8. Postgres 초기화 확인

`RONDO_DB_URL` 미설정 시 기본값은 `postgresql://rondo:rondo@localhost:5432/rondo`다.

```bash
uv run python - <<'EOF'
from store.db import connect
conn = connect()
print(conn.execute("""
    select table_name
    from information_schema.tables
    where table_schema = 'raw'
    order by table_name
""").fetchall())
conn.close()
EOF
```

`raw.competitions`, `raw.attempts`, `raw.reflections` 등이 보이면 정상.

---

## 9. SOPS + age secrets (ops-vm 배포용)

ops-vm에 daemon을 배포할 때는 평문 `.env` 대신 암호화된 `secrets/rondo.enc.env`를 git에 보관한다.

### 최초 키 생성 (ops-vm)
```bash
mkdir -p ~/.config/sops/age
age-keygen -o ~/.config/sops/age/keys.txt
# 출력된 공개키를 .sops.yaml의 age: 필드에 붙여넣는다
# 비밀키 파일 내용을 Bitwarden에 백업한다
```

### secrets 암호화
```bash
cp secrets/rondo.env.template secrets/rondo.env
# 실제 값을 채운다
sops --encrypt secrets/rondo.env > secrets/rondo.enc.env
rm secrets/rondo.env          # 평문은 즉시 삭제
git add secrets/rondo.enc.env .sops.yaml
git commit -m "chore: update encrypted env"
```

### 복호화 확인
```bash
sops --decrypt secrets/rondo.enc.env | grep RONDO_DB_URL
```

### 다른 머신에서 복원
Bitwarden에서 age 비밀키를 꺼내 `~/.config/sops/age/keys.txt`에 저장하면 동일하게 복호화된다.

---

## 10. 한 사이클 실행 (연결 전체 테스트)

대회를 먼저 등록한 뒤 사이클을 돌린다.

```bash
uv run python bin/start_competition.py \
    --id playground-series-s4e1 \
    --name "Bank Customer Churn Prediction" \
    --task binary --metric auc --target Exited
```

이후 운영 daemon을 실행하거나, 로컬 smoke/test 용도로 단일 사이클을 실행한다.

```bash
# 운영 daemon. AIRFLOW_URL이 있으면 Airflow super-cycle을 트리거한다.
uv run python -m bin.run_daemon

# daemon 없이 수동 단일 사이클 실행
uv run python -m bin.run_reflexion --competition s4e1 --stage bootstrap --cycles 1
```
