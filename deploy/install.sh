#!/usr/bin/env bash
# ops-vm 최초 1회 설치 — deploy/ 디렉토리 안에서 실행
set -euo pipefail

# 1. registry.internal insecure-registries 등록
if ! sudo grep -q registry.internal /etc/docker/daemon.json 2>/dev/null; then
    echo '{"insecure-registries": ["registry.internal:80"]}' | sudo tee /etc/docker/daemon.json
    sudo systemctl restart docker
    echo "[install] docker restarted with insecure-registries"
fi

# 2. 데이터 디렉토리 생성
sudo mkdir -p /var/lib/rondo/data
sudo mkdir -p /tmp/rondo-eval
echo "[install] directories created"

# 3. age + sops 설치
if ! command -v age &>/dev/null; then
    AGE_VER=1.2.1
    curl -fsSL "https://github.com/FiloSottile/age/releases/download/v${AGE_VER}/age-v${AGE_VER}-linux-amd64.tar.gz" \
        | sudo tar -xz -C /usr/local/bin --strip-components=1 age/age age/age-keygen
    echo "[install] age 설치 완료"
fi
if ! command -v sops &>/dev/null; then
    SOPS_VER=3.9.4
    sudo curl -fsSL "https://github.com/getsops/sops/releases/download/v${SOPS_VER}/sops-v${SOPS_VER}.linux.amd64" \
        -o /usr/local/bin/sops && sudo chmod +x /usr/local/bin/sops
    echo "[install] sops 설치 완료"
fi

# 4. age 키 안내 (최초 설치 시)
AGE_KEY_DIR="$HOME/.config/sops/age"
if [[ ! -f "$AGE_KEY_DIR/keys.txt" ]]; then
    mkdir -p "$AGE_KEY_DIR"
    echo "[install] age 키가 없습니다. 새로 생성하거나 Bitwarden에서 복원하세요:"
    echo "  age-keygen -o $AGE_KEY_DIR/keys.txt   # 신규 생성"
    echo "  # 공개키를 .sops.yaml의 age: 필드에 등록하고 secrets/rondo.enc.env 재암호화 필요"
fi

# 5. .env 복호화 (sops)
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$REPO_DIR/secrets/rondo.enc.env" ]]; then
    sops --decrypt "$REPO_DIR/secrets/rondo.enc.env" | sudo tee /var/lib/rondo/.env > /dev/null
    sudo chmod 600 /var/lib/rondo/.env
    echo "[install] .env 복호화 완료"
else
    echo "[install] WARNING: secrets/rondo.enc.env 없음. docs/setup.md §SOPS 참조."
fi

# 6. Tailscale (이미 가입된 경우 건너뜀)
if ! command -v tailscale &>/dev/null || ! tailscale status &>/dev/null; then
    echo "[install] Tailscale 미가입 — 수동으로 실행 필요:"
    echo "  sudo tailscale up --ssh"
    echo "  (가입 후 admin 콘솔에서 승인)"
fi

# 7. task 이미지 pull (Airflow DockerOperator 첫 실행 전 캐싱)
docker pull registry.internal:80/reflexion-rondo/task:latest || \
    echo "[install] WARNING: task 이미지 pull 실패 — 첫 사이클에서 자동 시도"

# 8. compose.yml 설치 및 시작 (Docker restart: always 로 재부팅 생존)
sudo mkdir -p /opt/rondo
sudo cp compose.yml /opt/rondo/compose.yml
cd /opt/rondo && sudo docker compose up -d

echo "[install] rondo-daemon 시작 완료"
echo "  상태: docker compose -f /opt/rondo/compose.yml ps"
echo "  로그: docker compose -f /opt/rondo/compose.yml logs -f"
