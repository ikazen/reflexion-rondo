#!/usr/bin/env bash
# worker-vm 최초 1회 설치 — deploy/ 디렉토리 안에서 실행
set -euo pipefail

# 1. registry.internal insecure-registries 등록
if ! sudo grep -q registry.internal /etc/docker/daemon.json 2>/dev/null; then
    echo '{"insecure-registries": ["registry.internal"]}' | sudo tee /etc/docker/daemon.json
    sudo systemctl restart docker
    echo "[install] docker restarted with insecure-registries"
fi

# 2. 데이터 디렉토리 생성
sudo mkdir -p /var/lib/rondo/data
sudo mkdir -p /tmp/rondo-eval
echo "[install] directories created"

# 3. .env 파일 확인
if [[ ! -f /var/lib/rondo/.env ]]; then
    echo "[install] WARNING: /var/lib/rondo/.env 없음. 배포 전 작성 필요."
    echo "  필수 키: OLLAMA_BASE_URL, OLLAMA_CLOUD_BASE_URL, OLLAMA_API_KEY,"
    echo "           MODEL_STRATEGIST, MODEL_REFLECTOR, MODEL_CODER, MODEL_EMBEDDING"
fi

# 4. Tailscale (이미 가입된 경우 건너뜀)
if ! command -v tailscale &>/dev/null || ! tailscale status &>/dev/null; then
    echo "[install] Tailscale 미가입 — 수동으로 실행 필요:"
    echo "  sudo tailscale up --ssh"
    echo "  (가입 후 admin 콘솔에서 승인)"
fi

# 5. eval 이미지 pull (daemon 첫 실행 전 캐싱)
docker pull registry.internal/reflexion-eval:latest || \
    echo "[install] WARNING: eval 이미지 pull 실패 — 첫 사이클에서 자동 시도"

# 6. compose.yml + systemd unit 설치
sudo mkdir -p /opt/rondo
sudo cp compose.yml /opt/rondo/compose.yml
sudo cp rondo-daemon.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rondo-daemon

echo "[install] rondo-daemon 시작 완료"
echo "  상태: sudo systemctl status rondo-daemon"
echo "  로그: sudo journalctl -u rondo-daemon -f"
