#!/usr/bin/env bash
# ops-vm daemon 시작 래퍼 — sops 복호화 후 compose up
# repo의 deploy/compose.yml에서 직접 기동 (compose project=deploy, release.sh와 동일)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENC_ENV="$REPO_DIR/secrets/rondo.enc.env"
PLAIN_ENV=/var/lib/rondo/.env

if [[ ! -f "$ENC_ENV" ]]; then
    echo "[start] $ENC_ENV 없음. secrets/rondo.enc.env를 먼저 생성하세요."
    exit 1
fi

sops --decrypt "$ENC_ENV" > "$PLAIN_ENV"
chmod 600 "$PLAIN_ENV"
echo "[start] .env 복호화 완료"

docker compose -f "$REPO_DIR/deploy/compose.yml" up -d
echo "[start] rondo-daemon 시작 완료 (compose project: deploy)"
