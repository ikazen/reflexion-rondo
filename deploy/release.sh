#!/usr/bin/env bash
# WSL에서 실행: bash deploy/release.sh <version>
#
# 흐름: 가드 → ops-vm 빌드+push → compose/DAG 태그 bump → ops-vm 재시작 → 스모크 → 사후 확인
#
# 전제:
#   - WSL에 ~/projects/reflexion-rondo, ~/projects/airflow-stack checkout
#   - ops-vm에 ~/projects/reflexion-rondo checkout, /var/lib/rondo/.env 존재
#   - ops-vm docker: registry.internal:5000 insecure-registries 등록
set -euo pipefail

VERSION=${1:?"Usage: bash deploy/release.sh <version>  (e.g. v1.2.0)"}

REGISTRY=registry.internal:5000
DAEMON_IMG=$REGISTRY/reflexion-rondo/daemon:$VERSION
TASK_IMG=$REGISTRY/reflexion-rondo/task:$VERSION

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
AIRFLOW_STACK_DIR=$(cd "$REPO_DIR/../airflow-stack" && pwd)

# ---- 1. 가드 ----------------------------------------------------------------

echo "[release] $VERSION"

BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "main" ]]; then
    echo "ERROR: current branch is '$BRANCH' — release from main only"
    exit 1
fi
if ! git -C "$REPO_DIR" diff --quiet || ! git -C "$REPO_DIR" diff --cached --quiet; then
    echo "ERROR: working tree is dirty — commit or stash first"
    exit 1
fi
git -C "$REPO_DIR" fetch --quiet
LOCAL=$(git -C "$REPO_DIR" rev-parse HEAD)
REMOTE=$(git -C "$REPO_DIR" rev-parse origin/main)
if [[ "$LOCAL" != "$REMOTE" ]]; then
    echo "ERROR: local main is not in sync with origin/main — pull first"
    exit 1
fi

# ---- 2. 빌드 + push (ops-vm) ------------------------------------------------

echo "[release] building on ops-vm ..."
ssh ops-vm "
    set -euo pipefail
    cd ~/projects/reflexion-rondo
    git pull --quiet
    docker build -q -f deploy/Dockerfile      -t $DAEMON_IMG .
    docker build -q -f deploy/Dockerfile.task -t $TASK_IMG  .
    docker push $DAEMON_IMG
    docker push $TASK_IMG
"
echo "[release] pushed $DAEMON_IMG"
echo "[release] pushed $TASK_IMG"

# ---- 3. 태그 bump + push (WSL) ----------------------------------------------

echo "[release] updating deploy/compose.yml ..."
sed -i "s|/daemon:[^ '\"]*|/daemon:$VERSION|g" "$REPO_DIR/deploy/compose.yml"
if ! git -C "$REPO_DIR" diff --quiet deploy/compose.yml; then
    git -C "$REPO_DIR" commit -m "chore: daemon image -> $VERSION" deploy/compose.yml
fi
git -C "$REPO_DIR" push --quiet

echo "[release] updating airflow-stack DAG ..."
sed -i "s|/task:[^'\"]*|/task:$VERSION|g" "$AIRFLOW_STACK_DIR/dags/reflexion_rondo_cycle.py"
if ! git -C "$AIRFLOW_STACK_DIR" diff --quiet dags/reflexion_rondo_cycle.py; then
    git -C "$AIRFLOW_STACK_DIR" commit -m "chore: task image -> $VERSION" \
        dags/reflexion_rondo_cycle.py
fi
git -C "$AIRFLOW_STACK_DIR" push --quiet

# ---- 4. 재시작 (ops-vm) -----------------------------------------------------

echo "[release] restarting daemon ..."
ssh ops-vm "
    cd ~/projects/reflexion-rondo
    git pull --quiet
    docker compose -f deploy/compose.yml up -d
"

# ---- 5. 스모크 (실행 중인 daemon 컨테이너 안에서) ---------------------------
# docker exec으로 실행해야 compose.yml 환경(AIRFLOW_URL 등)과 nexus 네트워크를 그대로 사용.
# ollama_local은 컨테이너 내 Tailscale 미지원으로 항상 접근 불가 — 명시적으로 skip.

echo "[release] smoke test (exec into running daemon) ..."
sleep 5
ssh ops-vm "
    CONTAINER=\$(docker compose -f ~/projects/reflexion-rondo/deploy/compose.yml ps -q rondo-daemon)
    docker exec \"\$CONTAINER\" \
        uv run --no-sync python -m bin.healthcheck --skip ollama_local
"

# ---- 6. 사후 확인 ------------------------------------------------------------

echo "[release] $VERSION deployed successfully"
