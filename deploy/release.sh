#!/usr/bin/env bash
# WSL에서 실행: bash deploy/release.sh <version>
#
# 흐름: 가드 → ops-vm 빌드+push → 스모크 → compose/DAG 태그 bump → ops-vm 재시작 → 사후 확인
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

# ---- 3. 스모크 (새 이미지 컨테이너 안에서) ----------------------------------

echo "[release] smoke test ..."
ssh ops-vm "
    docker run --rm \
        --env-file /var/lib/rondo/.env \
        --network nexus \
        $DAEMON_IMG \
        uv run --no-sync python -m bin.healthcheck
"

# ---- 4. 태그 bump + push (WSL) ----------------------------------------------

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

# ---- 5. 재시작 (ops-vm) -----------------------------------------------------

echo "[release] restarting daemon ..."
ssh ops-vm "
    cd ~/projects/reflexion-rondo
    git pull --quiet
    docker compose -f deploy/compose.yml up -d
"

# ---- 6. 사후 확인 ------------------------------------------------------------

echo "[release] waiting for daemon ..."
for i in $(seq 1 12); do
    if ssh ops-vm "curl -sf http://localhost:8000/api/heartbeat" >/dev/null 2>&1; then
        echo "[release] $VERSION deployed successfully"
        exit 0
    fi
    sleep 5
done
echo "WARNING: daemon did not respond within 60s — check logs:"
echo "  ssh ops-vm 'docker compose -f ~/projects/reflexion-rondo/deploy/compose.yml logs --tail=30'"
exit 1
