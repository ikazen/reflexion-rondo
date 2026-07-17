#!/usr/bin/env bash
# WSL에서 실행: bash deploy/release.sh <version>
#
# daemon 전용 배포 스크립트(issue #17). daemon+task 이미지 빌드는 airflow-stack의
# `reflexion_rondo_deploy` DAG(ops 큐 docker.sock 재사용, airflow-stack#2/#3)가 담당한다 —
# Airflow UI에서 {"tag": "vX.Y.Z"}로 트리거하면 두 이미지를 빌드+push하고 task 쪽은
# `rondo_task_image_version` Variable을 즉시 bump한다(git push 불필요).
#
# 이 스크립트는 그 DAG가 이미 registry에 올려둔 태그를 받아 daemon만 실제로 컷오버한다:
# 가드 → registry에 태그 존재 확인 → daemon 사전검증(일회성 컨테이너) → compose.yml
# 태그 bump+push → 재시작 → heartbeat 확인.
#
# 전제:
#   - WSL에 ~/projects/reflexion-rondo checkout
#   - Airflow UI에서 reflexion_rondo_deploy DAG를 이 버전으로 먼저 트리거해 완료했을 것
#   - ops-vm에 /var/lib/rondo/.env 존재
set -euo pipefail

VERSION=${1:?"Usage: bash deploy/release.sh <version>  (e.g. v1.2.0) — reflexion_rondo_deploy DAG로 먼저 빌드했을 것"}

REGISTRY=registry.internal:5000
DAEMON_IMG=$REGISTRY/reflexion-rondo/daemon:$VERSION

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

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

# ---- 2. registry에 태그 존재 확인 --------------------------------------------
# 빌드는 이제 이 스크립트의 일이 아니다 — reflexion_rondo_deploy DAG가 이미 만들어뒀어야 한다.
#
# Accept 헤더에 OCI 매니페스트 타입도 포함 — lib/image_deploy.py의 build_and_push가
# 최신 buildx/BuildKit으로 OCI 포맷(application/vnd.oci.image.manifest.v1+json)을
# 기본 출력하면서, 구 Docker 전용 media type만 요청하던 이 체크가 실제로 존재하는
# 이미지도 404로 오판했다(v1.2.27 배포 시 실측). registry에 태그는 있는데 이 체크만
# 실패하는 상황이면 media type 협상 문제를 우선 의심할 것.
echo "[release] checking $DAEMON_IMG exists in registry ..."
STATUS=$(curl -o /dev/null -sw "%{http_code}" \
    "http://${REGISTRY}/v2/${DAEMON_IMG#${REGISTRY}/}/manifests/${VERSION}" \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.manifest.v1+json,application/vnd.oci.image.index.v1+json" 2>/dev/null || true)
if [[ "$STATUS" != "200" ]]; then
    echo "ERROR: $DAEMON_IMG not found in registry (http $STATUS)"
    echo "       Airflow UI에서 reflexion_rondo_deploy DAG를 {\"tag\": \"$VERSION\"}로 먼저 트리거하세요"
    exit 1
fi

# ---- 3. 사전검증 (일회성 컨테이너, ops-vm) -----------------------------------
# reflexion_rondo_deploy DAG가 빌드 직후 이미 한 번 검증했지만(build 시점), 여기서는
# 컷오버 시점 기준으로 다시 확인한다 — DB/Ollama 등 외부 의존성이 빌드 이후 바뀌었을 수 있다.

echo "[release] pre-flight: daemon healthcheck (throwaway container) ..."
ssh ops-vm "
    docker run --rm --network nexus --env-file /var/lib/rondo/.env \
        -e AIRFLOW_URL='http://airflow-api-server-1:8080' \
        $DAEMON_IMG uv run --no-sync python -m bin.healthcheck --skip ollama_local
"

# ---- 4. 태그 bump + push (WSL) ----------------------------------------------
# 사전검증을 통과한 뒤에만 실행 — 여기부터는 실제 배포로 간주한다.

echo "[release] updating deploy/compose.yml ..."
sed -i "s|/daemon:[^ '\"]*|/daemon:$VERSION|g" "$REPO_DIR/deploy/compose.yml"
if ! git -C "$REPO_DIR" diff --quiet deploy/compose.yml; then
    git -C "$REPO_DIR" commit -m "chore: daemon image -> $VERSION" deploy/compose.yml
fi
git -C "$REPO_DIR" push --quiet

# ---- 5. 재시작 (ops-vm) -----------------------------------------------------

echo "[release] restarting daemon ..."
ssh ops-vm "
    cd ~/projects/reflexion-rondo
    git pull --quiet
    docker compose -f deploy/compose.yml up -d
"

# ---- 6. 재시작 후 확인 -------------------------------------------------------
# daemon 이미지 자체는 이미 3에서 검증했으므로 여기서는 컴포즈 배선(volume/env/network)
# 문제만 잡는 가벼운 heartbeat 확인으로 충분하다.

echo "[release] post-restart heartbeat check ..."
sleep 5
ssh ops-vm "curl -sf http://localhost:8000/api/heartbeat > /dev/null"

echo "[release] $VERSION deployed successfully (daemon only — task image는 reflexion_rondo_deploy DAG로 이미 반영됨)"
