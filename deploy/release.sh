#!/usr/bin/env bash
# WSL에서 실행: bash deploy/release.sh <version>
#
# 흐름: 가드 → ops-vm 빌드+push → 사전검증(daemon+task, 일회성 컨테이너) → compose/DAG 태그 bump
#       → ops-vm 재시작 → heartbeat 확인
#
# 사전검증을 태그 bump/재시작보다 먼저 실행한다 — 검증 실패 시 registry에 이미지는 push되지만
# compose.yml/DAG 태그도, 실 daemon도 바뀌지 않은 채 중단된다(issue #15).
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

# ---- 2. 태그 중복 검사 -------------------------------------------------------

echo "[release] checking tag uniqueness ..."
for IMG in "$DAEMON_IMG" "$TASK_IMG"; do
    STATUS=$(curl -o /dev/null -sw "%{http_code}" \
        "http://${REGISTRY}/v2/${IMG#${REGISTRY}/}/manifests/${VERSION}" \
        -H "Accept: application/vnd.docker.distribution.manifest.v2+json" 2>/dev/null || true)
    if [[ "$STATUS" == "200" ]]; then
        echo "ERROR: tag already exists in registry — $IMG"
        echo "       use a new version number to enforce immutable tags"
        exit 1
    fi
done

# ---- 3. 빌드 + push (ops-vm) ------------------------------------------------

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

# ---- 4. 사전검증 (일회성 컨테이너, ops-vm) -----------------------------------
# 아직 compose.yml/DAG 태그도, 실행 중인 daemon도 건드리지 않은 시점에 새 이미지를
# 검증한다. 여기서 실패하면 registry에 push된 것 외엔 아무 상태도 바뀌지 않은 채 중단된다
# (issue #15 — 기존엔 태그 bump+재시작 후에야 스모크가 돌아 실패해도 이미 배포된 상태였음).
#
# daemon: compose.yml의 rondo-daemon과 동일한 env_file/network/AIRFLOW_URL로 살아있는
# 컨테이너를 건드리지 않는 일회성 컨테이너에서 healthcheck 실행. ollama_local은 컨테이너 내
# Tailscale 미지원으로 항상 접근 불가 — 명시적으로 skip.
# task: 무거운 실제 eval 대신 import 스모크로 빌드 자체가 깨졌는지만 확인 — 이 이미지는
# CMD가 없고 DockerOperator가 매번 커맨드를 주입하는 구조라 이 이상은 과함.

echo "[release] pre-flight: daemon healthcheck (throwaway container) ..."
ssh ops-vm "
    docker run --rm --network nexus --env-file /var/lib/rondo/.env \
        -e AIRFLOW_URL='http://airflow-api-server-1:8080' \
        $DAEMON_IMG uv run --no-sync python -m bin.healthcheck --skip ollama_local
"

echo "[release] pre-flight: task image import smoke ..."
ssh ops-vm "
    docker run --rm $TASK_IMG uv run --no-sync python -c \"
from evaluator.harness import BasePipeline, PatchedPipeline, PipelineContext, evaluate_pipeline
from runtime import runner
import polars, sklearn, lightgbm, catboost, xgboost
print('task image import OK')
\"
"

# ---- 5. 태그 bump + push (WSL) ----------------------------------------------
# 사전검증을 통과한 뒤에만 실행 — 여기부터는 실제 배포로 간주한다.

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

# ---- 6. 재시작 (ops-vm) -----------------------------------------------------

echo "[release] restarting daemon ..."
ssh ops-vm "
    cd ~/projects/reflexion-rondo
    git pull --quiet
    docker compose -f deploy/compose.yml up -d
"

# ---- 7. 재시작 후 확인 -------------------------------------------------------
# daemon 이미지 자체는 이미 4에서 검증했으므로 여기서는 컴포즈 배선(volume/env/network)
# 문제만 잡는 가벼운 heartbeat 확인으로 충분하다.

echo "[release] post-restart heartbeat check ..."
sleep 5
ssh ops-vm "curl -sf http://localhost:8000/api/heartbeat > /dev/null"

echo "[release] $VERSION deployed successfully"
