#!/usr/bin/env bash
# dev 이미지를 stable로 retag 후 push.
# task: airflow-stack DAG IMAGE 태그 자동 교체
# daemon: deploy/compose.yml 태그 자동 교체
#
# Usage:
#   bash deploy/promote.sh v1.1.0-dev v1.1.0
#
# 전제: mac-server에서 실행, airflow-stack repo가 ~/Projects/airflow-stack 에 위치
set -euo pipefail

DEV_VERSION=${1:?"Usage: bash deploy/promote.sh <dev-version> <stable-version>"}
STABLE_VERSION=${2:?"Usage: bash deploy/promote.sh <dev-version> <stable-version>"}

REGISTRY=registry.internal:5000
DAEMON_BASE=$REGISTRY/reflexion-rondo/daemon
TASK_BASE=$REGISTRY/reflexion-rondo/task
RONDO_DIR=$(cd "$(dirname "$0")/.." && pwd)

echo "promoting $DEV_VERSION -> $STABLE_VERSION"

for BASE in "$DAEMON_BASE" "$TASK_BASE"; do
    docker pull "$BASE:$DEV_VERSION"
    docker tag  "$BASE:$DEV_VERSION" "$BASE:$STABLE_VERSION"
    docker push "$BASE:$STABLE_VERSION"
    echo "pushed: $BASE:$STABLE_VERSION"
done

# deploy/compose.yml — daemon 태그 교체
COMPOSE_FILE=$RONDO_DIR/deploy/compose.yml
sed -i '' "s|/daemon:$DEV_VERSION|/daemon:$STABLE_VERSION|g" "$COMPOSE_FILE"
cd "$RONDO_DIR"
git add deploy/compose.yml
git commit -m "chore: daemon image $DEV_VERSION -> $STABLE_VERSION"
git push
echo "compose.yml updated and pushed"

# airflow-stack — task DAG IMAGE 태그 교체
DAG_FILE=~/Projects/airflow-stack/dags/reflexion_rondo_cycle.py
if [ -f "$DAG_FILE" ]; then
    sed -i '' "s|/task:$DEV_VERSION|/task:$STABLE_VERSION|g" "$DAG_FILE"
    cd ~/Projects/airflow-stack
    git add dags/reflexion_rondo_cycle.py
    git commit -m "chore: task image $DEV_VERSION -> $STABLE_VERSION"
    git push
    echo "DAG updated and pushed"
else
    echo "WARNING: $DAG_FILE not found — update IMAGE tag manually"
fi
