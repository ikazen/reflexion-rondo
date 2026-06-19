#!/usr/bin/env bash
# mac-server(ARM64) 이미지 빌드 및 registry.internal push
#
# Usage:
#   bash deploy/build.sh v1.1.0-dev   # dev 빌드
#   bash deploy/build.sh v1.0.0       # stable 빌드 (promote.sh 경유 권장)
set -euo pipefail

VERSION=${1:?"Usage: bash deploy/build.sh <version>  (e.g. v1.1.0-dev)"}

REGISTRY=registry.internal:5000
DAEMON_BASE=$REGISTRY/reflexion-rondo/daemon
TASK_BASE=$REGISTRY/reflexion-rondo/task

docker build --provenance=false \
    -t "$DAEMON_BASE:$VERSION" \
    -f deploy/Dockerfile . && \
docker push "$DAEMON_BASE:$VERSION"

docker build --provenance=false \
    -t "$TASK_BASE:$VERSION" \
    -f deploy/Dockerfile.task . && \
docker push "$TASK_BASE:$VERSION"

echo "pushed: $DAEMON_BASE:$VERSION"
echo "pushed: $TASK_BASE:$VERSION"
