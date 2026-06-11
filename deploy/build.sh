#!/usr/bin/env bash
# mac-server(ARM64) 배포용 이미지 빌드 및 registry.internal push
# 실행 위치: repo 루트
set -euo pipefail

REGISTRY=registry.internal:80
SHA=$(git rev-parse --short HEAD)

DAEMON_BASE=$REGISTRY/reflexion-rondo/daemon
TASK_BASE=$REGISTRY/reflexion-rondo/task

docker build \
    -t "$DAEMON_BASE:$SHA" -t "$DAEMON_BASE:latest" \
    -f deploy/Dockerfile . && \
docker push "$DAEMON_BASE:$SHA" && docker push "$DAEMON_BASE:latest"

docker build \
    -t "$TASK_BASE:$SHA" -t "$TASK_BASE:latest" \
    -f deploy/Dockerfile.task . && \
docker push "$TASK_BASE:$SHA" && docker push "$TASK_BASE:latest"

echo "pushed: $DAEMON_BASE:$SHA (latest)"
echo "pushed: $TASK_BASE:$SHA (latest)"
