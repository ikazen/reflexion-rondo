#!/usr/bin/env bash
# worker-vm(ARM64) 배포용 이미지 빌드 및 registry.internal push
# 실행 위치: repo 루트
set -euo pipefail

REGISTRY=registry.internal
SHA=$(git rev-parse --short HEAD)

DAEMON_BASE=$REGISTRY/reflexion-rondo/daemon
EVAL_BASE=$REGISTRY/reflexion-rondo/eval

docker buildx build --platform linux/arm64 \
    -t "$DAEMON_BASE:$SHA" -t "$DAEMON_BASE:latest" \
    -f deploy/Dockerfile . --push

docker buildx build --platform linux/arm64 \
    -t "$EVAL_BASE:$SHA" -t "$EVAL_BASE:latest" \
    -f runtime/Dockerfile . --push

echo "pushed: $DAEMON_BASE:$SHA (latest)"
echo "pushed: $EVAL_BASE:$SHA (latest)"
