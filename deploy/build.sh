#!/usr/bin/env bash
# worker-vm(ARM64) 배포용 이미지 빌드 및 registry.internal push
# 실행 위치: repo 루트
set -euo pipefail

REGISTRY=registry.internal
DAEMON_IMAGE=$REGISTRY/reflexion-rondo/daemon:latest
EVAL_IMAGE=$REGISTRY/reflexion-rondo/eval:latest

docker buildx build --platform linux/arm64 \
    -t "$DAEMON_IMAGE" -f deploy/Dockerfile . --push

docker buildx build --platform linux/arm64 \
    -t "$EVAL_IMAGE" -f runtime/Dockerfile . --push

echo "pushed: $DAEMON_IMAGE"
echo "pushed: $EVAL_IMAGE"
