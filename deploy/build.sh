#!/usr/bin/env bash
# mac-server(ARM64) 이미지 빌드 및 registry.internal push
#
# Usage:
#   bash deploy/build.sh v1.1.0-dev   # dev 빌드
#   bash deploy/build.sh v1.0.0       # stable 빌드 (정본 배포는 Airflow reflexion_rondo_deploy DAG 경유 권장, issue #17)
#
# BUILDX_NO_DEFAULT_ATTESTATIONS=1: Docker Engine 내장 BuildKit이 attestation 매니페스트를
# 추가하지 않도록 억제 — OCI 인덱스 생성 방지. GC가 자식 매니페스트를 orphan으로 삭제하는
# 문제를 근본 차단. (BON-175)
set -euo pipefail

VERSION=${1:?"Usage: bash deploy/build.sh <version>  (e.g. v1.1.0-dev)"}

REGISTRY=registry.internal:5000
DAEMON_BASE=$REGISTRY/reflexion-rondo/daemon
TASK_BASE=$REGISTRY/reflexion-rondo/task

BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker build \
    -t "$DAEMON_BASE:$VERSION" \
    -f deploy/Dockerfile .
docker push "$DAEMON_BASE:$VERSION"

BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker build \
    -t "$TASK_BASE:$VERSION" \
    -f deploy/Dockerfile.task .
docker push "$TASK_BASE:$VERSION"

echo "pushed: $DAEMON_BASE:$VERSION"
echo "pushed: $TASK_BASE:$VERSION"
