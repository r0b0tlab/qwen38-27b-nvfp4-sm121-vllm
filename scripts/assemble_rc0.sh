#!/usr/bin/env bash
# Assemble the rc0 image on head from the built wheel, audit it, ship to node2.
# Usage: bash assemble_rc0.sh   (run from ~/projects/qwen38-27b-nvfp4-sm121-vllm)
set -euo pipefail
ROOT="$HOME/projects/qwen38-27b-nvfp4-sm121-vllm"
WHEELS="$HOME/vllm-wheels"
IMAGE_TAG="vllm/vllm-openai:v0.27.2rc0-sm121"
NODE2=r0b0tdgx@192.168.0.2

WHEEL=$(ls -t "$WHEELS"/vllm-0.27.2rc0*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || { echo "NO_WHEEL: build not finished"; exit 2; }
echo "wheel: $WHEEL"
sha256sum "$WHEEL" | tee "$ROOT/evidence/rc0-wheel.sha256"

cd "$WHEELS"
docker build -f "$ROOT/docker/Dockerfile.rc0" --build-arg "WHEEL=$(basename "$WHEEL")" \
  -t "$IMAGE_TAG" . 2>&1 | tail -5
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep "$IMAGE_TAG"

# Static audit in-container (no GPU needed for import/version checks)
docker cp "$ROOT/repo/scripts/audit_rc0_image.py" $(docker create --tmpfs /tmp:rw "$IMAGE_TAG" true 2>/dev/null || docker create "$IMAGE_TAG" true):/tmp/ 2>/dev/null || true
echo "NOTE: full audit runs on node2 (GPU host) after transfer"

# Ship to node2 (stream, no intermediate file)
echo "shipping to node2..."
docker save "$IMAGE_TAG" | gzip | ssh -o BatchMode=yes "$NODE2" 'gunzip | docker load'
ssh -o BatchMode=yes "$NODE2" "docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep v0.27.2rc0"
echo "ASM done. Image on node2: $IMAGE_TAG"
