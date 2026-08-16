#!/usr/bin/env bash
# Optional: layer a locally built v0.27.2rc0 wheel onto vllm/vllm-openai:v0.27.1.
# The published claim path is to pull GHCR, not rebuild.
# Usage:
#   WHEEL=/path/to/vllm-0.27.2rc0-cp312-cp312-linux_aarch64.whl \
#   bash scripts/assemble_rc0.sh
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WHEEL="${WHEEL:?set WHEEL to the built v0.27.2rc0 aarch64 wheel}"
IMAGE_TAG="${IMAGE_TAG:-ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121}"
[ -f "$WHEEL" ] || { echo "NO_WHEEL: $WHEEL"; exit 2; }
echo "wheel: $WHEEL"
sha256sum "$WHEEL"
WORKDIR=$(mktemp -d)
cp "$WHEEL" "$WORKDIR/$(basename "$WHEEL")"
cp "$ROOT/docker/Dockerfile.rc0" "$WORKDIR/Dockerfile.rc0"
docker build -f "$WORKDIR/Dockerfile.rc0" \
  --build-arg "WHEEL=$(basename "$WHEEL")" \
  -t "$IMAGE_TAG" "$WORKDIR"
rm -rf "$WORKDIR"
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep "$IMAGE_TAG" || true
echo "ASM done. Image: $IMAGE_TAG"
echo "Optional GPU audit: docker run --rm --gpus all --entrypoint python3 $IMAGE_TAG /audit/audit_rc0_image.py"
