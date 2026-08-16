#!/usr/bin/env bash
# Click-run launcher for the published Qwen3.8-27B NVFP4 SM121 profiles.
# Usage:
#   IMAGE=ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121 \
#   MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4-MTP-sm121 \
#   bash scripts/serve.sh mtp
#   bash scripts/serve.sh ar
#   DRAFT_DIR=/path/to/Qwen3.8-27B-DSpark-adapted bash scripts/serve.sh dspark
#   bash scripts/serve.sh long
set -euo pipefail

PROFILE="${1:-mtp}"
IMAGE="${IMAGE:-ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121}"
DRAFT_DIR="${DRAFT_DIR:-$HOME/models/r0b0tlab/Qwen3.8-27B-DSpark-adapted}"
NAME="${NAME:-qwen38-27b}"
PORT="${PORT:-8000}"
VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
DOCKER_CPUS="${DOCKER_CPUS:-14}"
DOCKER_MEM="${DOCKER_MEM:-112g}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "MODEL_DIR missing: $MODEL_DIR" >&2
  exit 2
fi
for shard in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  if [ ! -f "$MODEL_DIR/$shard" ]; then
    echo "incomplete checkpoint: missing $MODEL_DIR/$shard" >&2
    echo "The published HF tree must contain all four 4-of-4 shards. Re-pull r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121." >&2
    exit 3
  fi
done

COMMON=(
  --model /model
  --served-model-name qwen38-27b
  --trust-remote-code
  --kv-cache-dtype fp8
  --enforce-eager
  --no-enable-prefix-caching
)

case "$PROFILE" in
  ar)
    EXTRA=(--max-model-len 32768 --gpu-memory-utilization 0.70)
    VOLS=(-v "$MODEL_DIR:/model:ro")
    ;;
  mtp)
    EXTRA=(
      --max-model-len 32768
      --gpu-memory-utilization 0.70
      --enable-auto-tool-choice
      --tool-call-parser qwen3_xml
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    )
    VOLS=(-v "$MODEL_DIR:/model:ro")
    ;;
  dspark)
    if [ ! -f "$DRAFT_DIR/config.json" ]; then
      echo "DRAFT_DIR missing adapted draft: $DRAFT_DIR" >&2
      echo "Download RadixArk/Qwen3.8-27B-DSpark and run scripts/adapt_dspark_draft.py first." >&2
      exit 4
    fi
    if ! python3 - "$DRAFT_DIR/config.json" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
ok = cfg.get("architectures")==["Qwen3DSparkModel"] and cfg.get("model_type")=="qwen3"
raise SystemExit(0 if ok else 1)
PY
    then
      echo "draft config is still SpecForge-shaped; run scripts/adapt_dspark_draft.py" >&2
      exit 5
    fi
    EXTRA=(
      --max-model-len 32768
      --gpu-memory-utilization 0.70
      --speculative-config '{"method":"dspark","model":"/draft","num_speculative_tokens":7}'
    )
    VOLS=(-v "$MODEL_DIR:/model:ro" -v "$DRAFT_DIR:/draft:ro")
    ;;
  long)
    EXTRA=(
      --max-model-len 262144
      --gpu-memory-utilization 0.85
      --max-num-batched-tokens 8192
    )
    VOLS=(-v "$MODEL_DIR:/model:ro")
    ;;
  *)
    echo "unknown profile: $PROFILE (ar|mtp|dspark|long)" >&2
    exit 6
    ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --gpus all \
  --cpus "$DOCKER_CPUS" \
  --memory "$DOCKER_MEM" \
  -p "$PORT:8000" \
  -e VLLM_HOST_IP="$VLLM_HOST_IP" \
  -e HF_HUB_OFFLINE=1 \
  "${VOLS[@]}" \
  "$IMAGE" \
  "${COMMON[@]}" \
  "${EXTRA[@]}"
echo "launched $NAME profile=$PROFILE image=$IMAGE port=$PORT"
echo "logs: docker logs -f $NAME"
echo "ready probe: curl -fsS http://127.0.0.1:$PORT/health"
echo "canary: python3 scripts/run_semantic_gate.py --base-url http://127.0.0.1:$PORT --output /tmp/qwen38-semantic.json"
