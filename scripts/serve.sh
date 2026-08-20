#!/usr/bin/env bash
# Click-run launcher for the published Qwen3.8-27B NVFP4 SM121 profiles.
# Usage:
#   IMAGE=ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121 \
#   MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4-MTP-sm121 \
#   bash scripts/serve.sh mtp
#   bash scripts/serve.sh ar
#   DRAFT_DIR=/path/to/Qwen3.8-27B-DSpark-adapted bash scripts/serve.sh dspark
#   DRAFT_DIR=/path/to/Qwen3.8-27B-DFlash2 bash scripts/serve.sh dflash2
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
VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-}"

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
  dflash2)
    if [ ! -f "$DRAFT_DIR/config.json" ]; then
      echo "DRAFT_DIR missing DFlash2 draft: $DRAFT_DIR" >&2
      echo "Download z-lab/Qwen3.8-27B-DFlash2 first." >&2
      exit 4
    fi
    if ! python3 - "$DRAFT_DIR/config.json" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
df=cfg.get("dflash_config") or {}
ok = cfg.get("architectures")==["DFlash2DraftModel"] and df.get("conv_kernel_size") and df.get("selector_rank")
raise SystemExit(0 if ok else 1)
PY
    then
      echo "draft is not DFlash2DraftModel with conv/selector fields" >&2
      exit 5
    fi
    DFLASH_NUM_SPEC="${DFLASH_NUM_SPEC:-8}"
    EXTRA=(
      --max-model-len "${MAX_MODEL_LEN:-32768}"
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.70}"
      --enable-auto-tool-choice
      --tool-call-parser qwen3_xml
      --speculative-config "{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":${DFLASH_NUM_SPEC}}"
      --no-enable-flashinfer-autotune
      --kernel-config.enable_jit_warmup=false
      --kernel-config.enable_cutedsl_warmup=false
    )
    if [ -n "${MAX_NUM_SEQS:-}" ]; then
      EXTRA+=(--max-num-seqs "$MAX_NUM_SEQS")
    fi
    VOLS=(-v "$MODEL_DIR:/model:ro" -v "$DRAFT_DIR:/draft:ro")
    VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
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
    echo "unknown profile: $PROFILE (ar|mtp|dspark|dflash2|long)" >&2
    exit 6
    ;;
esac

ENV_ARGS=(-e VLLM_HOST_IP="$VLLM_HOST_IP" -e HF_HUB_OFFLINE=1)
if [ -n "${VLLM_USE_V2_MODEL_RUNNER}" ]; then
  ENV_ARGS+=(-e VLLM_USE_V2_MODEL_RUNNER="$VLLM_USE_V2_MODEL_RUNNER")
fi
if [ -n "${CUDA_LAUNCH_BLOCKING:-}" ]; then
  ENV_ARGS+=(-e CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING")
fi
if [ -n "${TORCH_USE_CUDA_DSA:-}" ]; then
  ENV_ARGS+=(-e TORCH_USE_CUDA_DSA="$TORCH_USE_CUDA_DSA")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --restart "${DOCKER_RESTART:-no}" \
  --gpus all \
  --cpus "$DOCKER_CPUS" \
  --memory "$DOCKER_MEM" \
  -p "$PORT:8000" \
  "${ENV_ARGS[@]}" \
  "${VOLS[@]}" \
  "$IMAGE" \
  "${COMMON[@]}" \
  "${EXTRA[@]}"
echo "launched $NAME profile=$PROFILE image=$IMAGE port=$PORT"
echo "logs: docker logs -f $NAME"
echo "ready probe: curl -fsS http://127.0.0.1:$PORT/health"
echo "canary: python3 scripts/run_semantic_gate.py --base-url http://127.0.0.1:$PORT --output /tmp/qwen38-semantic.json"
