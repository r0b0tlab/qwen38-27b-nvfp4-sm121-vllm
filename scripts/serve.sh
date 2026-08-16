#!/usr/bin/env bash
# Qwen3.8-27B serve launcher — single GB10 node2, SM121. SCAFFOLD v0.
# Adapted from puzzle-75b serve.sh containment + qwen36-27B release contract.
# Usage: SPEC_K= MAX_MODEL_LEN= bash serve.sh [image]
set -uo pipefail

IMAGE="${1:-vllm/vllm-openai:v0.27.1}"  # latest stable as of 2026-08-15 (v0.27.2 does not exist)
shift 2>/dev/null || true  # drop the image arg so "$@" only carries vllm flags
MODEL_DIR="${MODEL_DIR:-$HOME/models/llm/nvfp4/qwen/Qwen3.8-27B-NVFP4}"  # placeholder until identity lands
NAME="${NAME:-qwen38-27b}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"     # staged up after first admission
GPU_UTIL="${GPU_UTIL:-0.70}"
KV_DTYPE="${KV_DTYPE:-fp8}"    # fp8 is the production path. REQUIRES the checkpoint
                               # flag kv_cache_quant_algo="FP8" (this campaign's export has it).
                               # Without that flag, the runtime fp8 option takes a broken
                               # generic path (deterministic 19x23 -> "417"); with the flag,
                               # KV routes through ModelOptKVCacheMethod and is exact.
                               # "auto" (BF16 KV) is the fallback and is quality-neutral
                               # but halves KV capacity.
SPEC_K="${SPEC_K:-}"                        # empty = base-AR first (canonical); MTP only after AR qualifies
VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"   # NCCL fe80 lesson
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"         # SM121 codegen fragility; relax only with evidence
DOCKER_CPUS="${DOCKER_CPUS:-14}"
DOCKER_MEM="${DOCKER_MEM:-112g}"

SPEC_ARG=()
if [ -n "$SPEC_K" ]; then
  SPEC_ARG=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_K}}")
fi
EAGER_ARG=()
if [ "$ENFORCE_EAGER" = "1" ]; then
  EAGER_ARG=(--enforce-eager)
fi

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
  -v "$MODEL_DIR:/model:ro" \
  "$IMAGE" \
  --model /model \
  --served-model-name qwen38-27b \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-dtype "$KV_DTYPE" \
  --trust-remote-code \
  "${SPEC_ARG[@]}" "${EAGER_ARG[@]}" "$@" >/dev/null
# NOTE: NO --reasoning-parser qwen3 — qwen3_5/3.8 family has no <think> special
# tokens (verified: vocab ids are unk); v0.27 ReasoningConfig tokenize-validation
# fails on empty strings. Thinking control is chat-template kwargs
# (enable_thinking) only; floors (BF16 baseline, official-Q36 control) were
# measured without a reasoning parser.
echo "launched $NAME (image=$IMAGE len=$MAX_MODEL_LEN spec_k=${SPEC_K:-none} eager=$ENFORCE_EAGER)"
echo "logs: docker logs -f $NAME"
