#!/usr/bin/env bash
# Throughput + concurrency suite for Qwen3.8-27B NVFP4 on node2 (vllm bench serve).
# Dedicated c1 long-decode (median of 5) + concurrent ladder, one level at a time.
set -euo pipefail
ROOT="${ROOT:-/home/r0b0tdgx/projects/qwen38-27b-nvfp4-sm121-vllm}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-qwen38-27b}"
CONTAINER="${CONTAINER:-qwen38-27b}"
MODEL_PATH="${MODEL_PATH:-/model}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="$ROOT/evidence/perf/$RUN_ID"
mkdir -p "$OUT"

LEVELS="${LEVELS:-1 2 4 8}"
REPS="${REPS:-3}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
C1_TOKENS="${C1_TOKENS:-2048}"

{
  echo "run_id=$RUN_ID"
  echo "container=$CONTAINER port=$PORT model=$MODEL"
  echo "levels=$LEVELS reps=$REPS input=$INPUT_LEN output=$OUTPUT_LEN c1_tokens=$C1_TOKENS"
  date -u -Is
  curl -fsS "http://127.0.0.1:$PORT/v1/models"
} | tee "$OUT/preflight.txt"

fetch_result () {
  docker cp "$CONTAINER:/tmp/$1" "$OUT/" >/dev/null 2>&1 || true
  python3 - "$OUT/$1" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    ot = d.get("output_throughput")
    if ot is None:
        ot = (d.get("output_len") or 0) / max(d.get("duration") or 1, 1e-9)
    print(f"  {sys.argv[1].split('/')[-1]}: output_throughput={ot:.2f} tok/s")
except Exception as exc:
    print(f"  {sys.argv[1].split('/')[-1]}: no result ({exc})")
EOF
}

# --- dedicated c1: long decode, median of 5 ---
echo "=== dedicated c1 (max_tokens=$C1_TOKENS, 5 reps) ===" | tee -a "$OUT/progress.log"
for rep in 1 2 3 4 5; do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null || { echo "SERVER_DOWN c1-r${rep}"; exit 2; }
  docker exec "$CONTAINER" /usr/local/bin/vllm bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:${PORT}" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --tokenizer "$MODEL_PATH" \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len "$C1_TOKENS" \
    --num-prompts 1 \
    --max-concurrency 1 \
    --request-rate inf \
    --seed 0 --ignore-eos --temperature 0 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir /tmp \
    --result-filename "qwen38-dedicated-c1-r${rep}.json" \
    >"$OUT/dedicated-c1-r${rep}.log" 2>&1 || { echo "C1_R${rep}_FAIL" | tee -a "$OUT/progress.log"; exit 2; }
  fetch_result "qwen38-dedicated-c1-r${rep}.json"
done

# --- concurrent ladder ---
for c in $LEVELS; do
  prompts=$(( c * 4 )); (( prompts < 8 )) && prompts=8
  for rep in $(seq 1 "$REPS"); do
    echo "=== c=${c} rep=${rep} prompts=${prompts} $(date -u -Is) ===" | tee -a "$OUT/progress.log"
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null || { echo "SERVER_DOWN c${c}-r${rep}"; exit 2; }
    docker exec "$CONTAINER" /usr/local/bin/vllm bench serve \
      --backend openai-chat \
      --base-url "http://127.0.0.1:${PORT}" \
      --endpoint /v1/chat/completions \
      --model "$MODEL" \
      --tokenizer "$MODEL_PATH" \
      --dataset-name random \
      --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
      --num-prompts "$prompts" \
      --max-concurrency "$c" \
      --request-rate inf \
      --seed 0 --ignore-eos --temperature 0 \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir /tmp \
      --result-filename "qwen38-c${c}-r${rep}.json" \
      >"$OUT/c${c}-r${rep}.log" 2>&1 || { echo "C${c}_R${rep}_FAIL" | tee -a "$OUT/progress.log"; exit 2; }
    fetch_result "qwen38-c${c}-r${rep}.json"
  done
done
echo "SUITE_DONE $RUN_ID" | tee -a "$OUT/progress.log"
