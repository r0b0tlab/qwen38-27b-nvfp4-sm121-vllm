#!/usr/bin/env bash
# run-ptq.sh — Qwen3.8-27B NVFP4 PTQ on node2 (durable, sentinel-based).
# Adapts the proven Aquila run_ptq.sh pattern to the qwen38-quant:v2 container.
set -uo pipefail

ROOT="${Q38_OPS:-$HOME/qwen38-nvfp4-work}"
SRC="${Q38_BF16_SRC:-$HOME/models/Qwen/Qwen3.8-27B}"
IMAGE="${Q38_QUANT_IMAGE:-qwen38-quant:v3-modelopt046rc1}"
ATTEMPT="${1:?attempt id required}"
EVIDENCE="$ROOT/evidence/$ATTEMPT"
EXPORT="$ROOT/candidates/$ATTEMPT"
WORKER="$ROOT/ptq_worker.py"
mkdir -p "$EVIDENCE" "$ROOT/candidates"

[ -f "$WORKER" ] || { echo "worker missing: $WORKER" >&2; exit 20; }
[ -d "$SRC" ] || { echo "source missing: $SRC" >&2; exit 21; }
[ ! -e "$EXPORT" ] || { echo "candidate exists: $EXPORT" >&2; exit 22; }

{
  echo "attempt=$ATTEMPT"
  echo "image=$IMAGE"
  date -u -Is
  free -h
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
  sha256sum "$WORKER" "$ROOT/configs/qwen38_27b_nvfp4_w4a4_fp8_attn_kv.yaml"
} | tee "$EVIDENCE/preflight.txt"

# GPU + memory containment (Puzzle lesson): cap container so OOM kills the
# container, not sshd; detached launch, sentinel-wait below.
docker rm -f q38-ptq >/dev/null 2>&1 || true
docker run -d \
  --name q38-ptq \
  --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
  --cpus 14 --memory 108g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e Q38_SRC=/src -e Q38_OUT=/candidates/$ATTEMPT -e Q38_ATTEMPT=/attempt -e Q38_PROFILE="${Q38_PROFILE:-mixed}" -e Q38_KV="${Q38_KV:-fp8}" -e Q38_CALIB="${Q38_CALIB:-0}" -e Q38_CALIB_FILE="${Q38_CALIB_FILE:-/calib/calibration.jsonl}" -e Q38_CALIB_ROWS="${Q38_CALIB_ROWS:-512}" -e Q38_CALIB_MAXLEN="${Q38_CALIB_MAXLEN:-512}" \
  -v "$HOME/qwen38-ops/calibration:/calib:ro" \
  -v "$SRC:/src:ro" \
  -v "$ROOT/candidates:/candidates" \
  -v "$EVIDENCE:/attempt" \
  -v "$WORKER:/ptq_worker.py:ro" \
  --entrypoint python3 \
  "$IMAGE" \
  /ptq_worker.py >/dev/null

echo "container q38-ptq launched; logs: docker logs -f q38-ptq"
echo "$ATTEMPT" > "$EVIDENCE/attempt.txt"

# durable follow: wait for sentinel, capture logs
for i in $(seq 1 720); do   # 30s x 720 = 6h (local_hessian sweep)
  if [ -f "$EVIDENCE/PASS" ]; then
    docker logs q38-ptq > "$EVIDENCE/ptq.log" 2>&1
    echo "PASS at $(date -u -Is)" >> "$EVIDENCE/attempt.txt"
    docker rm -f q38-ptq >/dev/null 2>&1
    exit 0
  fi
  if [ -f "$EVIDENCE/FAIL" ]; then
    docker logs q38-ptq > "$EVIDENCE/ptq.log" 2>&1
    echo "FAIL at $(date -u -Is)" >> "$EVIDENCE/attempt.txt"
    docker rm -f q38-ptq >/dev/null 2>&1
    exit 1
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^q38-ptq$'; then
    sleep 5
    if ! docker ps -a --format '{{.Names}} {{.Status}}' | grep -q '^q38-ptq '; then
      echo "container vanished without sentinel at $(date -u -Is)" | tee -a "$EVIDENCE/attempt.txt"
      docker logs q38-ptq > "$EVIDENCE/ptq.log" 2>&1
      exit 2
    fi
  fi
  sleep 30
done
echo "TIMEOUT after 6h" | tee -a "$EVIDENCE/attempt.txt"
docker logs q38-ptq > "$EVIDENCE/ptq.log" 2>&1
exit 3
