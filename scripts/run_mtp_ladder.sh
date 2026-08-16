#!/usr/bin/env bash
# MTP K-ladder against a remote GPU host. Optional campaign helper, not required
# for the published click-run path (use scripts/serve.sh mtp instead).
# Usage: NODE2=user@gpu-host KLIST="2 3" bash scripts/run_mtp_ladder.sh <image> [run-tag]
set -uo pipefail
IMAGE="${1:?image required}"
TAG="${2:-mtp-ladder}"
KLIST="${KLIST:-2 3}"
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
NODE2="${NODE2:?set NODE2 to the GPU host, e.g. user@gpu-host}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121}"
BASE_URL="${Q38_BASE_URL:-http://127.0.0.1:8000}"
mkdir -p "$ROOT/evidence/mtp-ladder"

for K in $KLIST; do
  RUN="$TAG-k${K}"
  OUT="$ROOT/evidence/mtp-ladder/$RUN"
  mkdir -p "$OUT"
  echo "=== [$(date -u -Is)] K=$K ===" | tee -a "$ROOT/evidence/mtp-ladder/progress.log"

  ssh -o BatchMode=yes "$NODE2" "docker rm -f qwen38-27b >/dev/null 2>&1; \
    docker run -d --name qwen38-27b --gpus all \
    --restart unless-stopped --cpus 14 --memory 108g -p 8000:8000 \
    -e VLLM_HOST_IP=127.0.0.1 -e HF_HUB_OFFLINE=1 \
    -v \"$MODEL_DIR:/model:ro\" \
    \"$IMAGE\" \
    --model /model --served-model-name qwen38-27b \
    --max-model-len 32768 --gpu-memory-utilization 0.70 \
    --kv-cache-dtype fp8 --trust-remote-code --enforce-eager --no-enable-prefix-caching \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}' \
    >/dev/null" || { echo "LAUNCH_FAIL k${K}" | tee -a "$ROOT/evidence/mtp-ladder/progress.log"; exit 2; }

  ready=0
  for i in $(seq 1 90); do
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$BASE_URL/health" || true)
    [ "$code" = "200" ] && ready=1 && break
    if ! ssh -o BatchMode=yes "$NODE2" "docker inspect -f '{{.State.Running}}' qwen38-27b" 2>/dev/null | grep -q true; then
      ssh -o BatchMode=yes "$NODE2" "docker logs qwen38-27b 2>&1 | tail -40" > "$OUT/crash-tail.txt"
      echo "CRASH_LOOP k${K} — see $OUT/crash-tail.txt" | tee -a "$ROOT/evidence/mtp-ladder/progress.log"
      exit 3
    fi
    sleep 10
  done
  [ "$ready" = "1" ] || { echo "READY_TIMEOUT k${K}" | tee -a "$ROOT/evidence/mtp-ladder/progress.log"; exit 4; }
  echo "ready after ~$((i*10))s" | tee -a "$ROOT/evidence/mtp-ladder/progress.log"

  ssh -o BatchMode=yes "$NODE2" "docker logs qwen38-27b 2>&1 | grep -iE 'spec|mtp|draft|accept' | tail -15" > "$OUT/spec-markers.txt"
  curl -s -m 5 "$BASE_URL/v1/models" | head -c 200 > "$OUT/models.json"

  timeout 600 python3 "$ROOT/scripts/run_semantic_gate.py" \
    --base-url "$BASE_URL" --output "$OUT/semantic-gate.json" >/dev/null 2>&1
  python3 - "$OUT/semantic-gate.json" <<'EOF' | tee -a "$ROOT/evidence/mtp-ladder/progress.log"
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(f"K-GATE semantic passed={d.get('passed')}")
except Exception as exc:
    print(f"K-GATE semantic ERROR {exc}")
EOF

  timeout 1200 python3 - "$OUT/mtp-probe.json" "$BASE_URL" <<'EOF' || true
import json, sys, time, urllib.request
url = sys.argv[2].rstrip("/") + "/v1/chat/completions"
def gen(prompt, mt=512):
    payload = {"model": "qwen38-27b",
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": mt,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1100) as r:
        b = json.loads(r.read())
    u = b.get("usage", {})
    ct = u.get("completion_tokens", 0)
    return ct, time.perf_counter() - t0, b
rows = []
for rep in range(3):
    ct, el, b = gen("Write a detailed essay on the history of computing.", 512)
    rows.append({"rep": rep, "completion_tokens": ct, "elapsed_s": round(el, 2),
                 "tok_s": round(ct / el, 2)})
out = {"rows": rows, "median_tok_s": sorted(r["tok_s"] for r in rows)[1]}
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(json.dumps(out, indent=1))
EOF

  ssh -o BatchMode=yes "$NODE2" "docker logs qwen38-27b 2>&1 | grep -iE 'spec_num_accepted|acceptance|draft' | tail -5" > "$OUT/acceptance-tail.txt" || true
  echo "=== K=$K done ===" | tee -a "$ROOT/evidence/mtp-ladder/progress.log"
done
echo "LADDER_DONE" | tee -a "$ROOT/evidence/mtp-ladder/progress.log"
