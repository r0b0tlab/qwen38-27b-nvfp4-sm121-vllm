#!/usr/bin/env bash
# Click-run: download the published checkpoint (all 4 shards) and run the
# arithmetic canary against a freshly launched MTP server.
set -euo pipefail
CKPT="${CKPT:-$HOME/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121}"
IMAGE="${IMAGE:-ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121}"
mkdir -p "$(dirname "$CKPT")"
if [ ! -f "$CKPT/model-00001-of-00004.safetensors" ]; then
  huggingface-cli download r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 --local-dir "$CKPT"
fi
for shard in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  [ -f "$CKPT/$shard" ] || { echo "still missing $shard after download"; exit 2; }
done
MODEL_DIR="$CKPT" IMAGE="$IMAGE" bash "$(dirname "$0")/serve.sh" mtp
ready=0
for i in $(seq 1 90); do
  if curl -fsS -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "ready after ${i} probes"
    ready=1
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' qwen38-27b 2>/dev/null | grep -q true; then
    echo "container died during startup; last logs:" >&2
    docker logs --tail 40 qwen38-27b >&2 || true
    exit 3
  fi
  sleep 10
done
if [ "$ready" != 1 ]; then
  echo "READY_TIMEOUT after 90 probes" >&2
  docker logs --tail 40 qwen38-27b >&2 || true
  exit 4
fi
python3 "$(dirname "$0")/run_semantic_gate.py" --base-url http://127.0.0.1:8000 --output /tmp/qwen38-clickrun-semantic.json
python3 - <<'PY'
import json
d=json.load(open("/tmp/qwen38-clickrun-semantic.json"))
print(json.dumps({"passed": d.get("passed"), "checks": d.get("checks")}, indent=2))
raise SystemExit(0 if d.get("passed") else 1)
PY
