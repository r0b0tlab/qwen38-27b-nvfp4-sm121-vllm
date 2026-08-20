#!/usr/bin/env bash
# Click-run DFlash2: 4-of-4 NVFP4 body + z-lab DFlash2 draft + overlay image + canary.
set -euo pipefail
CKPT="${CKPT:-$HOME/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121}"
DRAFT="${DRAFT:-$HOME/models/z-lab/Qwen3.8-27B-DFlash2}"
IMAGE="${IMAGE:-qwen38-27b-vllm-dflash2-sm121:0.1.1}"
mkdir -p "$(dirname "$CKPT")" "$(dirname "$DRAFT")"
if [ ! -f "$CKPT/model-00001-of-00004.safetensors" ]; then
  huggingface-cli download r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 --local-dir "$CKPT"
fi
for shard in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  [ -f "$CKPT/$shard" ] || { echo "still missing $shard after download"; exit 2; }
done
if [ ! -f "$DRAFT/model.safetensors" ]; then
  huggingface-cli download z-lab/Qwen3.8-27B-DFlash2 --local-dir "$DRAFT"
fi
python3 - "$DRAFT/config.json" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
if cfg.get("architectures") != ["DFlash2DraftModel"]:
    raise SystemExit("draft is not DFlash2DraftModel")
df=cfg.get("dflash_config") or {}
if not (df.get("conv_kernel_size") and df.get("selector_rank")):
    raise SystemExit("draft config lacks DFlash2 conv/selector fields")
PY
docker image inspect "$IMAGE" >/dev/null 2>&1 || \
  docker build -f docker/Dockerfile.dflash2 -t "$IMAGE" docker/
MODEL_DIR="$CKPT" DRAFT_DIR="$DRAFT" IMAGE="$IMAGE" \
  bash "$(dirname "$0")/serve.sh" dflash2
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
python3 "$(dirname "$0")/run_semantic_gate.py" --base-url http://127.0.0.1:8000 --output /tmp/qwen38-dflash2-semantic.json
python3 - <<'PY'
import json
d=json.load(open("/tmp/qwen38-dflash2-semantic.json"))
print(json.dumps({"passed": d.get("passed"), "checks": d.get("checks")}, indent=2))
raise SystemExit(0 if d.get("passed") else 1)
PY
