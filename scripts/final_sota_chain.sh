#!/usr/bin/env bash
# Final SOTA chain: PTQ-done -> file restore -> serve rc0 -> semantic -> quality-200 -> flex verdict.
# Idempotent; run from head. Assumes candidates/final-sota-nvidia-recipe exists and q38-ptq is done.
set -uo pipefail
ROOT="$HOME/projects/qwen38-27b-nvfp4-sm121-vllm"
NODE2="${NODE2:-r0b0tdgx@192.168.0.2}"  # set your own host
CAND=final-sota-nvidia-recipe
EV="$ROOT/evidence/serve-$CAND"
mkdir -p "$EV"

echo "== 1. restore tokenizer/processor files =="
timeout 60 ssh -o BatchMode=yes "$NODE2" 'cd ~/qwen38-ops/candidates/final-sota-nvidia-recipe && ls tokenizer.json >/dev/null 2>&1 || { cp ~/models/llm/bf16/Qwen3.8-27B/{chat_template.jinja,merges.txt,preprocessor_config.json,tokenizer_config.json,tokenizer.json,video_preprocessor_config.json,vocab.json} . && echo FILES_RESTORED; } && echo files-ok; docker run --rm --user 0 -v /home/r0b0tdgx/qwen38-ops/candidates/final-sota-nvidia-recipe:/c --entrypoint chown qwen38-quant:v3-modelopt046rc1 -R 1001:1001 /c >/dev/null 2>&1; echo chown-ok'

echo "== 2. serve on rc0 + BF16 KV =="
timeout 60 ssh -o BatchMode=yes "$NODE2" 'docker rm -f qwen38-27b >/dev/null 2>&1; docker run -d --name qwen38-27b --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 --restart unless-stopped --cpus 14 --memory 108g -p 8000:8000 -e VLLM_HOST_IP=127.0.0.1 -e HF_HUB_OFFLINE=1 -v "$HOME/qwen38-ops/candidates/final-sota-nvidia-recipe:/model:ro" --entrypoint python3 vllm/vllm-openai:v0.27.2rc0-sm121 -m vllm.entrypoints.openai.api_server --model /model --served-model-name qwen38-27b --max-model-len 32768 --gpu-memory-utilization 0.70 --kv-cache-dtype fp8 --trust-remote-code --enforce-eager --no-enable-prefix-caching >/dev/null && echo SERVED'

echo "== 3. readiness (crash-detect) =="
for i in $(seq 1 90); do
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' ${Q38_BASE_URL:-http://127.0.0.1:8000}/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo "READY ~$((i*10))s"; break; }
  timeout 30 ssh -o BatchMode=yes "$NODE2" "docker inspect -f '{{.State.Running}} {{.RestartCount}}' qwen38-27b" 2>/dev/null | grep -q "^true 0" || { echo CRASH; timeout 30 ssh -o BatchMode=yes "$NODE2" 'docker logs qwen38-27b 2>&1 | tail -30' > "$EV/crash-tail.txt"; exit 3; }
  sleep 10
done
curl -s -m 5 ${Q38_BASE_URL:-http://127.0.0.1:8000}/health >/dev/null 2>&1 || { echo READY_TIMEOUT; exit 4; }
timeout 30 ssh -o BatchMode=yes "$NODE2" 'docker logs qwen38-27b 2>&1 | grep -iE "Selected|kernel|Marlin|KV cache size"' > "$EV/engine-markers.txt"
head -6 "$EV/engine-markers.txt"

echo "== 4. semantic gate (fixed 1024-token arithmetic) =="
timeout 600 python3 "$ROOT/repo/scripts/run_semantic_gate.py" --base-url ${Q38_BASE_URL:-http://127.0.0.1:8000} --output "$EV/semantic-gate.json" >/dev/null 2>&1
python3 -c "import json; d=json.load(open('$EV/semantic-gate.json')); print('SEMANTIC:', 'PASS' if d.get('passed') else 'FAIL')" || echo GATE_ERR

echo "== 5. quality-200 + flex verdict =="
cd "$ROOT" && python3 repo/scripts/run_quality_set.py --base-url ${Q38_BASE_URL:-http://127.0.0.1:8000} --run-id final-sota --set artifacts/quality-200.jsonl 2>&1 | tail -3
python3 repo/scripts/score_flex_gsm8k.py final-sota.rows.jsonl artifacts/quality-200.jsonl | tee "$EV/gsm8k-flex.json"
echo "CHAIN_DONE"
