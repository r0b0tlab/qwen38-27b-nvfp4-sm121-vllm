# Qwen3.8-27B NVFP4 on NVIDIA DGX Spark (GB10 / SM121)

Reproducibility package for serving Qwen3.8-27B (self-quantized native NVFP4,
ModelOpt 0.46.0rc1 mixed W4A4) on a single NVIDIA DGX Spark (GB10, SM121)
with vLLM.

## Headline profile

- Runtime: vLLM v0.27.1 (stable) / v0.27.2rc0 (source-built SM121 wheel)
- Quantization: ModelOpt mixed — 193 NVFP4 (W4A4, block-16 e4m3) + 208 FP8
  layers + BF16 passthrough; lm_head NVFP4 g16
- Native kernels: FlashInferCutlassNvFp4LinearKernel (linear),
  FlashInferFP8ScaledMM (FP8 attention projections), Triton/FLA GDN prefill
- KV cache: BF16 (`--kv-cache-dtype auto`). FP8 KV produces a deterministic
  arithmetic defect on this family (see Known issues).
- Eager mode (SM121), VLLM_HOST_IP=127.0.0.1, gpu-mem-util 0.70
- MTP speculative decoding: 15-tensor BF16 MTP head merged from source
  (mirrors official Qwen3.6-27B-NVFP4 contract), method `mtp`

## Reproduce

```bash
# 1. quantize (ModelOpt 0.46.0rc1, official 512-row calib, local_hessian)
bash scripts/run-ptq.sh          # see configs/qwen38_27b_nvfp4_w4a4_fp8_attn_kv.yaml
# 2. merge the BF16 MTP head from the BF16 source
python3 scripts/merge_mtp_head.py
# 3. serve
MODEL_DIR=~/qwen38-ops/candidates/attempt18-mixedhess-official512-mtp \
  KV_DTYPE=auto bash scripts/serve.sh [image]
# 4. gates
python3 scripts/run_semantic_gate.py --base-url http://<node>:8000 --output semantic.json
python3 scripts/run_sanity_suite.py --base-url http://<node>:8000 --output sanity.json
python3 scripts/run_quality_set.py --base-url http://<node>:8000 --run-id <id> --set ../artifacts/quality-200.jsonl
python3 scripts/score_flex_gsm8k.py <id>.rows.jsonl ../artifacts/-quality-200.jsonl
```

## Results

See `results/` and `evidence-summary/` (populated at publication).

## Known issues

- FP8 KV cache on vLLM 0.27.x + qwen3_5-family hybrid GDN models produces a
  deterministic arithmetic defect (19×23 → "417"). BF16 KV is unaffected.
  Upstream-report candidate; evidence in this repo.

## License

MIT for scripts/docs. Model weights follow the upstream Qwen license; weights
are not redistributed in this repo.
