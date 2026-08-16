---
license: apache-2.0
language: en
pipeline_tag: text-generation
tags:
- nvfp4
- vllm
- sm121
- gb10
- dgx-spark
- mtp
- speculative-decoding
- modelopt
---

# Qwen3.8-27B NVFP4 + MTP (DGX Spark / SM121 build)

Self-quantized Qwen3.8-27B checkpoint for NVIDIA GB10 (DGX Spark, SM121),
produced with **NVIDIA ModelOpt 0.46.0rc1** using the **shipped qwen3_5 family
recipe verbatim** (`w4a16_nvfp4-fp8_attn-kv_fp8_cast`), plus the trained MTP
head merged back in BF16 for speculative decoding.

## Quantization map (census-verified, matches the official family control)

- **193 × W4A16_NVFP4** — all MLP gate/up/down projections + lm_head
  (block-16, e4m3 scales; weights-only, activations BF16)
- **208 × FP8** — self-attn q/k/v/o + GDN linear-attn projections
  (static input scales exported)
- **257 × BF16** — remaining tensors (conv1d, gates/routers, small GDN paths)
- **15 × BF16 MTP tensors** — trained multi-token-prediction head (shard 4,
  sha256 `47202b11…`, mirrors the official NVFP4-family checkpoint contract)

## Calibration (NVIDIA-documented path)

`get_dataset_dataloader` with the registered combo `cnn_dailymail` +
`nvidia/Nemotron-Post-Training-Dataset-v2` (stem/chat/math/code splits),
2048 packed rows @ 1024 tokens, left padding, `pack=True`, `max` algorithm.

## IMPORTANT: FP8 KV flag

This checkpoint ships `kv_cache_quant_algo: "FP8"` (cast mode, default scales).
**Serve with `--kv-cache-dtype fp8`.** Do not strip this flag: on vLLM 0.27.x
with qwen3_5-family models, a checkpoint without the flag that is served with
the runtime fp8 option takes a broken generic path and produces a
deterministic arithmetic defect (e.g. 19×23 answered "417"). With the flag,
KV routes through `ModelOptKVCacheMethod` and is exact. Root-caused during
this campaign; upstream-report candidate.

## Verified results (single DGX Spark, vLLM v0.27.2rc0-sm121)

| Gate | Result |
|---|---|
| Semantic (10 checks) | PASS (AR / MTP K2 / K3 / fp8-KV profiles) |
| GSM8K flex (pre-quant BF16 floor) | 86.2% exact / 96.2% numeric-norm |
| GSM8K flex (official family control) | 86.2% exact / 93.8% numeric-norm |
| GSM8K flex (this checkpoint, FP8 KV) | 81.25% exact / 91.25% numeric-norm (5 of 8 fails are "26.00"-style formatting) |
| HumanEval / IFEval / agentic | 39/40 · 37/40 · 17/20 (release run) |
| Sanity suite | 8/8 incl. qwen3_xml tool-calls, 2.8K long-gen, determinism |
| NIAH @ 262,144 ctx | 8/8 PASS (5 positions @ ~247.7K actual + 8K/32K/131K ladder) |
| MTP (K*=3) | c1 22.4 tok/s vs AR 11.35 (1.97×); acceptance len 2.5-2.7 |
| Dedicated c1 (2048 tok) | 27.8-28.1 tok/s median (2.45-2.48× AR) |
| c8 aggregate | 84.3 tok/s best, 17/17 zero-error |

## Serve

```bash
docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 -p 8000:8000 \
  -v /path/to/this/checkpoint:/model:ro \
  ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121 \
  --model /model --served-model-name qwen38-27b \
  --max-model-len 32768 --gpu-memory-utilization 0.70 \
  --kv-cache-dtype fp8 --enforce-eager --no-enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

For 262K-context serving: `--max-model-len 262144 --gpu-memory-utilization 0.85
--max-num-batched-tokens 8192` (KV capacity ≈ 2.5M tokens with fp8 KV).

## Provenance

- Source: Qwen3.8-27B (BF16), self-quantized per the NVIDIA ModelOpt shipped
  recipe; calibration via ModelOpt's documented dataset path.
- Full reproducibility pack (PTQ worker, MTP merge, all gates, raw evidence):
  https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-vllm
- Runtime image: `ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121`
  (registry digest `sha256:5bd3f329…b775f`).

Weights follow the upstream Qwen license terms; this repository provides the
quantized derivative for reproducibility. No warranties.
