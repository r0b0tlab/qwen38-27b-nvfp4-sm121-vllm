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
- KV cache: FP8 (`--kv-cache-dtype fp8`). The checkpoint ships
  `kv_cache_quant_algo: "FP8"` — REQUIRED for correct fp8 KV (see Known issues).
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
  KV_DTYPE=fp8 bash scripts/serve.sh [image]
# 4. gates
python3 scripts/run_semantic_gate.py --base-url http://<node>:8000 --output semantic.json
python3 scripts/run_sanity_suite.py --base-url http://<node>:8000 --output sanity.json
python3 scripts/run_quality_set.py --base-url http://<node>:8000 --run-id <id> --set ../artifacts/quality-200.jsonl
python3 scripts/score_flex_gsm8k.py <id>.rows.jsonl ../artifacts/-quality-200.jsonl
```

## Speculative decoding: two validated speculators (same checkpoint)

Both were measured on the identical checkpoint/harness/hardware (17/17 runs, zero errors):

| Level | MTP K=3 (in-checkpoint head) | DSpark K=7 (RadixArk external draft) | winner |
|---|---:|---:|---|
| Dedicated c1 (2048 tok) | 27.83 tok/s | **28.46 tok/s** (best 30.43) | DSpark |
| c1 (256 tok) | **19.22 tok/s** | 16.05 tok/s | MTP |
| c2 | 26.86 tok/s | **28.47 tok/s** | DSpark |
| c4 | 34.61 tok/s | **43.88 tok/s** (+27%) | DSpark |
| c8 | **82.89 tok/s** | 61.53 tok/s | MTP |

- **MTP K=3**: use for throughput serving (≥c8) and short decodes. In-checkpoint BF16
  head, `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.
- **DSpark K=7**: use for latency-sensitive c1-c4 and reasoning workloads (mean
  acceptance 3.5 with thinking on — matches the RadixArk reference regime).
  Requires the external draft (RadixArk/Qwen3.8-27B-DSpark) with the config
  normalization described in `dspark-report.json`:
  `--speculative-config '{"method":"dspark","model":"/path/to/draft","num_speculative_tokens":7}'`.
  Draft trained against Qwen/Qwen3.8-27B-FP8; validated correct on both BF16 and
  this W4A16 target (semantic 10/10 both).

## Results — pre/post quantization (single DGX Spark, thinking off, temp 0)

Same 200-item set, same flex-extract scorer, every run on this hardware:

| Run | Quant | GSM8K exact | GSM8K numeric-norm | HumanEval | IFEval | Agentic |
|---|---|---|---|---|---|---|
| BF16 baseline (pre-quant floor) | — | 86.2% | 96.2% | 39/40 | 37/40 | 16/20 |
| official nvidia/Qwen3.6-27B-NVFP4 (family control) | W4A16 | 86.2% | 93.8% | 39/40 | 38/40 | 18/20 |
| attempt15 (early self-quant) | W4A16 | 78.8% | 88.8% | 39/40 | 38/40 | 18/20 |
| attempt18 (W4A4 local_hessian, v0.27.1) | W4A4 | 81.2% | 90.0% | 39/40 | 38/40 | 18/20 |
| attempt18 (W4A4, rc0 engine) | W4A4 | 82.5% | 92.5% | 38/40 | 37/40 | 18/20 |
| **final-sota (shipped recipe, BF16 KV)** | W4A16 | 81.25% | 91.25% | 38/40 | 38/40 | 18/20 |
| **final-sota (shipped recipe, FP8 KV) ← RELEASE** | W4A16 | 81.25% | 91.25% | 39/40 | 37/40 | 17/20 |

Notes: 5 of the release run's 8 GSM8K exact-fails are decimal-format artifacts
("26.00" vs "26") — hence the numeric-normalized column. Engine A/B
(v0.27.1 vs rc0) and KV A/B (BF16 vs FP8) are quality-neutral on the final
checkpoint. Full per-run rows: `*.rows.jsonl` referenced in the campaign log.

### Serving gates on the release profile (MTP K*=3 + FP8 KV)

| Gate | Result |
|---|---|
| Semantic 10-check | PASS (AR / K2 / K3 / fp8-KV) |
| Sanity suite | 8/8 (qwen3_xml tool-calls, 2.8K long-gen, deterministic) |
| MTP ladder | AR 11.35 → K2 21.67 → **K3 22.36 tok/s** (1.97×; acceptance 2.5-2.7) |
| Dedicated c1 (2048 tok) | 27.8-28.1 tok/s median (2.45-2.48× AR) |
| Concurrency ladder | c1 19.2 / c2 32.0 / c4 44.6 / c8 84.3 tok/s — 17/17 zero-error |
| NIAH @ 262,144 | 8/8 PASS (5 positions @ ~247.7K actual + 8K/32K/131K ladder) |

## Known issues

- FP8 KV arithmetic defect — ROOT-CAUSED AND FIXED. A ModelOpt export with
  `kv_cache_quant_algo: None` served with the runtime fp8 flag takes a broken
  generic path (deterministic 19×23 → "417"). The checkpoint flag `"FP8"`
  routes KV through `ModelOptKVCacheMethod` and is exact — this checkpoint
  ships the flag; serve with `--kv-cache-dtype fp8`. BF16 KV
  (`--kv-cache-dtype auto`) also works and is quality-neutral but halves KV
  capacity. Upstream-report candidate; evidence in this repo.

## License

MIT for scripts/docs. Model weights follow the upstream Qwen license; weights
are not redistributed in this repo.

## Final checkpoint (2026-08-16)

`final-sota-nvidia-recipe` — NVIDIA shipped qwen3_5 recipe verbatim
(`w4a16_nvfp4-fp8_attn-kv_fp8_cast`): 193 W4A16_NVFP4 (MLP+lm_head, g16) +
208 FP8 attn/GDN + 257 BF16, `max` algorithm, ModelOpt 0.46.0rc1, calibration
= documented combo (cnn_dailymail + Nemotron-Post-Training-Dataset-v2) scaled
to 2048 packed rows @1024 via get_dataset_dataloader (pack, left-pad).
Shard SHAs in `final-sota-shards.sha256`. Served on vLLM v0.27.2rc0-sm121,
FP8 KV (checkpoint `kv_cache_quant_algo=FP8` flag; see Known issues), eager.
