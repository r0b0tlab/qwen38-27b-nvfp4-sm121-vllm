# Qwen3.8-27B NVFP4 on NVIDIA DGX Spark (GB10 / SM121)

Click-run package for serving the published NVFP4+MTP checkpoint on a single
DGX Spark. Weights live on Hugging Face. This repo is scripts, profiles, and
the exact serve flags that produced the numbers below.

## What you need

| Piece | Where |
|---|---|
| Checkpoint (all **four** shards) | [`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121) |
| Runtime image | `ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121` |
| Optional DSpark draft | [`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) + `scripts/adapt_dspark_draft.py` |

If your HF tree only has `model-00004-of-00004.safetensors`, it is incomplete.
That was a publication bug (hardlinked body shards were omitted). Re-pull
until all four files below exist and match `final-sota-shards.sha256`.

```
4208cd3b…  model-00001-of-00004.safetensors   (~9.3 GiB)
024111b9…  model-00002-of-00004.safetensors   (~9.3 GiB)
927ee343…  model-00003-of-00004.safetensors   (~1.1 GiB)
47202b11…  model-00004-of-00004.safetensors   (~0.8 GiB, BF16 MTP head)
```

Do **not** point the server at a sibling NVFP4 tree and then mount only shard 4.
That changes the kernel path and is not this release.

## Fastest path (MTP, the published production profile)

```bash
huggingface-cli download r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 \
  --local-dir ~/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121
sha256sum -c final-sota-shards.sha256

docker pull ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121

MODEL_DIR=~/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 \
  bash scripts/serve.sh mtp

# wait until /health is 200, then:
python3 scripts/run_semantic_gate.py \
  --base-url http://127.0.0.1:8000 \
  --output /tmp/qwen38-semantic.json
```

The canary that must pass is `19 × 23 → 437`. If you get `417`, the checkpoint
is missing `kv_cache_quant_algo: "FP8"` or you dropped `--kv-cache-dtype fp8`.

One-liner that downloads, launches MTP, and runs the canary:

```bash
bash scripts/click_run_mtp.sh
```

## Why stock `vllm-nightly` is ~27% slower

These numbers were taken on the **source-built SM121 wheel**
`v0.27.2rc0-sm121` (`7f7a32c`, GHCR digest `sha256:5bd3f329…`,
local config `sha256:3dd1f94e…`), not on an untuned nightly. On stock
nightly you will typically see:

```
[AutoTuner] No tuned config covers fp4_gemm ... falling back tactic=-1
```

That fallback is the gap. Independent reproduction on stock nightly of
**AR 11.30 vs our 11.35** and **MTP 20.4 (1.80×)** vs our **27.8 (2.45×)**
is the expected shape: the base model matches, the spec path does not,
because the published image carries the SM121 FP4 GEMM tune.

| Stack | AR c1 | MTP K=3 dedicated c1 | notes |
|---|---:|---:|---|
| This image (`v0.27.2rc0-sm121`) | 11.35 | **27.8–28.1** | published claim |
| Stock nightly / untuned fp4_gemm | ~11.3 | ~20.4 | independent repro, expected |

Use the GHCR image. Do not expect the 2.45× number from an untuned nightly.

## Profiles

`scripts/serve.sh` accepts one argument. Every profile keeps
`--kv-cache-dtype fp8 --enforce-eager --no-enable-prefix-caching`.

| Profile | Extra flags | Use |
|---|---|---|
| `mtp` (default) | MTP K=3 + `qwen3_xml` tools | production / ≥c8 throughput |
| `ar` | no spec | baseline |
| `dspark` | external draft, K=7 | c1–c4 latency / thinking-on |
| `long` | `--max-model-len 262144 --gpu-memory-utilization 0.85 --max-num-batched-tokens 8192` | NIAH / 262K |

```bash
MODEL_DIR=~/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 bash scripts/serve.sh ar
MODEL_DIR=... bash scripts/serve.sh mtp
DRAFT_DIR=~/models/r0b0tlab/Qwen3.8-27B-DSpark-adapted MODEL_DIR=... bash scripts/serve.sh dspark
MODEL_DIR=... bash scripts/serve.sh long
```

## DSpark that actually returns tokens

DSpark on this engine is **not** the stock RadixArk / SpecForge tree and it
is **not** SGLang. The working path is vLLM `method=dspark` after rewriting
the draft config to the `Qwen3DSparkModel` contract (PR #47808).

```bash
huggingface-cli download RadixArk/Qwen3.8-27B-DSpark \
  --local-dir ~/models/RadixArk/Qwen3.8-27B-DSpark

python3 scripts/adapt_dspark_draft.py \
  --src ~/models/RadixArk/Qwen3.8-27B-DSpark \
  --out ~/models/r0b0tlab/Qwen3.8-27B-DSpark-adapted

DRAFT_DIR=~/models/r0b0tlab/Qwen3.8-27B-DSpark-adapted \
MODEL_DIR=~/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 \
  bash scripts/serve.sh dspark
```

What the adapter changes (weights untouched):

- `architectures`: `DSparkDraftModel` → `Qwen3DSparkModel`
- `model_type`: `dspark` → `qwen3`
- `dspark_target_layer_ids`: `[4, 16, 28, 40, 52]`
- `n_predict` / `dspark_block_size`: `7`
- `dflash_config.projector_type`: `dspark`

A raw SpecForge draft on this image hangs after FlashInfer autotune
(`dspark_worker_v2.py`). That hang is the unadapted config, not a missing
backend. If `/health` is up but a 20-token request never returns, check
`docker logs` for the draft architecture string — it must be `Qwen3DSparkModel`.

DSpark numbers on **this** image / eager / W4A16 target:

| Level | MTP K=3 | DSpark K=7 |
|---|---:|---:|
| Dedicated c1 (2048) | 27.83 | **28.46** (best 30.43) |
| c1 256 | **19.22** | 16.05 |
| c4 | 34.61 | **43.88** |
| c8 | **82.89** | 61.53 |
| thinking-on accept length | — | 3.5 |

Think-on matched `run_perf_suite.sh` (same 512→2048 / 1024→256 protocol,
`chat_template_kwargs.enable_thinking=true`):

| | dedicated c1 med | ladder best c1 / c2 / c4 / c8 |
|---|---:|---|
| vLLM MTP K3 think-off | 27.83 | 19.24 / 32.00 / 34.61 / 82.89 |
| vLLM MTP K3 think-on | **29.12** | 22.57 / 37.46 / 60.22 / 84.07 |
| vLLM DSpark K7 think-off | 28.46 | 16.05 / 28.47 / 43.88 / 61.53 |
| vLLM DSpark K7 think-on | 28.65 | 16.13 / 28.63 / 44.63 / 62.23 |

DSpark quality-200 (think-off, same `artifacts/quality-200.jsonl` sha256
`ca35650e…`, client e2e tok/s = completion_tokens / wall):

| Family | score | mean | median | top |
|---|---|---:|---:|---:|
| GSM8K flex / numeric | 66/80 = 82.5% · 74/80 = 92.5% | 33.2 | 32.7 | 41.3 (`gsm8k-010`) |
| HumanEval | 39/40 | 51.3 | 52.0 | **57.5** (`humaneval-031`) |
| IFEval | 37/40 | 15.5 | 14.3 | 32.9 (`ifeval-027`) |
| Agentic | **19/20** | 37.7 | 37.9 | 44.8 (`agentic-08`) |
| Hard reasoning | 20 written, not auto-graded | 29.6 | 29.5 | 39.2 (`hard-04`) |
| All 200 | — | 33.4 | 32.9 | **57.5** |

Token-weighted 29.0 tok/s (75.5k tokens / 43.4 min). File:
`quality-200-vllm-dspark.json`. Matched SGLang DSpark scores on GSM8K /
HumanEval / IFEval; SGLang is faster (mean 38.1, peak 66.8); vLLM DSpark
wins agentic 19/20 vs 18/20.

SGLang native path (same checkpoint, official cookbook image):
[`r0b0tlab/qwen38-27b-nvfp4-sm121-sglang`](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang).
SGLang's current production profile there is **DFlash2 K8** (z-lab draft):
dedicated c1 **28.38**, r0b0bench core-subset 11/11 PASS with full-262144
NIAH 3/3. SGLang EAGLE think-off still wins the concurrent ladder
(c8 **123.90**). vLLM MTP think-on wins dedicated c1 (**29.12**).

## Quant / serve facts that matter

- Recipe: ModelOpt 0.46.0rc1 shipped `qwen3_5` map — 193 W4A16_NVFP4 + 208 FP8 + 257 BF16 + 15 BF16 MTP tensors.
- Serve **must** keep `--kv-cache-dtype fp8`. The checkpoint ships `kv_cache_quant_algo: "FP8"`. Flag-less + runtime fp8 = `19×23 → 417`.
- No prefix cache on this hybrid GDN family.
- No `--reasoning-parser qwen3` (this family has no `<think>` special tokens). Use `chat_template_kwargs={"enable_thinking": false}`.
- Quality on this harness, thinking off, temp 0: MTP/fp8kv GSM8K flex 81.25% / numeric-norm 91.25% / HumanEval 39/40 / IFEval 37/40 / agentic 17/20. DSpark K7 on the same 200: GSM8K flex 82.5% / numeric-norm 92.5% / HumanEval 39/40 / IFEval 37/40 / agentic 19/20. NIAH 8/8 @ 262,144.

## Build the runtime yourself (optional)

Only needed if you do not pull GHCR.

```bash
# inside docker/ on an aarch64 CUDA 13 host
# 1) build the v0.27.2rc0 wheel at 7f7a32c with TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4
# 2) layer it:
docker build -f docker/Dockerfile.rc0 \
  --build-arg WHEEL=vllm-0.27.2rc0-cp312-cp312-linux_aarch64.whl \
  -t ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121 .
```

`docker/Dockerfile.rc0` is in this repo. The published image is the claim
runtime; a self-built wheel is only equivalent if it is the same source SHA
and the same SM121 arch list.

## License

MIT for scripts/docs. Weights follow the upstream Qwen license and are not
stored in this git tree.
