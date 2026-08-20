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
| Optional DFlash2 draft (K=8) | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) + overlay image from `docker/Dockerfile.dflash2` |

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

## Quickstart (MTP — no extra downloads)

MTP is the quickest way to a working server: the MTP head is already inside
the checkpoint and the profile runs on the published image with no extra
downloads. It is not the fastest profile at every concurrency point — see
[Which profile should I run?](#which-profile-should-i-run).

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

## Which profile should I run?

There is no single fastest profile — it depends on where you sit on the
concurrency curve:

- **≥c8 batch throughput** → `mtp` (c8 82.89 tok/s, best dedicated c1 27.8;
  also the best dedicated c1 with thinking on: 29.12)
- **c1–c6 latency / mixed concurrency, and full-262K NIAH** → `dflash2`
  (r0b0bench ladder c1 67.1 / c2 121.5 / c4 211.5 / c6 279.2 agg tok/s;
  NIAH 3/3 @ 262,144; needs the overlay image + 3.6 GB z-lab draft)
- **c1–c4 with thinking on** → `dspark` (c4 43.88, thinking-on accept 3.5;
  needs the adapted draft)
- **Baseline / no speculation** → `ar` (11.35 c1)

Note the two measurement methodologies and never mix them: dedicated-c1 /
ladder figures come from `run_perf_suite.sh`; the c1/c2/c4/c6 ladder in the
DFlash2 table is r0b0bench aggregate client throughput.

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
| `mtp` (default) | MTP K=3 + `qwen3_xml` tools | ≥c8 throughput; quickest on-ramp (no extra downloads) |
| `ar` | no spec | baseline |
| `dspark` | external draft, K=7 | c1–c4 latency / thinking-on |
| `dflash2` | external z-lab draft, K=8, V2 runner, autotune/JIT/CuteDSL warmup off, overlay image | strongest low–mid concurrency + full-262K NIAH (see below) |
| `long` | `--max-model-len 262144 --gpu-memory-utilization 0.85 --max-num-batched-tokens 8192` | NIAH / 262K (no spec) |

```bash
MODEL_DIR=~/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 bash scripts/serve.sh ar
MODEL_DIR=... bash scripts/serve.sh mtp
DRAFT_DIR=~/models/r0b0tlab/Qwen3.8-27B-DSpark-adapted MODEL_DIR=... bash scripts/serve.sh dspark
DRAFT_DIR=~/models/z-lab/Qwen3.8-27B-DFlash2 MODEL_DIR=... IMAGE=qwen38-27b-vllm-dflash2-sm121:0.1.1 bash scripts/serve.sh dflash2
MODEL_DIR=... bash scripts/serve.sh long
```

# DFlash2 speculative decoding (K=8)

DFlash2 is the strongest spec path on this engine at low–mid concurrency.
It is **not** in the published wheel — serve it with the overlay image built
from this repo (`docker/Dockerfile.dflash2`). The overlay layers surgical
hunks from [vllm-project/vllm#52816](https://github.com/vllm-project/vllm/pull/52816)
@ `19c93519` onto the published `v0.27.2rc0-sm121` wheel (no wheel rebuild,
FP4 GEMM tune intact) plus four SM121 fixes the stock PR needs here:

1. **Quantized target LM head** — the stock PR rejects any
   non-`UnquantizedEmbeddingMethod` head; our target ships `W4A16_NVFP4`
   on `lm_head`. The overlay routes through `lm_head.quant_method.apply`
   (the same path `LogitsProcessor` uses). tp=1 only.
2. **V2 runner force** — `Qwen3_5ForConditionalGeneration` is not a
   default-V2 architecture; without the force, DFlash2 silently drafts as
   DFlash1 and the candidate selector never runs.
3. **Warmup is not the product path on SM121** — flashinfer autotune's
   dummy metadata is not the DFlash `1+K` query layout, and the
   memory-profile dummy run faults the selector GEMM
   (`CUBLAS_STATUS_INTERNAL_ERROR`). The dflash2 profile serves with
   `--no-enable-flashinfer-autotune` and jit/cutedsl warmup off, and the
   speculator skips the selector when `attn_metadata is None`.
4. **Selector-walk index clamps** — the DFlash2 selector-walk Triton kernel
   uses `BLOCK_K` (next power of two ≥ `selector_top_k`) as the
   losing-lane sentinel. All-masked / NaN score rows make every lane lose,
   `index == BLOCK_K`, and the kernel loads one past the candidate row.
   The garbage draft ids then hit the target `embed_tokens`, which at tp=1
   does not mask out-of-range ids — a CUDA device-side assert
   (`vectorized_gather_kernel`) minutes into any concurrent mixed
   tool-call batch (BFCL at C≥2). The overlay clamps the walk index to
   `top_k - 1`, NaN-sanitizes + clamps the lm-head top-k ids, and clamps
   candidate ids before the codebook gather. Verified: official BFCL-MT at
   4 threads ran 8h36 with zero device asserts (previously died in ~6 min).

The draft is the z-lab release; `is_causal: false` wins over the SWA
layer-type default (non-causal FlashInfer for the draft).

```bash
# draft (3.6 GB)
huggingface-cli download z-lab/Qwen3.8-27B-DFlash2 \
  --local-dir ~/models/z-lab/Qwen3.8-27B-DFlash2

# build the overlay image (takes ~30 s; base is the published GHCR image)
docker build -f docker/Dockerfile.dflash2 -t qwen38-27b-vllm-dflash2-sm121:0.1.1 docker/

DRAFT_DIR=~/models/z-lab/Qwen3.8-27B-DFlash2 \
MODEL_DIR=~/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 \
IMAGE=qwen38-27b-vllm-dflash2-sm121:0.1.1 \
  bash scripts/serve.sh dflash2
```

One-liner that downloads both trees, builds the overlay, launches, and
gates: `bash scripts/click_run_dflash2.sh`.

DFlash2 K8 vs the other profiles (same 1×GB10, r0b0bench core-subset,
think-off; vLLM rows from the overlay image `0.1.1`):

| | GSM8K | HE@1 | QA | IFEval | BFCL-MT | ASTµ | NIAH (262K) | c1 / c2 / c4 / c6 agg tok/s | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| vLLM **DFlash2 K8** | 0.870 | 0.890 | 0.963 | 0.825 | 0.565 | 0.270 | PASS 3/3 | 67.1 / 121.5 / 211.5 / 279.2 | 259.1 |
| SGLang DFlash2 K8 (sibling repo) | 0.865 | 0.872 | 0.963 | 0.820 | 0.690 | 0.273 | PASS 3/3 | 68.6 / 124.3 / 212.0 / 276.4 | 214.6 |

NIAH depths 65,472 / 130,944 / 235,699, all `finish=stop`. vLLM prefill in
its row is an e2e wall proxy (~826 tok/s for 22.7k-prompt requests), not
the pure-prefill `run_perf_suite` methodology behind SGLang's 22,663 —
do not compare those two numbers. IFEval is the lightweight constraint
scorer. Ledger: `r0b0bench` entry
`qwen38-27b-nvfp4-vllm-dflash2-k8-core-subset-20260820`.

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
NIAH 3/3. **This repo now serves DFlash2 K8 too** (overlay image, section
above) — quality parity within noise on GSM8K/QA/AST/NIAH, concurrency
ladder within ~3%. SGLang EAGLE think-off still wins the concurrent ladder
(c8 **123.90**). vLLM MTP think-on wins dedicated c1 (**29.12**).

## Quant / serve facts that matter

- Recipe: ModelOpt 0.46.0rc1 shipped `qwen3_5` map — 193 W4A16_NVFP4 + 208 FP8 + 257 BF16 + 15 BF16 MTP tensors.
- Serve **must** keep `--kv-cache-dtype fp8`. The checkpoint ships `kv_cache_quant_algo: "FP8"`. Flag-less + runtime fp8 = `19×23 → 417`.
- No prefix cache on this hybrid GDN family.
- No `--reasoning-parser qwen3` (this family has no `<think>` special tokens). Use `chat_template_kwargs={"enable_thinking": false}`.
- Quality on this harness, thinking off, temp 0: MTP/fp8kv GSM8K flex 81.25% / numeric-norm 91.25% / HumanEval 39/40 / IFEval 37/40 / agentic 17/20. DSpark K7 on the same 200: GSM8K flex 82.5% / numeric-norm 92.5% / HumanEval 39/40 / IFEval 37/40 / agentic 19/20. NIAH 8/8 @ 262,144. DFlash2 K8 (overlay 0.1.1) r0b0bench core-subset 11/11: GSM8K 0.870 / HE 0.890 / QA 0.9625 / IFEval 0.825 / NIAH 3/3 @ 262,144.

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
