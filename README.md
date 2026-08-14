# Qwen3.8-27B NVFP4 on GB10 / SM121 (DGX Spark) — vLLM

[![status](https://img.shields.io/badge/status-CAMPAIGN__IN__PREPARATION-orange)]()
[![engine](https://img.shields.io/badge/vLLM-0.27.1-blue)]()

Reproducible serving + qualification package for **Qwen3.8-27B** (Qwen3.5-family hybrid
Gated-DeltaNet/Gated-Attention dense VLM, 262,144 native context) on a single
NVIDIA GB10 (SM121, aarch64, 128GB unified memory), node of a DGX Spark cluster.

> STATUS: campaign in preparation (2026-08-14). Release identity, image digests, and all
> metric fields below are placeholders until the admission gates pass. Nothing here is a
> measured claim yet.

## What this will contain

- `scripts/serve.sh` — containerized vLLM launcher with GB10 containment rules
- `scripts/audit_runtime.py` — fail-closed runtime audit (SM121 capability, CUDA 13,
  Qwen3_5 family modules, ModelOpt NVFP4 loader, no-fallback env)
- `scripts/run_semantic_gate.py` — deterministic semantic ladder (arithmetic, exact-string,
  word problem, code, determinism repeats)
- `scripts/run_long_generation.py` — sustained long generation gate
- `scripts/run_max_context_gate.py` — atomic full-context NIAH ladder: tokenize-verified
  depths 65,536 → 262,144, five needle positions, dual-code ordered retrieval,
  forced-512 continuation
- `results/` — raw evidence JSON per gate (added as gates pass)
- Reproducibility instructions with exact digests (added at publication)

## Model (pre-release facts)

| Field | Value |
|---|---|
| Architecture | Qwen3.5-family hybrid: 64 layers, per 16-block 3× Gated DeltaNet + 1× Gated Attention |
| Params | 27B dense (text) + vision encoder |
| Context | 262,144 native (YaRN extensible to 1M) |
| Vocab | 248,320 padded |
| MTP | trained (multi-token prediction) |
| Thinking | default on; `enable_thinking` / `preserve_thinking` / `reasoning_effort` kwargs |

## Engine identity (verified pre-release)

- Image: `vllm/vllm-openai:v0.27.1`
- Registry: `Qwen3_5ForConditionalGeneration`, `Qwen3_5MTP` present (live-verified)
- transformers 5.15.0 parses `rope_parameters` / `mrope_interleaved` (live-verified)
- Runtime audit: PASS on SM121 (see above scripts; re-run in-container at qualification)

## SM121 constraints applied (from prior Qwen3.5/3.6-family campaigns)

- FP8 KV cache production default (NVFP4 KV blocked on SM121)
- No prefix caching for hybrid GDN models
- No trtllm-gen NVFP4 attention on SM121 (no compatible cubins)
- `VLLM_HOST_IP=127.0.0.1` (avoids NCCL link-local init storms)
- `--enforce-eager` initially (SM121 codegen fragility, vllm#37431); relaxed only with evidence
- GPU memory utilization capped + hard container memory ceiling

## Credits

Qwen3.8-27B by the Qwen team (Alibaba). Serving via vLLM. FlashInfer kernels.
Benchmark harnesses adapted from the r0b0tlab SM121 campaign series.
