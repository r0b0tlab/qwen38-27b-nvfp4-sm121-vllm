#!/usr/bin/env python3
"""Fail-closed audit of the DFlash2 overlay on the published SM121 wheel."""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def _site() -> Path:
    import vllm

    return Path(vllm.__file__).resolve().parent


def main() -> int:
    import vllm

    errors: list[str] = []
    if vllm.__version__ != "0.27.2rc0":
        errors.append(f"wheel replaced: vllm.__version__={vllm.__version__!r}")

    try:
        from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP  # noqa: F401
    except Exception as exc:
        errors.append(f"qwen3_5_mtp import failed: {exc}")

    root = _site()
    registry = (root / "model_executor/models/registry.py").read_text()
    if '"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")' not in registry:
        errors.append("registry missing DFlash2DraftModel")

    spec_init = (root / "v1/worker/gpu/spec_decode/__init__.py").read_text()
    if "DFlash2DraftModel" not in spec_init or "DFlash2Speculator" not in spec_init:
        errors.append("init_speculator does not branch on DFlash2DraftModel")

    vllm_cfg = (root / "config/vllm.py").read_text()
    if "def _is_dflash2_draft" not in vllm_cfg:
        errors.append("use_v2_model_runner missing _is_dflash2_draft")
    if "if self._is_dflash2_draft():" not in vllm_cfg:
        errors.append("use_v2_model_runner does not force V2 for DFlash2")

    d2 = (root / "model_executor/models/qwen3_dflash2.py").read_text()
    if "DFlash2 requires an unquantized target LM head" in d2:
        errors.append("stock quantized-head reject still present")
    if "quant_method.apply" not in d2 and "apply(self.lm_head" not in d2:
        errors.append("quantized-head apply path missing")

    d1 = (root / "model_executor/models/qwen3_dflash.py").read_text()
    if "decoder_layer_cls" not in d1 or "model_cls" not in d1:
        errors.append("DFlash1 host missing decoder_layer_cls/model_cls hooks")

    if not (root / "v1/worker/gpu/spec_decode/dflash2/speculator.py").is_file():
        errors.append("dflash2 speculator missing")

    gdn = (root / "v1/attention/backends/gdn_attn.py").read_text()
    if "r0b0tlab/SM121" in gdn or 'spec_method == "dflash"' in gdn:
        errors.append(
            "GDN attention must be STOCK (the spec-decode reroute corrupted "
            "target verify steps; quality collapse 2026-08-20)"
        )

    speculator = (root / "v1/worker/gpu/spec_decode/dflash2/speculator.py").read_text()
    if "tl.minimum(index, top_k - 1)" not in speculator:
        errors.append("selector walk does not clamp BLOCK_K index")
    if "candidate_ids.clamp(0, self.vocab_size - 1)" not in speculator:
        errors.append("candidate_ids not clamped before codebook gather")

    # syntax of overlay modules
    for rel in (
        "model_executor/models/qwen3_dflash2.py",
        "v1/worker/gpu/spec_decode/dflash2/speculator.py",
    ):
        ast.parse((root / rel).read_text(), filename=rel)

    if errors:
        print("AUDIT FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("AUDIT PASS")
    print("vllm", vllm.__version__)
    print("DFlash2DraftModel registered")
    print("V2 force present")
    print("quantized-head apply path present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
