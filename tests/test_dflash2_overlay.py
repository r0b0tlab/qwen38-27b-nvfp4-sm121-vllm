#!/usr/bin/env python3
"""Static needles for the DFlash2 overlay (no GPU)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_overlay_files_exist() -> None:
    required = (
        "docker/Dockerfile.dflash2",
        "docker/overlay/vllm/model_executor/models/qwen3_dflash2.py",
        "docker/overlay/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py",
        "scripts/audit_dflash2_image.py",
        "scripts/click_run_dflash2.sh",
    )
    missing = [name for name in required if not (ROOT / name).is_file()]
    assert not missing, missing


def test_overlay_does_not_reject_quantized_head() -> None:
    text = (ROOT / "docker/overlay/vllm/model_executor/models/qwen3_dflash2.py").read_text()
    assert "DFlash2 requires an unquantized target LM head" not in text
    assert "apply(self.lm_head" in text or "apply(self.lm_head, hidden_states" in text


def test_overlay_forces_v2_and_speculator_branch() -> None:
    cfg = (ROOT / "docker/overlay/vllm/config/vllm.py").read_text()
    assert "def _is_dflash2_draft" in cfg
    assert "if self._is_dflash2_draft():" in cfg
    spec = (ROOT / "docker/overlay/vllm/v1/worker/gpu/spec_decode/__init__.py").read_text()
    assert "DFlash2DraftModel" in spec
    assert "DFlash2Speculator" in spec
    reg = (ROOT / "docker/overlay/vllm/model_executor/models/registry.py").read_text()
    assert "DFlash2DraftModel" in reg


def test_serve_script_has_dflash2_k8() -> None:
    text = (ROOT / "scripts/serve.sh").read_text()
    for needle in ("dflash2)", "DFlash2DraftModel", "num_speculative_tokens", "DFLASH_NUM_SPEC"):
        assert needle in text, needle
    assert 'DFLASH_NUM_SPEC="${DFLASH_NUM_SPEC:-8}"' in text
    # V2 env only for dflash2, not a global default-1
    assert 'VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"' in text
    assert text.count("VLLM_USE_V2_MODEL_RUNNER") >= 2


def test_selector_walk_clamps_block_k() -> None:
    spec = (
        ROOT / "docker/overlay/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"
    ).read_text()
    assert "tl.minimum(index, top_k - 1)" in spec
    assert "candidate_ids.clamp(0, self.vocab_size - 1)" in spec
    d2 = (ROOT / "docker/overlay/vllm/model_executor/models/qwen3_dflash2.py").read_text()
    assert "nan_to_num" in d2
    assert "ids.clamp_" in d2


def test_gdn_attention_is_stock() -> None:
    # The 2026-08-20 overlay tried routing DFlash spec tokens through the GDN
    # prefill kernel. That reroute applies to the TARGET's verify steps and
    # collapsed long-generation quality (GSM8K 0.295 vs SGLang 0.865). Stock
    # gdn_attn must never be overlaid again for this campaign.
    assert not (ROOT / "docker/overlay/vllm/v1/attention/backends/gdn_attn.py").is_file()


def test_click_run_dflash2_fail_closed() -> None:
    text = (ROOT / "scripts/click_run_dflash2.sh").read_text()
    assert "READY_TIMEOUT" in text
    assert "container died during startup" in text
    assert "z-lab/Qwen3.8-27B-DFlash2" in text
