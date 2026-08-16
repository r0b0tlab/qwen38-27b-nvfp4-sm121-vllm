#!/usr/bin/env python3
"""Normalize a RadixArk / SpecForge DSpark draft so vLLM 0.27.2rc0 will load it.

The published RadixArk checkpoint uses architectures=["DSparkDraftModel"] and
model_type=dspark. The SM121 rc0 engine expects the Qwen3DSparkModel contract
(vLLM PR #47808). This script copies the draft tree and rewrites only
config.json. Weights are never mutated.

Usage:
  python3 scripts/adapt_dspark_draft.py \\
      --src ~/.cache/huggingface/hub/.../Qwen3.8-27B-DSpark \\
      --out ~/drafts/Qwen3.8-27B-DSpark-adapted
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TARGET_LAYER_IDS = [4, 16, 28, 40, 52]
BLOCK_SIZE = 7


def adapt_config(cfg: dict) -> dict:
    out = dict(cfg)
    out["architectures"] = ["Qwen3DSparkModel"]
    out["model_type"] = "qwen3"
    out["dspark_target_layer_ids"] = list(TARGET_LAYER_IDS)
    out["n_predict"] = BLOCK_SIZE
    out["dspark_block_size"] = BLOCK_SIZE
    out["block_size"] = BLOCK_SIZE
    out.setdefault("markov_rank", 256)
    out.setdefault("markov_head_type", "vanilla")
    out.setdefault("enable_confidence_head", True)
    out.setdefault("confidence_head_with_markov", True)
    dflash = dict(out.get("dflash_config") or {})
    dflash.setdefault("attention_mode", "gqa")
    dflash.setdefault("confidence_head_alpha", 1.0)
    dflash["confidence_head_with_markov"] = True
    dflash["enable_confidence_head"] = True
    dflash.setdefault("markov_head_type", "vanilla")
    dflash.setdefault("markov_rank", 256)
    dflash.setdefault("mask_token_id", 248077)
    dflash["projector_type"] = "dspark"
    dflash["target_layer_ids"] = list(TARGET_LAYER_IDS)
    out["dflash_config"] = dflash
    auto_map = dict(out.get("auto_map") or {})
    auto_map.setdefault("AutoModel", "dspark.DSparkDraftModel")
    out["auto_map"] = auto_map
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="RadixArk/Qwen3.8-27B-DSpark tree")
    parser.add_argument("--out", required=True, help="adapted draft output directory")
    args = parser.parse_args()
    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not (src / "config.json").is_file():
        raise SystemExit(f"missing {src / 'config.json'}")
    if not any((src / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")):
        raise SystemExit(f"missing draft weights under {src}")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    shutil.copytree(src, out, dirs_exist_ok=False)
    cfg = json.loads((out / "config.json").read_text())
    adapted = adapt_config(cfg)
    (out / "config.json").write_text(json.dumps(adapted, indent=1) + "\n")
    print(json.dumps({
        "src": str(src),
        "out": str(out),
        "architectures": adapted["architectures"],
        "model_type": adapted["model_type"],
        "dspark_target_layer_ids": adapted["dspark_target_layer_ids"],
        "n_predict": adapted["n_predict"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
