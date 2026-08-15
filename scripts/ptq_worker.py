#!/usr/bin/env python3
"""Qwen3.8-27B NVFP4 PTQ worker (dense Qwen3.5-family hybrid GDN VLM).

Pattern: Muse Glimmer dense-VLM precedent — quantize model.model.language_model
submodel only; vision tower / MTP / GDN auxiliaries stay BF16; export full model.
Recipe contract lives in the campaign config YAML (converted here to mtq cfg).

attempt18 (mixed_hess): same layer map as mixed (W4A4 NVFP4 MLP+lm_head,
FP8 attn/GDN with STATIC input quantizers) + algorithm local_hessian
(fp8_scale_sweep) per the shipped nvfp4_w4a4_weight_local_hessian preset.
Census contract (from official nvidia/Qwen3.6-27B-NVFP4 control, verified
on node2): 193 NVFP4 + 208 FP8 + FP8 KV.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForImageTextToText, AutoTokenizer

SRC = os.environ.get("Q38_SRC", "/src")
OUT = os.environ.get("Q38_OUT", "/out")
LOG = os.environ.get("Q38_LOG", "/dev/stderr")
ATTEMPT_DIR = os.environ.get("Q38_ATTEMPT", "/attempt")
DRY = os.environ.get("Q38_DRY", "0") == "1"

t0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time()-t0:8.1f}s] {msg}"
    print(line, flush=True)


def fail(msg: str) -> None:
    log(f"FATAL: {msg}")
    (Path(ATTEMPT_DIR) / "FAIL").write_text(msg + "\n")
    sys.exit(1)


# --- 1) Build the quant config from the recipe contract (dense variant) ---
# NVFP4 W4A4: mlp gate/up/down + lm_head; FP8: self_attn + linear_attn big projections;
# excluded: *visual*, *vision_tower*, *mtp*, linear_attn.in_proj_a/b, norms, embeddings.
cfg = mtq.NVFP4_DEFAULT_CFG
quant_cfg_list = cfg["quant_cfg"]
# ensure list form (0.45 uses list-of-dicts)
if not isinstance(quant_cfg_list, list):
    fail(f"unexpected quant_cfg form: {type(quant_cfg_list)}")

# Use mtq presets for exactness
fp8_preset = mtq.FP8_DEFAULT_CFG["quant_cfg"]
# FP8 entries are dicts with quantizer_name + enable/cfg; find the default fp8 cfg shape
fp8_weight_entry = next(e for e in fp8_preset if e.get("quantizer_name") == "*weight_quantizer")
fp8_input_entry = next(e for e in fp8_preset if e.get("quantizer_name") == "*input_quantizer")

# CRITICAL: static FP8 input quantizers force calibration so input_scale
# tensors are exported — vLLM's ModelOpt-FP8 kernel requires them (official
# checkpoint ships them; uncalibrated export omits them -> garbage logits).
# Schema-legal marker: need_calibration() treats type != "dynamic" as static.
static_fp8_input_cfg = dict(fp8_input_entry.get("cfg") or {})
static_fp8_input_cfg["type"] = "static"


def add_nvfp4(pattern: str) -> None:
    nvfp4_w = next(e for e in cfg["quant_cfg"] if e.get("quantizer_name") == "*weight_quantizer")
    nvfp4_i = next(e for e in cfg["quant_cfg"] if e.get("quantizer_name") == "*input_quantizer")
    quant_cfg_list.append({"quantizer_name": pattern, **{k: v for k, v in nvfp4_w.items() if k != "quantizer_name"}})
    # input quantizer pattern derived by replacing weight_quantizer with input_quantizer
    quant_cfg_list.append({"quantizer_name": pattern.replace("*weight_quantizer*", "*input_quantizer*"),
                           **{k: v for k, v in nvfp4_i.items() if k != "quantizer_name"}})


def add_fp8(pattern_w: str, pattern_i: str, static_input: bool = False) -> None:
    quant_cfg_list.append({"quantizer_name": pattern_w, **{k: v for k, v in fp8_weight_entry.items() if k != "quantizer_name"}})
    input_cfg = dict(static_fp8_input_cfg) if static_input else dict(fp8_input_entry.get("cfg") or {})
    quant_cfg_list.append({"quantizer_name": pattern_i, "cfg": input_cfg,
                           **{k: v for k, v in fp8_input_entry.items() if k not in ("quantizer_name", "cfg")}})


# Disable-all base FIRST (later entries override) — without this,
# NVFP4_DEFAULT_CFG silently quantizes attn/GDN projections too.
quant_cfg_list.insert(0, {"quantizer_name": "*quantizer*", "enable": False})

PROFILE = os.environ.get("Q38_PROFILE", "mixed")  # mixed | mlp_only | no_lmhead_mixed | w4a16 | mixed_hess

if PROFILE in ("w4a16", "mlp_only"):
    # attempt8: EXPLICIT layer map from an EMPTY base (no preset defaults —
    # presets re-enable everything incl. lm_head, which corrupts logits).
    # Mirrors official nvidia/Qwen3.6-27B-NVFP4: W4A16_NVFP4 MLPs + FP8 attn/GDN.
    w4a16_cfg = next(e for e in mtq.W4A16_NVFP4_CFG["quant_cfg"]
                     if e.get("quantizer_name") == "*weight_quantizer")["cfg"]
    quant_cfg_list = [
        # disable everything by default
        {"quantizer_name": "*weight_quantizer", "enable": False},
        {"quantizer_name": "*input_quantizer", "enable": False},
        {"quantizer_name": "*bmm_quantizer", "enable": False},
        # W4A16 NVFP4 on MLP weights (official shape; group 16)
        *[{"quantizer_name": f"*mlp*{p}*weight_quantizer", "cfg": dict(w4a16_cfg)}
          for p in ("gate_proj", "up_proj", "down_proj")],
        # FP8 weights on attn/GDN projections (official shape)
        *[{"quantizer_name": w, **{k: v for k, v in fp8_weight_entry.items() if k != "quantizer_name"}}
          for w in ("*self_attn*weight_quantizer", "*linear_attn.in_proj_qkv*weight_quantizer",
                    "*linear_attn.in_proj_z*weight_quantizer", "*linear_attn.out_proj*weight_quantizer")],
        # FP8 STATIC inputs on attn/GDN projections (official shape)
        *[{"quantizer_name": i, "cfg": dict(static_fp8_input_cfg)}
          for i in ("*self_attn*input_quantizer", "*linear_attn.in_proj_qkv*input_quantizer",
                    "*linear_attn.in_proj_z*input_quantizer", "*linear_attn.out_proj*input_quantizer")],
    ]
    cfg = {"algorithm": {"method": "max"}, "quant_cfg": quant_cfg_list}
elif PROFILE in ("no_lmhead_mixed", "mixed", "mixed_hess"):
    static_in = PROFILE == "mixed_hess"
    for p in ("*mlp*gate_proj*weight_quantizer*", "*mlp*up_proj*weight_quantizer*", "*mlp*down_proj*weight_quantizer*"):
        add_nvfp4(p)
    for w, i in (
        ("*self_attn*weight_quantizer", "*self_attn*input_quantizer"),
        ("*linear_attn.in_proj_qkv*weight_quantizer", "*linear_attn.in_proj_qkv*input_quantizer"),
        ("*linear_attn.in_proj_z*weight_quantizer", "*linear_attn.in_proj_z*input_quantizer"),
        ("*linear_attn.out_proj*weight_quantizer", "*linear_attn.out_proj*input_quantizer"),
    ):
        add_fp8(w, i, static_input=static_in)
    if PROFILE in ("mixed", "mixed_hess"):
        add_nvfp4("*lm_head*weight_quantizer*")
    if PROFILE == "mixed_hess":
        # shipped nvfp4_w4a4_weight_local_hessian preset: method local_hessian + fp8_scale_sweep
        cfg = {"algorithm": {"method": "local_hessian", "fp8_scale_sweep": True},
               "quant_cfg": quant_cfg_list}
        log("algorithm: local_hessian fp8_scale_sweep=True (attempt18)")
    else:
        cfg = {"algorithm": {"method": "max"}, "quant_cfg": quant_cfg_list}

# NVFP4 KV cache (user-requested for the campaign profile):
# merge NVFP4_KV_CFG quantizer entries into the config (per-entry cfg merge).
if os.environ.get("Q38_KV", "fp8") == "nvfp4":
    kv_entries = [e for e in mtq.NVFP4_KV_CFG["quant_cfg"] if "kv" in e.get("quantizer_name", "")]
    quant_cfg_list.extend(kv_entries)
    log(f"NVFP4 KV entries appended: {len(kv_entries)}")

# Exclusions (late entries override)
for pattern in ("*visual*", "*vision_tower*", "*mtp*", "*linear_attn.in_proj_a*", "*linear_attn.in_proj_b*"):
    quant_cfg_list.append({"quantizer_name": pattern, "enable": False})

log("quant cfg built")

# --- 2) Load the full VLM BF16 onto the GPU ---
log(f"loading {SRC} (bf16, device 0)...")
try:
    model = AutoModelForImageTextToText.from_pretrained(
        SRC, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
except Exception as exc:
    fail(f"model load failed: {exc!r}")
log(f"loaded; class={type(model).__name__}")

# locate language_model submodule (Qwen3_5ForConditionalGeneration: model.model.language_model)
lm = getattr(model.model, "language_model", None) or getattr(model, "language_model", None)
if lm is None:
    # dump top-level module names for diagnosis
    names = [n for n, _ in model.named_children()]
    fail(f"no language_model submodule; top children: {names}")
log(f"language_model submodule: {type(lm).__name__}")

# attempt18: quantize the FULL model (official recipe contract — lm_head lives
# OUTSIDE language_model in the qwen3_5 family, and the official control
# quantizes it: 193 NVFP4 = 192 mlp + lm_head). Vision/MTP stay BF16 via the
# *visual* / *vision_tower* / *mtp* exclusions (same patterns as the shipped
# qwen3_5 w4a16_nvfp4-fp8_attn-kv_fp8_cast.quant_cfg.yaml).
QUANT_TARGET = model

# --- 3) Quantize (full model; vision/MTP excluded via patterns) ---
# OFFICIAL FLOW (hf_ptq.py): create_forward_loop(dataloader=...) -> plain callable,
# one forward per invocation; ModelOpt drives it during mtq.quantize(..., forward_loop=).
# A python generator is called ONCE and never iterated (attempt13 no-op lesson).
fwd_loop = None
calib_rows = 0
if os.environ.get("Q38_CALIB", "0") == "1":
    from transformers import AutoTokenizer
    from modelopt.torch.utils.dataset_utils import create_forward_loop
    calib_path = os.environ.get("Q38_CALIB_FILE", "/calib/calibration.jsonl")
    calib_max = int(os.environ.get("Q38_CALIB_ROWS", "512"))
    prompts = [json.loads(l)["text"][:2000] for l in open(calib_path)][:calib_max]
    calib_rows = len(prompts)
    tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
    device = next(model.parameters()).device
    calib_batches = []
    for p in prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=512)
        calib_batches.append({k: v.to(device) for k, v in enc.items()})
    fwd_loop = create_forward_loop(dataloader=calib_batches)
    log(f"calib loop ready: {calib_rows} batches (create_forward_loop)")

log("mtq.quantize(model, cfg, forward_loop=loop)...")
try:
    mtq.quantize(QUANT_TARGET, cfg, forward_loop=fwd_loop)
except Exception as exc:
    fail(f"quantize failed: {exc!r}")
log("quantized")


# --- 3b) Input-scale calibration for static FP8 inputs ---
# local_hessian refines WEIGHT scales only (inputs untouched). Static FP8 input
# quantizers still need amax from the calibration set (attempt2-8 lesson:
# uncalibrated static inputs export without input_scale -> garbage logits).
# Guard: max_calibrate must NOT clobber the local_hessian-optimized weight
# scales — snapshot them and hard-fail on any change.
def weight_scale_snapshot(module) -> dict:
    snap = {}
    for name, mod in module.named_modules():
        wq = getattr(mod, "weight_quantizer", None)
        if wq is None:
            continue
        amax = getattr(wq, "_amax", None)
        if isinstance(amax, torch.Tensor):
            snap[name] = amax.detach().clone()
    return snap


needs_input_calib = PROFILE in ("mixed_hess", "mixed", "no_lmhead_mixed", "w4a16", "mlp_only")
if fwd_loop is not None and needs_input_calib:
    snap_before = weight_scale_snapshot(QUANT_TARGET) if PROFILE == "mixed_hess" else None
    try:
        log("mtq.calibrate(model, 'max', fwd_loop)...")
        mtq.calibrate(QUANT_TARGET, "max", fwd_loop)
        log("calibrate('max') complete — input scales collected")
    except Exception as exc:
        fail(f"calibrate failed: {exc!r}")
    if snap_before is not None:
        clobbered = []
        for name, before in snap_before.items():
            mod = QUANT_TARGET.get_submodule(name)
            after = getattr(getattr(mod, "weight_quantizer", None), "_amax", None)
            if not isinstance(after, torch.Tensor) or not torch.equal(before, after):
                clobbered.append(name)
        if clobbered:
            fail(f"max calibrate clobbered {len(clobbered)} local_hessian weight scales: {clobbered[:10]}")
        log(f"clobber guard: {len(snap_before)} weight scales unchanged after max-calibrate")

# --- 3c) Census (attempt18 contract: 193 NVFP4 + 208 FP8) ---
exp_nvfp4 = int(os.environ.get("Q38_EXPECT_NVFP4", "193"))
exp_fp8 = int(os.environ.get("Q38_EXPECT_FP8", "208"))
census = {"total_linears": 0, "nvfp4": 0, "fp8": 0, "bf16": 0}


def classify_wq(wq):
    """Return 4/8/None for a weight quantizer (None = unknown).
    0.46: num_bits is an (exp, mantissa) tuple — FP8=(4,3) -> 8, FP4=(2,1) -> 4.
    """
    s = str(wq)
    nb = getattr(wq, "num_bits", None)
    if isinstance(nb, (tuple, list)):
        nb = 1 + sum(int(x) for x in nb)  # 1 sign + exp + mantissa
    if nb is None:
        m = re.search(r"\((\d+),\s*(\d+)\)\s*bit", s)
        if m:
            nb = 1 + int(m.group(1)) + int(m.group(2))
    if nb is None and ("block" in s or "Sequential" in type(wq).__name__):
        nb = 4  # StaticBlockScaleQuantizer -> NVFP4 block scales
    return nb


diag = []
for name, mod in QUANT_TARGET.named_modules():
    wq = getattr(mod, "weight_quantizer", None)
    if wq is None:
        continue
    census["total_linears"] += 1
    enabled = bool(getattr(wq, "is_enabled", True))
    if len(diag) < 10:
        diag.append(f"{name[:60]} | {type(wq).__name__} | enabled={enabled} | nb={getattr(wq, 'num_bits', 'NONE')} | {str(wq)[:50]}")
    if not enabled:
        census["bf16"] += 1
        continue
    nb = classify_wq(wq)
    if nb == 4:
        census["nvfp4"] += 1
    elif nb == 8:
        census["fp8"] += 1
    else:
        census["bf16"] += 1
        if len(diag) < 20:
            diag.append(f"UNCLASSIFIED {name[:60]} | {type(wq).__name__} | {str(wq)[:80]}")
for line in diag:
    log(f"census-diag: {line}")
log(f"census: {census}")
if census["nvfp4"] != exp_nvfp4 or census["fp8"] != exp_fp8:
    fail(f"census mismatch: got nvfp4={census['nvfp4']} fp8={census['fp8']}; expected {exp_nvfp4}/{exp_fp8}")

# --- 3d) DRY parse: audit and stop before export ---
if DRY:
    fp8_input_amax_ok = 0
    fp8_input_amax_missing = []
    for name, mod in QUANT_TARGET.named_modules():
        wq = getattr(mod, "weight_quantizer", None)
        if wq is None:
            continue
        if classify_wq(wq) != 8:
            continue
        iq = getattr(mod, "input_quantizer", None)
        amax = getattr(iq, "_amax", None) if iq is not None else None
        if isinstance(amax, torch.Tensor):
            fp8_input_amax_ok += 1
        else:
            fp8_input_amax_missing.append(name)
    audit = {
        "dry": True,
        "profile": PROFILE,
        "algorithm": cfg.get("algorithm"),
        "census": census,
        "calib_rows": calib_rows,
        "fp8_input_amax": {"ok": fp8_input_amax_ok, "missing": fp8_input_amax_missing[:10]},
        "elapsed_s": time.time() - t0,
    }
    (Path(ATTEMPT_DIR) / "DRY-AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    (Path(ATTEMPT_DIR) / "DRY-PASS").write_text(json.dumps(audit) + "\n")
    log("DRY-PASS sentinel written (no export)")
    sys.exit(0)

# --- 4) Export full model ---
log(f"export_hf_checkpoint -> {OUT}")
try:
    export_hf_checkpoint(model, export_dir=OUT)
except Exception as exc:
    fail(f"export failed: {exc!r}")
log("exported")

# --- 5) Post-export audit: lm_head presence + input_scale census ---
out = Path(OUT)
idx = out / "model.safetensors.index.json"
if idx.exists():
    weight_map = json.loads(idx.read_text())["weight_map"]
    has_lm_head = any("lm_head" in k for k in weight_map)
    n_shards = len(set(weight_map.values()))
    n_input_scale = sum(1 for k in weight_map if k.endswith("input_scale"))
    n_weight_scale = sum(1 for k in weight_map if k.endswith("weight_scale"))
    log(f"index: {len(weight_map)} tensors, {n_shards} shards, lm_head present={has_lm_head}; "
        f"weight_scale={n_weight_scale} input_scale={n_input_scale}")
    if n_input_scale < census["fp8"]:
        fail(f"input_scale audit: exported {n_input_scale} < {census['fp8']} FP8-class layers")
else:
    single = out / "model.safetensors"
    log(f"single-file export: {single.exists()}")

# Export completeness audit (attempt18 lesson): export_hf_checkpoint on the
# FULL model did NOT copy tokenizer/processor files (attempt15 lm-only export
# did). vLLM cannot serve without them — hard fail.
for required_file in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                      "vocab.json", "merges.txt"):
    if not (out / required_file).exists():
        log(f"export missing {required_file} — copying from source")
        src_f = Path(SRC) / required_file
        if not src_f.exists():
            fail(f"export missing {required_file} and source lacks it")
        import shutil
        shutil.copy2(src_f, out / required_file)
log("export completeness audit passed (tokenizer/processor files present)")

(Path(ATTEMPT_DIR) / "PASS").write_text(json.dumps({
    "elapsed_s": time.time() - t0,
    "profile": PROFILE,
    "algorithm": cfg.get("algorithm"),
    "src": SRC, "out": str(out),
    "census": census,
}, indent=2) + "\n")
log("DONE — PASS sentinel written")
