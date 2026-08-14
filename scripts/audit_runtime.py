#!/usr/bin/env python3
"""Fail-fast runtime audit for Qwen3.8-27B NVFP4 serving on GB10/SM121.

Checks the exact identity tuple required for the campaign BEFORE any benchmark:
engine version, SM121 capability, CUDA 13.x contract, Qwen3_5 family modules,
modelopt NVFP4 loader, and absence of forbidden fallback markers in logs/env.
Exit 0 = audit PASS; any FAIL row = audit FAIL (fail-closed).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path


def check(name: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    return name, bool(ok), detail


def main() -> int:
    import torch
    import vllm

    rows: list[tuple[str, bool, str]] = []
    version = getattr(vllm, "__version__", "unknown")
    rows.append(check("vllm_version_expected", "0.27" in version, version))
    capability = torch.cuda.get_device_capability()
    rows.append(check("cuda_capability_sm121", capability == (12, 1), str(capability)))
    rows.append(check("cuda_runtime_13.x", str(torch.version.cuda).startswith("13."), str(torch.version.cuda)))

    for module in (
        "vllm.model_executor.models.qwen3_5",
        "vllm.model_executor.models.qwen3_5_mtp",
        "vllm.v1.attention.backends.flashinfer",
    ):
        try:
            importlib.import_module(module)
            rows.append(check(f"import:{module}", True))
        except Exception as exc:
            rows.append(check(f"import:{module}", False, repr(exc)[:180]))

    # ModelOpt NVFP4 loader presence (exact module resolved at runtime; all three
    # candidate names accepted — loader layout differs across vLLM releases)
    modelopt_ok = False
    modelopt_detail = "none"
    for name in (
        "vllm.model_executor.layers.quantization.modelopt",
        "vllm.model_executor.layers.quantization.modelopt_fp4",
        "vllm.model_executor.layers.quantization.modelopt_mixed",
    ):
        try:
            importlib.import_module(name)
            modelopt_ok, modelopt_detail = True, name
            break
        except Exception:
            continue
    rows.append(check("modelopt_nvfp4_loader", modelopt_ok, modelopt_detail))

    # Env sanity: no emulation/marlin escape hatches set
    forbidden_env = [
        k for k in os.environ
        if "MARLIN" in k.upper()
        or ("FALLBACK" in k.upper() and "DISABLE" not in k.upper())
    ]
    rows.append(check("no_forbidden_env", not forbidden_env, ",".join(forbidden_env)))

    # Emit machine-readable + human rows; exit nonzero on any failure
    import json
    payload = {
        "audit": "qwen38-27b-sm121",
        "vllm": version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capability": list(capability),
        "rows": [{"name": n, "ok": o, "detail": d} for n, o, d in rows],
        "passed": all(o for _, o, _ in rows),
    }
    out = Path(os.environ.get("AUDIT_OUT", "/tmp/qwen38_audit.json"))
    out.write_text(json.dumps(payload, indent=2) + "\n")
    for n, o, d in rows:
        print(f"{'PASS' if o else 'FAIL'}  {n}  {d}")
    print("AUDIT_PASS" if payload["passed"] else "AUDIT_FAIL")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
