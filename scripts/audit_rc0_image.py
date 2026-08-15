#!/usr/bin/env python3
"""Post-build audit of the rc0 image: version + SM121 arch + qwen3_5 imports.

Usage (against the assembled image):
  docker run --rm --entrypoint python3 <image> /audit/audit_rc0_image.py
"""
import importlib
import json
import subprocess
import sys

ok_rows = []


def check(name: str, ok: bool, detail: str = "") -> None:
    ok_rows.append({"name": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (" :: " + detail if detail else ""))


def main() -> int:
    import vllm
    check("vllm_import", True, vllm.__version__)
    from vllm.version import __version__ as v2
    check("version_consistent", vllm.__version__ == v2, f"{vllm.__version__} == {v2}")

    import torch
    cap = torch.cuda.get_device_capability()
    check("device_capability", cap == (12, 1), str(cap))

    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
    check("qwen3_5_mtp_import", True, "Qwen3_5MTP")
    from vllm.model_executor.models import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    check("qwen3_5_arch_registered",
          any("Qwen3_5" in a for a in archs),
          ",".join(a for a in archs if "Qwen3_5" in a))

    # Spec method contract
    from vllm.config.speculative import MTPModelTypes  # type: ignore[attr-defined]
    print("MTPModelTypes:", MTPModelTypes)

    # Architecture audit of the compiled extension (cuobjdump on _C)
    try:
        import vllm._C as vllm_c  # noqa: F401
        so = vllm_c.__file__
        out = subprocess.run(["cuobjdump", "--list-elf", so],
                             capture_output=True, text=True, timeout=300)
        archs_found = sorted({l.strip().split()[-1] for l in out.stdout.splitlines()
                              if "sm_" in l or "compute_" in l})
        has_121 = any("121" in a for a in archs_found)
        has_120a = any("120a" in a or "f" in a.split("_")[-1] for a in archs_found)
        check("native_sm121_sass", has_121, ",".join(archs_found[:12]))
        print("arch note (120a-family also present = fine, 121 must exist):",
              has_120a)
    except Exception as exc:
        check("native_sm121_sass", False, f"cuobjdump failed: {exc}")

    passed = all(r["ok"] for r in ok_rows)
    print("AUDIT_PASS" if passed else "AUDIT_FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
