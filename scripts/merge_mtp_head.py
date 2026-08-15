#!/usr/bin/env python3
"""Merge the trained BF16 MTP head from the BF16 source into the attempt18
NVFP4 checkpoint, mirroring the official Qwen3.6-27B-NVFP4 control contract
(15 mtp.* tensors, BF16, absent from hf_quant_config).

Usage: python3 merge_mtp_head.py
Creates candidates/attempt18-mixedhess-official512-mtp: shards 1-3 hardlinked
from attempt18, MTP tensors appended as shard 4, index rewritten to 4-of-4,
audit JSON + sha256 manifest written.
"""
import hashlib
import json
import os
import shutil
import struct
import sys

SRC = os.path.expanduser("~/models/llm/bf16/Qwen3.8-27B")
BASE = os.path.expanduser("~/qwen38-ops/candidates/attempt18-mixedhess-official512")
OUT = os.path.expanduser("~/qwen38-ops/candidates/attempt18-mixedhess-official512-mtp")
CHUNK = 1 << 24


def shard_header(path: str) -> tuple[dict, int]:
    """Parse a safetensors file header; return (header_dict, data_start)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def main() -> None:
    src_idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    mtp_keys = sorted(k for k in src_idx["weight_map"] if k.startswith("mtp."))
    if len(mtp_keys) != 15:
        print(f"FATAL: expected 15 mtp tensors in source, found {len(mtp_keys)}")
        sys.exit(1)

    base_idx = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
    if any(k.startswith("mtp.") for k in base_idx["weight_map"]):
        print("FATAL: attempt18 already has mtp tensors")
        sys.exit(1)
    qc = json.load(open(os.path.join(BASE, "hf_quant_config.json")))
    qc_layers = qc.get("quantization", {}).get("layers", {})
    if any(k.startswith("mtp.") for k in qc_layers):
        print("FATAL: hf_quant_config already maps mtp layers")
        sys.exit(1)

    if os.path.exists(OUT):
        print(f"FATAL: {OUT} already exists; refusing to overwrite")
        sys.exit(1)
    os.makedirs(OUT)
    copied, linked = [], []
    for name in sorted(os.listdir(BASE)):
        s = os.path.join(BASE, name)
        if name.endswith(".safetensors"):
            os.link(s, os.path.join(OUT, name))
            linked.append(name)
        elif os.path.isfile(s):
            shutil.copy2(s, os.path.join(OUT, name))
            copied.append(name)

    # MTP tensor slices from the BF16 source shards (raw bytes preserved).
    by_shard: dict[str, list[str]] = {}
    for k in mtp_keys:
        by_shard.setdefault(src_idx["weight_map"][k], []).append(k)
    extracted: dict[str, tuple[str, str, list, int, int]] = {}
    for shard, keys in sorted(by_shard.items()):
        hdr, data_start = shard_header(os.path.join(SRC, shard))
        for k in keys:
            m = hdr[k]
            extracted[k] = (shard, m["dtype"], m["shape"],
                            data_start + m["data_offsets"][0],
                            m["data_offsets"][1] - m["data_offsets"][0])
    total_new = sum(v[4] for v in extracted.values())

    # Write shard 4 with all MTP tensors in key order.
    new_shard = "model-00004-of-00004.safetensors"
    header: dict = {"__metadata__": {"format": "pt",
                                     "merge": "mtp-head-bf16-from-source"}}
    off = 0
    for k in mtp_keys:
        _, dt, shp, _, nb = extracted[k]
        header[k] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + nb]}
        off += nb
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - (len(hb) % 8)) % 8)
    out_path = os.path.join(OUT, new_shard)
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        for k in mtp_keys:
            shard, _, _, abs_off, nb = extracted[k]
            with open(os.path.join(SRC, shard), "rb") as src:
                src.seek(abs_off)
                remaining = nb
                while remaining:
                    chunk = src.read(min(CHUNK, remaining))
                    if not chunk:
                        raise RuntimeError(f"short read on {k}")
                    out.write(chunk)
                    remaining -= len(chunk)

    # Rewrite index: 4 shards, weight_map fully re-pointed.
    wm = dict(base_idx["weight_map"])
    for k in mtp_keys:
        wm[k] = new_shard
    inv: dict[str, list[str]] = {}
    for k, s in wm.items():
        inv.setdefault(s, []).append(k)
    final_map: dict[str, str] = {}
    for i, s in enumerate(sorted(inv), 1):
        nn = f"model-{i:05d}-of-00004.safetensors"
        if s != nn:
            os.link(os.path.join(OUT, s), os.path.join(OUT, nn))
            os.unlink(os.path.join(OUT, s))
        for k in inv[s]:
            final_map[k] = nn
    total_size = base_idx["metadata"]["total_size"] + total_new
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": final_map}, f, indent=1)

    audit = {
        "mtp_tensors": len(mtp_keys),
        "keys": mtp_keys,
        "dtypes": {k: extracted[k][1] for k in mtp_keys},
        "new_shard_bytes": os.path.getsize(out_path),
        "index_total_size": total_size,
        "quant_cfg_mtp_entries": 0,
        "hardlinked": linked,
        "copied_files": copied,
        "source": SRC,
        "base": BASE,
    }
    with open(os.path.join(OUT, "mtp-merge-audit.json"), "w") as f:
        json.dump(audit, f, indent=1)

    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    with open(os.path.join(OUT, "mtp-shard4.sha256"), "w") as f:
        f.write(f"{h.hexdigest()}  {new_shard}\n")
    print(json.dumps({"mtp_tensors": audit["mtp_tensors"],
                      "new_shard_bytes": audit["new_shard_bytes"],
                      "index_total_size_GiB": round(total_size / 2**30, 2)},
                     indent=1))
    print("shard4 sha256:", h.hexdigest())
    print("OUT:", OUT)


if __name__ == "__main__":
    main()
