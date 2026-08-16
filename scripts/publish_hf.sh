#!/usr/bin/env bash
# Publish / repair the HF checkpoint. Always upload the 4-of-4 MTP-merged tree.
# Usage: CKPT=/path/to/final-sota-nvidia-recipe-mtp bash scripts/publish_hf.sh
set -euo pipefail
CKPT="${CKPT:?set CKPT to the local 4-of-4 MTP-merged checkpoint}"
REPO="${REPO:-r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121}"
for shard in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  [ -f "$CKPT/$shard" ] || { echo "refusing to publish incomplete tree: missing $CKPT/$shard" >&2; exit 2; }
done
python3 - "$CKPT" "$REPO" <<'PY'
from pathlib import Path
import hashlib, sys
from huggingface_hub import HfApi
ckpt, repo = Path(sys.argv[1]), sys.argv[2]
expected = {
    "model-00001-of-00004.safetensors": "4208cd3b45f605d9f67e29e2939c35f4bdc255d9baf72c951a6aa77c560e25f5",
    "model-00002-of-00004.safetensors": "024111b980be5bfa93616777e567110b5f3b1340578614165c51c78917457296",
    "model-00003-of-00004.safetensors": "927ee34337d1cc4419b97b82370229128003a5f2a17bb45670cdaaab8374d169",
    "model-00004-of-00004.safetensors": "47202b11daf026e3a472c030b18c4b7e6021c475d1f31ff06f1877d16639912d",
}
for name, digest in expected.items():
    h = hashlib.sha256()
    with (ckpt / name).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != digest:
        raise SystemExit(f"sha mismatch {name}: {got} != {digest}")
api = HfApi()
api.create_repo(repo, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=str(ckpt), repo_id=repo, repo_type="model",
                  commit_message="Publish complete 4-of-4 NVFP4+MTP checkpoint")
print("UPLOAD_DONE", repo)
PY
