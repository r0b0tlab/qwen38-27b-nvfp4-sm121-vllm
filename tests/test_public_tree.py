#!/usr/bin/env python3
"""Fail-closed public-tree checks for an independent clone."""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "192.168.",
    "r0b0tdgx@",
    "/home/r0b0tdgx",
    "HF_TOKEN=",
    "ghp_",
)
REQUIRED = (
    "LICENSE",
    "README.md",
    "HF-CARD.md",
    "final-sota-shards.sha256",
    "docker/Dockerfile.rc0",
    "scripts/serve.sh",
    "scripts/click_run_mtp.sh",
    "scripts/click_run_dflash2.sh",
    "scripts/adapt_dspark_draft.py",
    "scripts/run_semantic_gate.py",
    "scripts/publish_hf.sh",
    "reference/dspark-adapted-config.json",
    "tests/test_adapt_dspark_draft.py",
    "quality-200-vllm-dspark.json",
)
SHARDS = (
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
)


def test_required_files_exist() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    assert not missing, missing


def test_no_private_host_paths() -> None:
    hits = []
    skip = {".git", "__pycache__", ".pytest_cache"}
    skip_files = {"tests/test_public_tree.py"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in skip for part in path.parts):
            continue
        rel = str(path.relative_to(ROOT))
        if rel in skip_files or path.suffix in {".pyc", ".safetensors"}:
            continue
        text = path.read_text(errors="ignore")
        for needle in FORBIDDEN:
            if needle in text:
                hits.append(f"{rel}:{needle}")
    assert not hits, hits


def test_shard_manifest_is_complete_4_of_4() -> None:
    lines = [line for line in (ROOT / "final-sota-shards.sha256").read_text().splitlines() if line.strip()]
    names = [line.split()[-1] for line in lines]
    assert names == list(SHARDS), names
    assert all(len(line.split()[0]) == 64 for line in lines)


def test_readme_documents_click_run_and_gaps() -> None:
    text = (ROOT / "README.md").read_text()
    for needle in (
        "scripts/serve.sh mtp",
        "scripts/click_run_mtp.sh",
        "scripts/adapt_dspark_draft.py",
        "Qwen3DSparkModel",
        "tactic=-1",
        "model-00001-of-00004.safetensors",
        "sha256:5bd3f329",
        "19 × 23",
    ):
        assert needle in text, needle


def test_serve_script_covers_five_profiles() -> None:
    text = (ROOT / "scripts/serve.sh").read_text()
    for needle in ("ar)", "mtp)", "dspark)", "dflash2)", "long)", "Qwen3DSparkModel", "DFlash2DraftModel", 'num_speculative_tokens":3'):
        assert needle in text, needle


def test_click_run_fails_closed_on_ready_timeout() -> None:
    text = (ROOT / "scripts/click_run_mtp.sh").read_text()
    assert "READY_TIMEOUT" in text
    assert "container died during startup" in text


def test_python_scripts_parse() -> None:
    for path in ROOT.joinpath("scripts").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))


def test_shell_scripts_parse() -> None:
    scripts = [
        ROOT / "scripts/serve.sh",
        ROOT / "scripts/click_run_mtp.sh",
        ROOT / "scripts/click_run_dflash2.sh",
        ROOT / "scripts/publish_hf.sh",
        ROOT / "scripts/assemble_rc0.sh",
        ROOT / "scripts/run_mtp_ladder.sh",
        ROOT / "scripts/run_perf_suite.sh",
        ROOT / "scripts/run-ptq.sh",
    ]
    for path in scripts:
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{path.name}: {proc.stderr}"


def test_dspark_adapter_contract() -> None:
    from scripts.adapt_dspark_draft import adapt_config

    out = adapt_config({"architectures": ["DSparkDraftModel"], "model_type": "dspark"})
    assert out["architectures"] == ["Qwen3DSparkModel"]
    fixture = json.loads((ROOT / "reference/dspark-adapted-config.json").read_text())
    assert fixture["dspark_target_layer_ids"] == [4, 16, 28, 40, 52]
