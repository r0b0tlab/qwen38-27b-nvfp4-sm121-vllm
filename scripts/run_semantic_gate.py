#!/usr/bin/env python3
"""Deterministic semantic release gate for the native FP8-KV profile."""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

MODEL = "qwen38-27b"
LEAKAGE = ("<|assistant|>", "<|im_start|>", "<|im_end|>", "chat_template")


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def complete(base: str, prompt: str, max_tokens: int = 128) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        # Exact-output release probes must not spend the entire response budget
        # inside the model's optional reasoning channel.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read())
        status = response.status
    text = body["choices"][0]["message"].get("content") or ""
    return {
        "prompt": prompt,
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "text": text,
        "finish_reason": body["choices"][0].get("finish_reason"),
        "usage": body.get("usage"),
    }


def generic_ok(text: str) -> bool:
    if not text.strip() or any(token in text for token in LEAKAGE):
        return False
    if re.search(r"([^\w\s])\1{12,}", text):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    models = get_json(base + "/v1/models")
    model_ids = [item["id"] for item in models.get("data", [])]

    probes = {
        "arithmetic": complete(base, "Compute 19 times 23. End your answer with the integer result.", 1024),
        "exact_string": complete(base, "Reply with exactly this text and nothing else: GB10_NATIVE_OK", 32),
        "word_problem": complete(base, "Six boxes contain nine bolts each. How many bolts are there? End with the integer result.", 128),
        "code": complete(base, "Write a Python function named add that returns the sum of arguments a and b. Return only the function definition.", 128),
    }
    repeats = [complete(base, "Reply with exactly this text and nothing else: DETERMINISTIC_NATIVE", 32) for _ in range(3)]

    checks = {
        "model_exact": model_ids == [MODEL],
        "arithmetic_437": bool(re.search(r"\b437\b", probes["arithmetic"]["text"])),
        "exact_string": probes["exact_string"]["text"].strip() == "GB10_NATIVE_OK",
        "word_problem_54": bool(re.search(r"\b54\b", probes["word_problem"]["text"])),
        "code_shape": "def add" in probes["code"]["text"] and bool(re.search(r"return\s+a\s*\+\s*b", probes["code"]["text"])),
        "repeat_deterministic": len({item["text"] for item in repeats}) == 1,
        "repeat_exact": all(item["text"].strip() == "DETERMINISTIC_NATIVE" for item in repeats),
        "all_http_200": all(item["status"] == 200 for item in [*probes.values(), *repeats]),
        "no_empty_loop_or_template_leakage": all(generic_ok(item["text"]) for item in [*probes.values(), *repeats]),
    }
    result = {
        "model_ids": model_ids,
        "profile": "W4A16 NVFP4 + FP8 KV + MTP K=3",
        "probes": probes,
        "repeats": repeats,
        "checks": checks,
        "passed": all(checks.values()),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(out), "checks": checks, "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
