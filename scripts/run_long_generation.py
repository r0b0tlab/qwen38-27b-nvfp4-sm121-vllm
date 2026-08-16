#!/usr/bin/env python3
"""Run deterministic long-generation and MTP smoke requests against vLLM."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.request
from pathlib import Path


def request(url: str, payload: dict) -> tuple[int, dict, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            result = json.loads(response.read().decode())
            return response.status, result, time.perf_counter() - started
    except Exception as exc:
        return 0, {"error": repr(exc)}, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = "qwen38-27b"
    prompts = [
        {
            "name": "arithmetic_mtp_smoke",
            "messages": [{"role": "user", "content": "Compute 19 * 23 and give the exact answer."}],
            "max_tokens": 512,
            "enable_thinking": False,
        },
        {
            "name": "long_generation",
            "messages": [{"role": "user", "content": "Write a detailed technical explanation of how grouped NVFP4 weight-only quantization differs from FP8 KV-cache quantization. Include headings, equations in plain text, and practical trade-offs. Aim for a sustained answer of roughly 700 words."}],
            "max_tokens": 2048,
            "enable_thinking": False,
        },
        {
            "name": "reasoning_stability",
            "messages": [{"role": "user", "content": "Solve this carefully: A warehouse has 19 pallets with 23 boxes on each pallet. If 17 boxes are damaged, how many good boxes remain? Show the reasoning and final integer."}],
            "max_tokens": 2048,
            "enable_thinking": True,
        },
    ]
    rows = []
    for item in prompts:
        payload = {
            "model": model,
            "messages": item["messages"],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": item["max_tokens"],
            "chat_template_kwargs": {"enable_thinking": item["enable_thinking"]},
            "stream": False,
        }
        status, response, elapsed = request(f"{args.base_url}/v1/chat/completions", payload)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        rows.append({
            "name": item["name"],
            "request": payload,
            "http_status": status,
            "elapsed_s": elapsed,
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage"),
            "content": message.get("content"),
            "reasoning": message.get("reasoning"),
            "response_error": response.get("error"),
        })
    semantic_checks = {
        "arithmetic_437": "437" in (rows[0].get("content") or ""),
        "long_content_present": len(rows[1].get("content") or "") >= 1000,
        "word_problem_420": "420" in ((rows[2].get("content") or "") + (rows[2].get("reasoning") or "")),
        "all_finished_stop": all(row.get("finish_reason") == "stop" for row in rows),
    }
    output = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model,
        "base_url": args.base_url,
        "tests": rows,
        "all_http_200": all(row["http_status"] == 200 for row in rows),
        "all_nonempty": all(bool(row.get("content") or row.get("reasoning")) for row in rows),
        "semantic_checks": semantic_checks,
        "passed": all(row["http_status"] == 200 for row in rows)
        and all(bool(row.get("content") or row.get("reasoning")) for row in rows)
        and all(semantic_checks.values()),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "all_http_200": output["all_http_200"], "all_nonempty": output["all_nonempty"], "semantic_checks": semantic_checks, "passed": output["passed"]}))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
