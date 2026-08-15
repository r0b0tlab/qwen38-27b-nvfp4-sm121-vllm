#!/usr/bin/env python3
"""Short sanity suite: semantic ladder + thinking modes + tool-call smoke + long gen.

Run against a live endpoint after engine/ckpt changes. Emits JSON verdict file.
Usage: run_sanity_suite.py --base-url http://192.168.0.2:8000 --output <json> [--tag NAME]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

MODEL = "qwen38-27b"


def chat(base: str, prompt: str, max_tokens: int = 512,
         thinking: bool | None = False, tools: list | None = None,
         timeout: int = 900) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    msg = body["choices"][0]["message"]
    return {
        "text": msg.get("content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish": body["choices"][0].get("finish_reason"),
        "usage": body.get("usage"),
        "elapsed": round(time.perf_counter() - t0, 2),
    }


WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tag", default="sanity")
    args = ap.parse_args()

    checks: list[dict] = []

    def record(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + " :: " + str(detail)[:120])

    # 1. arithmetic (thinking off)
    r = chat(args.base_url, "Compute 19 times 23. End with the integer result only.", 256, thinking=False)
    record("arith_437_thinkoff", "437" in (r["text"] or ""), r["text"][-80:])

    # 2. arithmetic (thinking on — reasoning goes through chat kwargs)
    r = chat(args.base_url, "Compute 19 times 23. End with the integer result only.", 1024, thinking=True)
    record("arith_437_thinkon", "437" in (r["text"] or ""), r["text"][-80:])

    # 3. exact string
    r = chat(args.base_url, 'Reply with exactly the word BANANA and nothing else.', 16, thinking=False)
    record("exact_string", (r["text"] or "").strip().upper().startswith("BANANA"), r["text"])

    # 4. word problem
    r = chat(args.base_url,
             "A train travels 360 km in 4.5 hours. What is its average speed in km/h? "
             "End with the numeric result only.", 256, thinking=False)
    record("word_problem_80", "80" in (r["text"] or ""), r["text"][-80:])

    # 5. code shape
    r = chat(args.base_url,
             "Write a Python function fib(n) returning the n-th Fibonacci number "
             "iteratively. Output only the code block.", 512, thinking=False)
    ok = ("def fib" in r["text"]) and ("for" in r["text"] or "while" in r["text"])
    record("code_shape", ok, r["text"][:80])

    # 6. tool-call smoke
    r = chat(args.base_url, "What is the weather in Tokyo right now? Use the tool.",
             256, thinking=False, tools=WEATHER_TOOL)
    tc = r["tool_calls"]
    ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" \
        and "tokyo" in tc[0]["function"]["arguments"].lower()
    record("tool_call_get_weather", ok, tc[:1] if tc else "no tool_calls")

    # 7. long generation 2K+
    r = chat(args.base_url,
             "Write a detailed 2000-word technical overview of speculative decoding "
             "in LLM inference. Be continuous, do not stop early.",
             3072, thinking=False, timeout=1800)
    toks = (r.get("usage") or {}).get("completion_tokens", 0)
    record("long_gen_2k", toks >= 2000, f"completion_tokens={toks} finish={r['finish']}")

    # 8. repeat determinism (temp 0)
    a = chat(args.base_url, "List the first 10 prime numbers, comma-separated.", 64, thinking=False)
    b = chat(args.base_url, "List the first 10 prime numbers, comma-separated.", 64, thinking=False)
    record("repeat_deterministic", a["text"] == b["text"], "identical" if a["text"] == b["text"] else "drift")

    passed = sum(c["ok"] for c in checks)
    verdict = {"tag": args.tag, "base_url": args.base_url, "passed": passed,
               "total": len(checks), "all_ok": passed == len(checks),
               "checks": checks,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.output).write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"SANITY_{'PASS' if verdict['all_ok'] else 'FAIL'} {passed}/{len(checks)}")
    return 0 if verdict["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
