#!/usr/bin/env python3
"""Needle-in-a-haystack via OpenAI chat completions (vLLM).

Constructs exact token-count prompts with the model tokenizer, places a unique
code near the middle of the filler, and checks exact retrieval.
"""
from __future__ import annotations

import argparse
import os
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def find_subseq(hay: list[int], needle: list[int]) -> int:
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i : i + n] == needle:
            return i
    raise RuntimeError("marker token sequence not found in template")


def post_chat(url: str, payload: dict, timeout: int = 7200) -> tuple[int, dict, float]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return resp.status, body, time.perf_counter() - t0
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"error": raw}
        return exc.code, body, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("Q38_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument(
        "--tokenizer",
        default="/home/r0b0tdgx/qwen38-ops/candidates/attempt18-mixedhess-official512",
    )
    ap.add_argument("--depths", type=int, nargs="+", default=None,
                    help="Explicit prompt depths. Default: derive from served max context.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--needle-fraction", type=float, default=0.5, help="0=start .. 1=end of filler")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.50, 0.90],
                    help="Depths as fractions of (max_model_len - max_tokens) when --depths omitted")
    ap.add_argument("--native-max-position", type=int, default=262144,
                    help="Model weight max_position_embeddings for protocol labeling")
    ap.add_argument("--protocol-label", default="MAX_CONTEXT_TEST")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--code", default="QWEN38-NIAH-9X4K")
    args = ap.parse_args()

    # Discover served max context (fail-closed for max-context protocol)
    try:
        with urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=30) as resp:
            models = json.loads(resp.read().decode())
    except Exception as exc:
        raise SystemExit(f"MAX_CONTEXT_UNKNOWN: cannot read /v1/models: {exc}") from exc
    served_m = None
    for m in models.get("data") or []:
        if m.get("id") == args.model and m.get("max_model_len"):
            served_m = int(m["max_model_len"])
            break
    if served_m is None:
        for m in models.get("data") or []:
            if m.get("max_model_len"):
                served_m = int(m["max_model_len"])
                break
    if not served_m:
        raise SystemExit("MAX_CONTEXT_UNKNOWN: no max_model_len in /v1/models")

    if args.depths:
        depths = list(args.depths)
        depth_source = "explicit"
    else:
        u = served_m - int(args.max_tokens)
        if u < 2048:
            raise SystemExit(f"usable context too small: M={served_m} G={args.max_tokens}")
        depths = sorted({max(1024, min(int(u * f), u)) for f in args.fractions})
        while len(depths) < len(args.fractions) and depths[-1] < u:
            depths.append(min(u, depths[-1] + max(1, (u - depths[-1]) // 2)))
        depths = depths[: len(args.fractions)]
        depth_source = "frac_of_max_context"

    print(json.dumps({
        "protocol": "NIAH_MAX_CONTEXT",
        "label": args.protocol_label,
        "native_max_position_embeddings": args.native_max_position,
        "served_max_model_len": served_m,
        "generation_reserve": args.max_tokens,
        "depth_source": depth_source,
        "fractions": args.fractions if depth_source == "frac_of_max_context" else None,
        "depths": depths,
        "matches_native": served_m == args.native_max_position,
    }), flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)

    marker = "<<<AQUILA_NIAH_SLOT_F4C2>>>"
    instruction = (
        "You are given a long archival record. A secret verification code appears "
        "exactly once. Read carefully and remember the code.\n"
        f"{marker}\n"
        "Question: What is the secret verification code? Reply with only the code, nothing else."
    )
    messages = [{"role": "user", "content": instruction}]

    # Prefer disabling thinking if the template supports it; fall back silently.
    try:
        rendered = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    base_ids = tok.encode(rendered, add_special_tokens=False)
    if hasattr(base_ids, "ids"):
        base_ids = list(base_ids.ids)
    else:
        base_ids = list(base_ids)

    mids = tok.encode(marker, add_special_tokens=False)
    pos = find_subseq(base_ids, mids)
    prefix = base_ids[:pos]
    suffix = base_ids[pos + len(mids) :]

    needle_text = f"\nIMPORTANT SECRET VERIFICATION CODE: {args.code}\n"
    needle = tok.encode(needle_text, add_special_tokens=False)
    unit = tok.encode(
        " The archival record contains neutral observations about weather, tools, books, roads, markets, and ordinary daily events.",
        add_special_tokens=False,
    )
    if not unit:
        raise RuntimeError("empty filler unit tokenization")

    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    rows = []

    for depth in depths:
        # Reserve generation budget inside max context if needed; depths are prompt targets.
        target_prompt = depth
        insertion = target_prompt - len(prefix) - len(suffix)
        if insertion < len(needle):
            rows.append(
                {
                    "depth": depth,
                    "status": "SKIP",
                    "reason": f"horizon too small insertion={insertion} needle={len(needle)}",
                    "passed": False,
                }
            )
            continue

        remaining = insertion - len(needle)
        left_n = int(remaining * args.needle_fraction)
        right_n = remaining - left_n
        left = (unit * ((left_n // len(unit)) + 3))[:left_n]
        right = (unit * ((right_n // len(unit)) + 3))[:right_n]
        input_ids = prefix + left + needle + right + suffix
        if len(input_ids) != target_prompt:
            # Fix off-by-ones from floor ops
            delta = target_prompt - len(input_ids)
            if delta > 0:
                input_ids = prefix + left + needle + right + unit[:delta] + suffix
            elif delta < 0:
                input_ids = input_ids[:target_prompt]
                # ensure suffix still present roughly — prefer trimming filler from right of left pad
                # rebuild safer:
                insertion2 = target_prompt - len(prefix) - len(suffix)
                remaining2 = insertion2 - len(needle)
                left_n = max(0, int(remaining2 * args.needle_fraction))
                right_n = max(0, remaining2 - left_n)
                left = (unit * ((left_n // len(unit)) + 3))[:left_n]
                right = (unit * ((right_n // len(unit)) + 3))[:right_n]
                input_ids = prefix + left + needle + right + suffix
                assert len(input_ids) == target_prompt, (len(input_ids), target_prompt)

        prompt_text = tok.decode(input_ids, skip_special_tokens=False)
        # Send as a single user message body (already templated content reconstructed).
        # Safer for exact length: use the decoded user-visible path via raw completions if available.
        # vLLM chat with full rendered string as user content can re-template; use /v1/completions with prompt tokens if supported.
        # Prefer chat with pre-rendered prompt via "prompt" is not standard. Use completions endpoint with decoded prompt without re-applying template.
        # Actually apply_chat_template already includes specials; for chat API we'd double-template.
        # Use completions API with prompt string from decode, or token prompt if supported.
        payload = {
            "model": args.model,
            "prompt": None,
            "messages": None,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            # vLLM extension: pass tokens directly when supported
            "prompt_token_ids": input_ids,
        }
        # Try chat completions with echo-free prompt_embeds alternative:
        # Many vLLM builds accept prompt as string on /v1/completions.
        comp_url = args.base_url.rstrip("/") + "/completions"
        comp_payload = {
            "model": args.model,
            "prompt": input_ids,  # vLLM accepts token id list for prompt
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "skip_special_tokens": True,
        }
        status, body, elapsed = post_chat(comp_url, comp_payload, timeout=1800)
        text = ""
        if status == 200 and body.get("choices"):
            ch0 = body["choices"][0]
            text = (ch0.get("text") or ch0.get("message", {}).get("content") or "").strip()
        else:
            # fallback: chat API with plain content (approximate length)
            chat_payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": tok.decode(input_ids, skip_special_tokens=True)}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
            }
            status, body, elapsed = post_chat(endpoint, chat_payload, timeout=1800)
            if status == 200 and body.get("choices"):
                msg = body["choices"][0].get("message") or {}
                text = (msg.get("content") or msg.get("reasoning") or "").strip()

        usage = body.get("usage") or {}
        passed = args.code in text
        row = {
            "depth": depth,
            "status": "OK" if status == 200 else f"HTTP_{status}",
            "http_status": status,
            "prompt_tokens_constructed": len(input_ids),
            "prompt_sha256": hashlib.sha256(json.dumps(input_ids).encode()).hexdigest(),
            "needle_token_offset": len(prefix) + left_n,
            "needle_fraction_effective": (len(prefix) + left_n) / max(1, len(input_ids)),
            "needle_code": args.code,
            "elapsed_s": elapsed,
            "response_text": text,
            "passed": passed,
            "usage": usage,
            "error": body.get("error"),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "depth": depth,
                    "prompt_tokens": len(input_ids),
                    "needle_frac": round(row["needle_fraction_effective"], 4),
                    "elapsed_s": round(elapsed, 2),
                    "passed": passed,
                    "text": text[:120],
                    "http": status,
                    "usage": usage,
                }
            ),
            flush=True,
        )

    report = {
        "protocol": {
            "id": "NIAH_MAX_CONTEXT",
            "label": args.protocol_label,
            "native_max_position_embeddings": args.native_max_position,
            "served_max_model_len": served_m,
            "generation_reserve": args.max_tokens,
            "depth_source": depth_source,
            "fractions": list(args.fractions) if depth_source == "frac_of_max_context" else None,
            "matches_native_capacity": served_m == args.native_max_position,
            "description": (
                "MAX CONTEXT TEST: three needle depths at fixed fractions of the served "
                "max_model_len (discovered from /v1/models). When served M equals native "
                "max_position_embeddings, this stresses full model context capacity."
            ),
        },
        "model": args.model,
        "base_url": args.base_url,
        "tokenizer": args.tokenizer,
        "depths": depths,
        "max_tokens": args.max_tokens,
        "needle_fraction": args.needle_fraction,
        "code": args.code,
        "rows": rows,
        "all_passed": bool(rows) and all(r.get("passed") for r in rows),
        "pass_count": sum(1 for r in rows if r.get("passed")),
        "total": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"all_passed": report["all_passed"], "pass_count": report["pass_count"], "total": report["total"], "output": str(args.output)}))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
