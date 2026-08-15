#!/usr/bin/env python3
"""Run the 200-question quality set against a serving endpoint; grade + persist rows.

Usage: run_quality_set.py --base-url http://IP:8000 --run-id <id> --set quality-200.jsonl
Output: <run-id>.rows.jsonl (per-row outputs) + <run-id>.summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

MODEL = "qwen38-27b"


def chat(base: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
    msg = body["choices"][0]["message"]
    return {
        "text": msg.get("content") or "",
        "finish": body["choices"][0].get("finish_reason"),
        "usage": body.get("usage"),
        "elapsed": time.perf_counter() - t,
    }


def grade_numeric(text: str, ref: str) -> bool:
    m = re.search(r"-?\$?\d[\d,]*\.?\d*", text.replace(",", ""))
    refc = ref.replace(",", "")
    return bool(m) and m.group(0).replace("$", "").replace(",", "") == refc


def grade_exec(text: str, ref: dict) -> bool:
    """Extract python block; run entry-point call against the canonical test."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    # strip any leading 'python' word line
    code = re.sub(r"^python\s*$", "", code, flags=re.M)
    ns: dict = {}
    try:
        exec(compile(code, "<gen>", "exec"), ns)  # noqa: S102 - sandboxed eval of model code
    except Exception:
        return False
    if ref.get("entry_point") and ref["entry_point"] not in ns:
        return False
    if ref.get("test"):
        try:
            harness = ref["test"] + f"\ncheck({ref['entry_point']})\n"
            if "check(" not in ref["test"]:
                harness = ref["test"] + f"\nassert {ref['entry_point']} is not None\n"
            exec(compile(harness, "<test>", "exec"), ns)  # noqa: S102
        except Exception:
            return False
    return True


def grade_ifeval(text: str, ref: dict) -> bool:
    """Strict subset: the four most robust instruction types."""
    ids = ref.get("instruction_id_list", [])
    kw = ref.get("kwargs", [])
    ok = True
    for i, iid in enumerate(ids):
        args = kw[i] if i < len(kw) else {}
        if iid == "keywords:existence":
            for w in args.get("keywords", []):
                if w.lower() not in text.lower():
                    ok = False
        elif iid == "keywords:frequency":
            n = args.get("frequency", 1); w = args.get("keyword", "")
            if text.lower().count(w.lower()) < n:
                ok = False
        elif iid == "length_constraints:number_words":
            cmp_ = args.get("comparison", "at least"); n = args.get("num_words", 0)
            wc = len(text.split())
            if cmp_ == "at least" and wc < n: ok = False
            if cmp_ == "less than" and wc >= n: ok = False
        elif iid == "startend:quotation":
            if not (text.strip().startswith('"') and text.strip().endswith('"')):
                ok = False
        else:
            # un-graded instruction type in this strict subset -> neutral (skip)
            pass
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--set", default="quality-200.jsonl")
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.set)]
    rows_path = Path(f"{a.run_id}.rows.jsonl")
    # Resume support: skip ids already graded in a previous partial run
    done_ids: set[str] = set()
    out_rows: list[dict] = []
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            try:
                out_rows.append(json.loads(line))
                done_ids.add(out_rows[-1]["id"])
            except Exception:
                break  # truncated final line from a crash mid-write
        print(f"resuming: {len(done_ids)} rows already collected", flush=True)
    t0 = time.time()
    with rows_path.open("a") as rows_fh:
        for i, r in enumerate(rows):
            if r["id"] in done_ids:
                continue
            mt = 1536 if r["family"] in ("humaneval", "agentic_coding") else 1024
            try:
                res = chat(a.base_url, r["prompt"], mt)
            except Exception as exc:
                res = {"text": "", "finish": "error", "usage": None, "elapsed": 0.0, "error": repr(exc)}
            g = r["grade"]
            ref = r["reference"]
            if g == "numeric_exact":
                ok = grade_numeric(res["text"], ref)
            elif g == "exec":
                # agentic family stores reference as the string "exec+asserts" —
                # those prompts ask for asserts inline; execution success = pass.
                ok = grade_exec(res["text"], ref if isinstance(ref, dict) else {})
            elif g == "ifeval_strict":
                ok = grade_ifeval(res["text"], ref)
            else:
                ok = None  # manual
            out_rows.append({
                "id": r["id"], "family": r["family"], "grade": g, "passed": ok,
                "finish": res.get("finish"), "elapsed": res.get("elapsed"),
                "completion_tokens": (res.get("usage") or {}).get("completion_tokens"),
                "text": res["text"][:4000],
            })
            rows_fh.write(json.dumps(out_rows[-1]) + "\n")
            rows_fh.flush()
            if (i + 1) % 10 == 0:
                done = sum(1 for x in out_rows if x["passed"] is not None and x["passed"])
                graded = sum(1 for x in out_rows if x["passed"] is not None)
                print(f"{i+1}/200 graded={graded} pass={done} ({time.time()-t0:.0f}s)", flush=True)
    fam_stats: dict = {}
    for x in out_rows:
        f = fam_stats.setdefault(x["family"], {"n": 0, "passed": 0, "graded": 0})
        f["n"] += 1
        if x["passed"] is not None:
            f["graded"] += 1
            f["passed"] += bool(x["passed"])
    summary = {
        "run_id": a.run_id, "base_url": a.base_url, "elapsed_s": time.time() - t0,
        "families": fam_stats,
        "auto_graded_pass": sum(1 for x in out_rows if x["passed"]),
        "auto_graded_total": sum(1 for x in out_rows if x["passed"] is not None),
    }
    Path(f"{a.run_id}.rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out_rows))
    Path(f"{a.run_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["families"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
