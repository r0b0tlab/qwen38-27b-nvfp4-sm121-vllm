#!/usr/bin/env python3
"""Flex-extract GSM8K scoring over a run_quality_set.py rows file.

Campaign-standard scorer (parity with the bf16-baseline 86.2% / official-Q36
control 86.2% / attempt15 78.8% numbers): final bolded number wins; else the
last number in the text; exact string compare against the reference.

Usage: score_flex_gsm8k.py <rows.jsonl> <quality-set.jsonl>
"""
from __future__ import annotations

import json
import re
import sys


def final_answer(text: str):
    if not text:
        return None
    bolds = re.findall(r"\*\*\\?\$?(-?\d[\d,]*\.?\d*)\\?\$?\*\*", text)
    if bolds:
        return bolds[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    rows = [json.loads(l) for l in open(sys.argv[1])]
    qset = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2])}
    n = passed = 0
    fails: list[tuple[str, str, str]] = []
    for r in rows:
        if r["family"] != "gsm8k":
            continue
        n += 1
        ref = qset[r["id"]]["reference"].replace(",", "")
        got = final_answer(r.get("text") or "")
        if got is not None and got == ref:
            passed += 1
        else:
            fails.append((r["id"], ref, got or "NO-EXTRACT"))
    pct = 100.0 * passed / n if n else 0.0
    print(json.dumps({
        "rows_file": sys.argv[1],
        "gsm8k_flex": {"n": n, "passed": passed, "pct": round(pct, 2)},
        "failures_head": fails[:8],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
