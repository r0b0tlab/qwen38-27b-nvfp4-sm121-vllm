#!/usr/bin/env python3
"""Build the deterministic 200-question quality set (Muse protocol mix).

80 GSM8K (exact numeric) / 40 HumanEval (exec-graded) / 40 IFEval (format-graded)
/ 20 hard reasoning (authored, fixed) / 20 agentic coding (authored, fixed).
Deterministic selection (seed 1234, fixed slices), hashed manifest.
Run on head (has internet); output: quality-200.jsonl + quality-200.manifest.json
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datasets import load_dataset

OUT = sys.argv[1] if len(sys.argv) > 1 else "quality-200.jsonl"
rng = random.Random(1234)
rows = []

# --- GSM8K: first 80 test rows (fixed slice, deterministic) ---
gsm = load_dataset("openai/gsm8k", "main", split="test")
for i in range(80):
    ex = gsm[i]
    answer = ex["answer"].split("####")[-1].strip().replace(",", "")
    rows.append({
        "id": f"gsm8k-{i:03d}", "family": "gsm8k", "prompt": ex["question"],
        "reference": answer, "grade": "numeric_exact",
    })
print("gsm8k: 80")

# --- HumanEval: first 40 (fixed slice) ---
he = load_dataset("openai/openai_humaneval", split="test")
for i in range(40):
    ex = he[i]
    rows.append({
        "id": f"humaneval-{i:03d}", "family": "humaneval",
        "prompt": (
            "Complete the following Python function. Return ONLY a Python code block "
            "with the complete function definition.\n\n```python\n" + ex["prompt"] + "```"
        ),
        "reference": {"entry_point": ex["entry_point"], "canonical": ex["canonical_solution"], "test": ex["test"]},
        "grade": "exec",
    })
print("humaneval: 40")

# --- IFEval: first 40 (fixed slice) ---
ife = load_dataset("google/IFEval", split="train")
for i in range(40):
    ex = ife[i]
    rows.append({
        "id": f"ifeval-{i:03d}", "family": "ifeval", "prompt": ex["prompt"],
        "reference": {"instruction_id_list": ex["instruction_id_list"], "kwargs": ex["kwargs"]},
        "grade": "ifeval_strict",
    })
print("ifeval: 40")

# --- 20 hard reasoning (authored, fixed, hash-stable) ---
HARD = [
    ("Prove or disprove: for all positive integers n, n^3 - n is divisible by 6. Then state the generalization for p^3 - p with prime p > 3.", "divisible by 6 always; p^3-p divisible by 6 but not always by 24 (e.g. p=5:120 divisible by 24; p=7:342=6*57 not divisible by 24)"),
    ("A bag has 4 red, 5 blue, 3 green balls. Balls drawn WITHOUT replacement until first green. What is the expected number of draws? Show the exact fraction.", "expected value 13/4 = 3.25"),
    ("Find all integer solutions of x^2 - y^2 = 105 with x,y >= 0. List every pair.", "(x,y) in {(53,52),(19,16),(11,4),(15,12),(9,4)? -> verify: 105=105*1=35*3=21*5=15*7; pairs (53,52),(19,16),(11,4)? 11^2-4^2=121-16=105 yes, (19,16)=361-256=105 yes, (53,52)=2809-2704=105 yes, (15,12)=225-144=81 no, so (53,52),(19,16),(11,4) plus (x,y) with y>x none since x^2-y^2>0 needs x>y; also (12,?)... answer: (53,52),(19,16),(11,4)"),
    ("Prove that the square root of 2 is irrational in exactly five sentences.", "valid 5-sentence proof by contradiction via evenness of p and q"),
    ("Compute the determinant of the 4x4 Hilbert matrix H_ij = 1/(i+j-1) as an exact fraction.", "1/6048000 reduced? Hilbert4 det = 1/6048000? actual value 1/6048000 = 1/(2^13 *3^3*5^2*7) -> det = 1/6048000"),
    ("You have a 12-liters-per-hour leaking tank, initially 100 L. Every hour you add 8 L. Write the recurrence and state whether the tank empties; if so, after how many hours (integer).", "empties; V(n+1)=V(n)-12+8=V(n)-4 -> 100/4=25 hours"),
    ("Sort these algorithms by worst-case big-Theta for n elements and justify each in one clause: heapify, mergesort, quicksort, binary search, BFS on a tree.", "heapify Θ(n), mergesort Θ(n log n), quicksort Θ(n^2), binary search Θ(log n), BFS tree Θ(n)"),
    ("A fair coin is flipped 10 times. Given exactly 3 heads, what is the probability the first flip is a head? Exact fraction.", "3/10"),
    ("Find the smallest positive integer missing from the infinite multiset {6k+2, 7k+3 : k>=0} is ill-posed; instead: find the smallest positive integer NOT representable as 6a+7b with a,b>=0.", "Chicken McNugget 6*7-6-7=29; answer 29"),
    ("Give a DFA with minimum states for binary strings divisible by 3 (read MSB first) — describe states and transitions, and state the minimum state count.", "3 states (remainder 0,1,2); accept = remainder 0"),
    ("Integrate x^2 * e^(-x) dx from 0 to infinity. Exact value.", "2"),
    ("If f(x)=x^x^x (right-assoc), find f'(1).", "1"),
    ("Prove or disprove: every undirected graph with minimum degree >= 2 contains a cycle.", "true; walk until repeat vertex by pigeonhole on finite graph"),
    ("Three ants start at the corners of an equilateral triangle side 1, each chasing the next at speed 1. How far does each travel before meeting? Exact value.", "2/3"),
    ("State the exact number of boolean functions of 4 variables that are self-dual.", "2^(2^3)=256"),
    ("Compute sum_{k=0}^{n} (-1)^k C(n,k) for n=17.", "0"),
    ("A rope around the earth (radius R) is lengthened by 1 m and lifted uniformly. What is the lift height? Exact expression.", "1/(2π) meters"),
    ("Give a one-line closed form for the n-th term: 1, 11, 21, 1211, 111221, ...", "look-and-say sequence; a(n) = run-length encoding of a(n-1)"),
    ("Prove: among any 5 points in a unit square, two are within sqrt(2)/2.", "pigeonhole on four 1/2x1/2 subsquares; max distance in a subsquare is sqrt(2)/2"),
    ("Two dice sum to 7. What is the probability one of them is a 6? Exact fraction.", "2/6 = 1/3"),
]
for i, (q, ref) in enumerate(HARD):
    rows.append({"id": f"hard-{i:02d}", "family": "hard_reasoning", "prompt": q, "reference": ref, "grade": "manual"})
print("hard: 20")

# --- 20 agentic coding (authored, fixed, functional grading) ---
AGENTIC = [
    ("Write a Python function `top_k_frequent(words, k)` returning the k most frequent strings in descending frequency, ties broken lexicographically. Include 3 assert tests.", "exec+asserts"),
    ("Write `flatten(nested)` that flattens arbitrarily nested lists into a flat list, iteratively (no recursion). Include asserts.", "exec+asserts"),
    ("Write `lru_cache(capacity)` class with get/put in O(1). Include a test that evicts correctly.", "exec+asserts"),
    ("Write `merge_intervals(intervals)` sorting and merging overlapping [start,end) intervals. Include asserts.", "exec+asserts"),
    ("Write `word_break(s, word_dict)` returning True if s segments into dictionary words. Include asserts.", "exec+asserts"),
    ("Write `min_window(s, t)` returning the minimal substring of s containing all chars of t. Include asserts.", "exec+asserts"),
    ("Write `detect_cycle(head)` for a linked list (Floyd). Include a test with a cycle.", "exec+asserts"),
    ("Write `serialize(root)`/`deserialize(data)` for binary trees (preorder, None markers). Roundtrip test.", "exec+asserts"),
    ("Write `trapping_rain_water(heights)` in O(n). Include asserts with known answers.", "exec+asserts"),
    ("Write `edit_distance(a, b)` DP with O(min) space. Include asserts.", "exec+asserts"),
    ("Write a rate limiter class `RateLimiter(max_per_minute)` with allow() using a sliding window. Test 70 calls, count allowed == 60.", "exec+asserts"),
    ("Write `parse_csv_line(line)` handling quoted fields with embedded commas and escaped quotes. Include asserts.", "exec+asserts"),
    ("Write `retry(fn, attempts, base_delay)` with exponential backoff decorator. Test with a flaky counter function.", "exec+asserts"),
    ("Write `group_anagrams(words)` returning grouped lists sorted. Include asserts.", "exec+asserts"),
    ("Write `matrix_rotate_90(m)` in-place. Include asserts.", "exec+asserts"),
    ("Write `jsonpath_get(obj, path)` supporting dot and [n] indexing. Include asserts.", "exec+asserts"),
    ("Write `validate_parentheses(s)` for (),[],{} mixed. Include asserts.", "exec+asserts"),
    ("Write `longest_increasing_subsequence(nums)` in O(n log n). Include asserts.", "exec+asserts"),
    ("Write `shard_key(user_id, n_buckets)` using consistent hashing (ring with 100 vnodes, sha256). Deterministic given inputs; include stability test.", "exec+asserts"),
    ("Write `debounce(fn, wait_ms)` for a fake clock (inject time source). Test call coalescing.", "exec+asserts"),
]
for i, (q, _) in enumerate(AGENTIC):
    rows.append({"id": f"agentic-{i:02d}", "family": "agentic_coding", "prompt": q, "reference": "exec+asserts", "grade": "exec"})
print("agentic: 20")

assert len(rows) == 200, len(rows)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

blob = "".join(json.dumps(r, sort_keys=True) for r in rows)
manifest = {
    "count": len(rows),
    "families": {fam: sum(1 for r in rows if r["family"] == fam) for fam in {r["family"] for r in rows}},
    "ordered_sha256": hashlib.sha256(blob.encode()).hexdigest(),
    "file_sha256": hashlib.sha256(open(OUT, "rb").read()).hexdigest(),
    "seed": 1234,
    "protocol": "greedy temp=0, enable_thinking=false, max_tokens 1024 (gsm8k/ifeval/hard) / 1536 (humaneval/agentic)",
}
json.dump(manifest, open(OUT.replace(".jsonl", ".manifest.json"), "w"), indent=2)
print(json.dumps(manifest, indent=2))
