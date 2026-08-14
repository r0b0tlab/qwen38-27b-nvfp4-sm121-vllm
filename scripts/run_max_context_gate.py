#!/usr/bin/env python3
"""Atomic, no-retry Qwen3.8-27B near-window context qualification."""
from __future__ import annotations
import argparse, datetime as dt, json, time, urllib.error, urllib.request
from pathlib import Path

MODEL = "qwen38-27b"
FILLER = "Native kernels preserve calibrated execution while evidence remains deterministic and auditable. "
FRACTIONS = {"begin": 0.01, "quarter": 0.25, "middle": 0.50, "three_quarter": 0.75, "end": 0.99}


def post(url: str, payload: dict, timeout: int) -> tuple[int, dict, float]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read()), time.perf_counter()-started
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try: parsed = json.loads(body)
        except Exception: parsed = {"error": body}
        return exc.code, parsed, time.perf_counter()-started


def tokenize(base: str, prompt: str) -> int:
    status, body, _ = post(base+"/tokenize", {"model":MODEL,"messages":[{"role":"user","content":prompt}],"chat_template_kwargs":{"enable_thinking":False}}, 900)
    if status != 200: raise RuntimeError(f"tokenize HTTP {status}: {body}")
    if isinstance(body.get("count"), int): return body["count"]
    tokens = body.get("tokens") or body.get("token_ids")
    if isinstance(tokens, list): return len(tokens)
    raise RuntimeError(f"unsupported tokenize response: {body}")


def prompt_for(base: str, depth: int, label: str, fraction: float, dual: bool=False) -> tuple[str,int,list[str]]:
    target = depth - 768
    if dual:
        codes = [f"R0B0T-Q38-A-{depth}", f"R0B0T-Q38-B-{depth}"]
        needle = f"First verified code: {codes[0]}. Second verified code: {codes[1]}."
        question = "\nReturn only the second code followed by a space and then the first code."
    else:
        codes = [f"R0B0T-Q38-{label.upper()}-{depth}"]
        needle = f"The verified context code is {codes[0]}."
        question = "\nReturn only the verified context code."
    low, high = 0, max(1, target//4)
    def make(n: int) -> str:
        left = max(0, min(n, int(n*fraction)))
        return FILLER*left + needle + " " + FILLER*(n-left) + question
    while tokenize(base, make(high)) < target:
        low, high = high, high*2
    best, best_count = "", 0
    while low <= high:
        mid=(low+high)//2; prompt=make(mid); count=tokenize(base,prompt)
        if count <= target: best,best_count=prompt,count; low=mid+1
        else: high=mid-1
    if best_count < target-64:
        raise RuntimeError(f"prompt construction too short at depth {depth}: {best_count} < {target-64}")
    return best,best_count,codes


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2)+"\n")
    tmp.replace(path)


def completion(base: str, prompt: str, max_tokens: int, min_tokens: int=0, ignore_eos: bool=False) -> tuple[int,dict,float]:
    payload={"model":MODEL,"messages":[{"role":"user","content":prompt}],"temperature":0,"top_p":1,"max_tokens":max_tokens,"stream":False,"chat_template_kwargs":{"enable_thinking":False}}
    if min_tokens: payload["min_tokens"]=min_tokens
    if ignore_eos: payload["ignore_eos"]=True
    return post(base+"/v1/chat/completions",payload,3600)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-url",default="http://127.0.0.1:18080")
    ap.add_argument("--max-depth",type=int,required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--profile",choices=("ar","mtp2"),required=True)
    a=ap.parse_args(); base=a.base_url.rstrip("/"); out=Path(a.output)
    schedule=[d for d in (65536,131072,196608,a.max_depth) if d<=a.max_depth]
    schedule=list(dict.fromkeys(schedule)); rows=[]
    state={"generated_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"profile":a.profile,"configured_max_depth":a.max_depth,"depth_schedule":schedule,"rows":rows,"passed":False}
    atomic(out,state)
    for depth in schedule:
        labels=list(FRACTIONS) if depth==a.max_depth else ["middle"]
        for label in labels:
            prompt,tok,codes=prompt_for(base,depth,label,FRACTIONS[label])
            status,body,elapsed=completion(base,prompt,128)
            choice=(body.get("choices") or [{}])[0]; msg=choice.get("message") or {}; content=(msg.get("content") or "")+(msg.get("reasoning") or "")
            usage=body.get("usage") or {}; passed=status==200 and all(c in content for c in codes) and usage.get("prompt_tokens")==tok
            row={"kind":"needle","depth":depth,"position":label,"tokenized_prompt_tokens":tok,"reported_prompt_tokens":usage.get("prompt_tokens"),"completion_tokens":usage.get("completion_tokens"),"elapsed_s":elapsed,"finish_reason":choice.get("finish_reason"),"content":content,"codes":codes,"http_status":status,"passed":passed}
            rows.append(row); atomic(out,state); print(json.dumps({k:row[k] for k in ("kind","depth","position","tokenized_prompt_tokens","elapsed_s","passed")}),flush=True)
            if not passed: return 1
    prompt,tok,codes=prompt_for(base,a.max_depth,"dual",0.5,dual=True)
    status,body,elapsed=completion(base,prompt,128); choice=(body.get("choices") or [{}])[0]; msg=choice.get("message") or {}; content=(msg.get("content") or "")+(msg.get("reasoning") or ""); usage=body.get("usage") or {}
    passed=status==200 and f"{codes[1]} {codes[0]}" in content and usage.get("prompt_tokens")==tok
    row={"kind":"dual_code","depth":a.max_depth,"position":"middle","tokenized_prompt_tokens":tok,"reported_prompt_tokens":usage.get("prompt_tokens"),"completion_tokens":usage.get("completion_tokens"),"elapsed_s":elapsed,"finish_reason":choice.get("finish_reason"),"content":content,"codes":codes,"http_status":status,"passed":passed}
    rows.append(row); atomic(out,state); print(json.dumps({k:row[k] for k in ("kind","depth","position","tokenized_prompt_tokens","elapsed_s","passed")}),flush=True)
    if not passed: return 1
    long_question="\nContinue with a technically accurate discussion of native quantized inference until the generation limit."
    long_prompt=prompt.rsplit("\n",1)[0]+long_question
    long_tok=tokenize(base,long_prompt)
    status,body,elapsed=completion(base,long_prompt,512,512,True); choice=(body.get("choices") or [{}])[0]; usage=body.get("usage") or {}; content=((choice.get("message") or {}).get("content") or "")
    passed=status==200 and usage.get("prompt_tokens")==long_tok and (usage.get("completion_tokens") or 0)>=512 and bool(content)
    row={"kind":"forced_512","depth":a.max_depth,"position":"middle","tokenized_prompt_tokens":long_tok,"reported_prompt_tokens":usage.get("prompt_tokens"),"completion_tokens":usage.get("completion_tokens"),"elapsed_s":elapsed,"finish_reason":choice.get("finish_reason"),"content_prefix":content[:500],"http_status":status,"passed":passed}
    rows.append(row); state["passed"]=all(r["passed"] for r in rows); state["completed_at_utc"]=dt.datetime.now(dt.timezone.utc).isoformat(); atomic(out,state)
    print(json.dumps({k:row[k] for k in ("kind","depth","position","tokenized_prompt_tokens","completion_tokens","elapsed_s","passed")}),flush=True)
    return 0 if state["passed"] else 1

if __name__=="__main__": raise SystemExit(main())
