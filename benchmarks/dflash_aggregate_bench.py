#!/usr/bin/env python3
"""Aggregate (concurrent-stream) DFlash bench: C-sweep, mixed workloads, temp 0, thinking off."""
import json, time, sys, threading, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "MiMo-V2.5-NVFP4-DFlash-NVFP4KV"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 512
LEVELS = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else [1, 2, 4, 6])]

PROMPTS = [
    "Produce a JSON array of 40 objects, each with fields id (int), name (string), price (float), tags (array of 2 strings). Output only JSON.",
    "Solve step by step: A train leaves city A at 60 mph. Two hours later a second train leaves A at 90 mph. How long until it catches the first, and how far from A? Show all work.",
    "Write a Python function that parses an Apache access log line into a dict, with a regex, type hints, and 3 unit tests using pytest.",
    "Draft a professional email to a vendor asking for a revised quote on 40 office chairs, referencing a 15% competitor discount and a Q3 delivery deadline.",
    "Produce a JSON object describing a fictional bookstore inventory: 5 books with title, author, isbn, price, stock, tags. Output only JSON.",
    "Explain the difference between TCP and UDP in a structured list of 6 numbered points with a one-line summary each.",
]

def one_request(idx, results):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[idx % len(PROMPTS)]}],
        "max_tokens": MAXTOK, "temperature": 0, "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    results[idx] = (resp["usage"]["completion_tokens"], time.time() - t0)

for c in LEVELS:
    results = {}
    threads = [threading.Thread(target=one_request, args=(i, results)) for i in range(c)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    total = sum(v[0] for v in results.values())
    per = [round(v[0] / v[1], 1) for v in results.values()]
    print(f"C{c}: aggregate {total/wall:.1f} tok/s ({total} tokens in {wall:.1f}s), per-stream {per}", flush=True)
