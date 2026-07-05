#!/usr/bin/env python3
"""Honest DFlash bench: varied prompt categories, per-category decode tok/s + acceptance metrics."""
import json, time, urllib.request, sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "MiMo-V2.5-NVFP4-DFlash"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 512
THINKING = (sys.argv[4].lower() == "on") if len(sys.argv) > 4 else False

PROMPTS = {
    "narrative": "Write a vivid two-paragraph story about a lighthouse keeper who discovers a message in a bottle.",
    "math": "Solve step by step: A train leaves city A at 60 mph. Two hours later a second train leaves A at 90 mph on a parallel track. How long until the second train catches the first, and how far from A? Show all work.",
    "code": "Write a Python function that parses an Apache access log line into a dict, with a regex, type hints, and 3 unit tests using pytest.",
    "json": "Produce a JSON object describing a fictional bookstore inventory: 5 books, each with title, author, isbn, price, stock, tags (array). Output only JSON.",
    "structured_json": "Produce a JSON array of 40 objects, each with fields id (int), name (string), price (float), tags (array of 2 strings). Output only JSON.",
    "comms": "Draft a professional email to a vendor asking for a revised quote on 40 office chairs, referencing a 15% competitor discount and a Q3 delivery deadline.",
}

def metrics():
    try:
        with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
            return r.read().decode()
    except Exception:
        return ""

def spec_counters(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("#"): continue
        for key in ("spec_decode_num_draft_tokens", "spec_decode_num_accepted_tokens", "spec_decode_num_drafts"):
            if line.startswith("vllm:" + key):
                try: out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError: pass
    return out

results = {}
for name, prompt in PROMPTS.items():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": THINKING},
        "repetition_penalty": 1.0,
    }).encode()
    m0 = spec_counters(metrics())
    t0 = time.time()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    dt = time.time() - t0
    m1 = spec_counters(metrics())
    usage = resp.get("usage", {})
    ct = usage.get("completion_tokens", 0)
    drafts = m1.get("spec_decode_num_drafts", 0) - m0.get("spec_decode_num_drafts", 0)
    drafted = m1.get("spec_decode_num_draft_tokens", 0) - m0.get("spec_decode_num_draft_tokens", 0)
    accepted = m1.get("spec_decode_num_accepted_tokens", 0) - m0.get("spec_decode_num_accepted_tokens", 0)
    accept_len = (accepted / drafts + 1) if drafts else None
    results[name] = {
        "completion_tokens": ct, "wall_s": round(dt, 2),
        "tok_per_s": round(ct / dt, 2) if dt else None,
        "drafts": drafts, "drafted": drafted, "accepted": accepted,
        "mean_accept_len": round(accept_len, 2) if accept_len else None,
        "preview": (resp["choices"][0]["message"].get("content") or "")[:120].replace("\n", " "),
    }
    print(f"[{name}] {results[name]['tok_per_s']} tok/s, accept_len={results[name]['mean_accept_len']}, tokens={ct}", flush=True)

vals = [r["tok_per_s"] for r in results.values() if r["tok_per_s"]]
print(json.dumps(results, indent=2))
print(f"RANGE: {min(vals):.1f} - {max(vals):.1f} tok/s, mean {sum(vals)/len(vals):.1f}")
