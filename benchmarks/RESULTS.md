# Measured results — MiMo-V2.5 NVFP4 weights + NVFP4 KV + DFlash, TP=2, 2x DGX Spark (GB10)

Bench protocol (same as the FP8 sibling): single stream, temp 0, 512 max tokens,
eager, `completion_tokens / wall` via [`dflash_bench.py`](dflash_bench.py), thinking
off, repetition_penalty 1.0. Acceptance truth from engine-log `SpecDecoding metrics`
lines — **with async scheduling on, the prometheus drafts counter under-counts and
inflates client-computed accept_len** (client values like 10.34 are inflated; the
engine log is truth).

Benches are request-accounted against the server log (expected-vs-actual request
delta) to rule out concurrent-traffic contamination — the first 1M-config run was
discarded for exactly this and re-run clean.

## 1M-context go-live — VERIFIED 2026-07-05 (shipping config)

**Config:** NVFP4 4-bit target KV (`triton_attn_diffkv` + WMMA), drafter on stock
`triton_attn` with bf16 KV, **1M max context**, GMU 0.83, seqs 6, batched 4096, TP2
Bluey+Reddie, NCCL LL tuning, async scheduling on.

Boot evidence:

```text
GPU KV cache size: 3,427,495 tokens
Maximum concurrency for 1,000,000 tokens per request: 3.43x
```

Single-stream, 512 tok, temp 0, thinking off, rp 1.0:

| workload | tok/s |
| --- | ---: |
| structured JSON (40-object array) | **53.0** |
| json (short varied) | 45.7 |
| math (step-by-step) | 45.3 |
| code | 35.8 |
| comms (email) | 27.6 |
| narrative prose | 19.4 |
| **mean** | **37.8** |

**The million-token config is free:** mean at 1M ctx (37.8) matches the 500K shape
(37.6) and the FP8 sibling (38.0).

Aggregate (concurrent mixed-workload) sweep via
[`dflash_aggregate_bench.py`](dflash_aggregate_bench.py), 512 tok/stream:

| streams | aggregate tok/s |
| --- | ---: |
| C1 | 62.1 |
| C2 | 62.3 |
| C4 | 61.2 |
| **C6** | **68.8** |

Honest note: per-stream speculation dilutes under batched verification (C2–C4 flat);
aggregate peaks at C6 = **68.8 tok/s of total box throughput**.

## 500K-shape checkpoint — VERIFIED 2026-07-05

**Config:** NVFP4 4-bit target KV (`triton_attn_diffkv` + WMMA), drafter on stock
`triton_attn` with bf16 KV, 500K max context, GMU 0.83, seqs 6, batched 4096, TP2
Bluey+Reddie, NCCL LL tuning, async scheduling on.

Boot evidence:

```text
GPU KV cache size: 3,167,247 tokens
Maximum concurrency for 500,000 tokens per request: 6.33x
```

**3.17M tokens — 3.2x the FP8 sibling's 980,748 pool at the identical 500K/GMU-0.83
shape.** The conservative 2x projection undersold it.

6-category bench, 512 tok, temp 0, thinking off, rp 1.0, single stream, same NCCL LL
tuning as the FP8 go-live:

| workload | NVFP4 KV tok/s | FP8 sibling | delta |
| --- | ---: | ---: | ---: |
| structured JSON (40-object array) | 55.4 | 62.9 | −12% |
| json (short varied) | 45.0 | 42.4 | +6% |
| math (step-by-step) | 44.1 | 43.3 | ~even |
| code | 32.2 | 33.9 | −5% |
| comms (email) | 29.2 | 27.5 | +6% |
| narrative prose | 19.8 | 18.2 | +9% |
| **mean** | **37.6** | **38.0** | **−1%** |

**Framing:** the mean holds within 1% of FP8; only the structured-JSON peak pays a
~12% dequant tax. The trade in one line: **FP8 = peak speed (63), NVFP4 = 3.2x the
context at the same average speed.**

**Tool calling verified on this exact serve:** an OpenAI-style request with `tools` +
`tool_choice:"auto"` returned a clean `get_weather` tool_call via the mimo parser.

## Comparison vs the FP8 sibling

| deploy | KV | max ctx | KV pool | best structured-JSON tok/s | mean tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| [FP8 sibling](https://github.com/tonyd2wild/MiMo-V2.5-DFlash-FP8-KV-2x-DGX-Spark), go-live 2026-07-05 | fp8 | 500K | 980,748 tokens | 62.9 | 38.0 |
| this repo, 500K shape 2026-07-05 | nvfp4 (4-bit) | 500K | 3,167,247 tokens (3.2x) | 55.4 | 37.6 |
| **this repo, 1M shipping config 2026-07-05** | **nvfp4 (4-bit)** | **1M** | **3,427,495 tokens (3.5x)** | 53.0 | **37.8** |

## Historical reference — 8K test config (measured 2026-07-04, pre-NCCL-tuning)

The engine stack for this lane is already proven live; these are the 8K-config
numbers from that first serve (eager, async off, rp 1.08, pre-LL-tuning — an older,
slower operating point than the go-live protocol above):

| category | tok/s |
| --- | ---: |
| math | 42.7 |
| json | 40.8 |
| code | 38.9 |
| comms | 26.2 |
| narrative | 19.4 |
| **mean** | **33.6** |

- accept len: 3.29–4.36 across categories
- **KV pool: 250,790 tokens @ 8K cfg — ~2.3x the fp8 pool at the same config**
  (fp8 measured ~108K at that config), with speed holding within 2% of fp8
- To our knowledge this was the first NVFP4-weights + NVFP4-KV + DFlash serve.

## Honest framing

Speedup is workload-dependent because acceptance tracks output structure: JSON/math at
temp 0 draft best; free-form prose at temp > 0 drafts worst. Quote the shipping (1M)
config as **19.4–53.0 tok/s single-stream depending on workload, mean 37.8, up to 68.8
aggregate at 6 streams** — not the peak. Same rule as the FP8 sibling (18.2–62.9, mean
38.0).
