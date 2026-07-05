# Measured results — MiMo-V2.5 NVFP4 weights + NVFP4 KV + DFlash, TP=2, 2x DGX Spark (GB10)

> ## ALL GO-LIVE NUMBERS PENDING — serve loading now (2026-07-05)
>
> The 500K long-context NVFP4-KV serve (NCCL LL tuning, async scheduling on,
> thinking off, rp 1.0) is booting as of 2026-07-05. This file gets the verified
> numbers when the bench completes. Until then, only the **historical 8K-config
> reference** at the bottom of this file is measured fact for this lane.

Bench protocol (same as the FP8 sibling): single stream, temp 0, 512 max tokens,
eager, `completion_tokens / wall` via [`dflash_bench.py`](dflash_bench.py), thinking
off, repetition_penalty 1.0. Acceptance truth from engine-log `SpecDecoding metrics`
lines (with async scheduling on, the prometheus drafts counter under-counts and
inflates client-computed accept_len).

## Go-live checkpoint (PENDING)

**Config:** NVFP4 4-bit target KV (`triton_attn_diffkv` + WMMA), drafter on stock
`triton_attn` with bf16 KV, 500K max context, GMU 0.83, seqs 6, batched 4096, TP2
Bluey+Reddie, NCCL LL tuning.

| metric | value |
| --- | ---: |
| KV pool @ 500K cfg | **PENDING** (expected ≈2x the FP8 sibling's 980,748 tokens) |
| max concurrency for 500K streams | PENDING |

6-category bench, 512 tok, temp 0, single stream:

| workload | tok/s |
| --- | ---: |
| structured JSON (40-object array) | PENDING |
| math (step-by-step) | PENDING |
| json (short varied) | PENDING |
| code | PENDING |
| comms (email) | PENDING |
| narrative prose | PENDING |
| **range / mean** | **PENDING** |

Engine-log acceptance at the structured-JSON operating point: PENDING.

## Comparison vs the FP8 sibling (fill in when benched)

| deploy | KV | max ctx | KV pool | best structured-JSON tok/s |
| --- | --- | ---: | ---: | ---: |
| [FP8 sibling](https://github.com/tonyd2wild/MiMo-V2.5-DFlash-FP8-KV-2x-DGX-Spark), go-live 2026-07-05 | fp8 | 500K | 980,748 tokens | 62.9 |
| **this repo (NVFP4 KV)** | **nvfp4 (4-bit)** | **500K** | **PENDING** | **PENDING** |

The claim to verify: ~2x the FP8 pool (→ ~2M-token class on 2 Sparks) at
near-identical decode speed.

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
temp 0 draft best; free-form prose at temp > 0 drafts worst. When the go-live numbers
land, quote the **range and mean, not the peak** — same rule as the FP8 sibling
(whose go-live is 18.2–62.9, mean 38.0).
