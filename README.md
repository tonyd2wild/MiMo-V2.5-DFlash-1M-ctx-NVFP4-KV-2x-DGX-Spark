# MiMo-V2.5 + DFlash + NVFP4 4-bit KV Cache on 2x DGX Spark — Long-Context Variant

Self-contained two-node DGX Spark recipe for serving **MiMo-V2.5** with:

- **NVFP4 4-bit weights** — [`mitomtuna/MiMo-V2.5-0703-NVFP4`](https://huggingface.co/mitomtuna/MiMo-V2.5-0703-NVFP4) (171G, fits TP=2 across two GB10 Sparks) — the **drafter-matched 2026-07-03 target** (+13% structured-output speed vs the April-base [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4), which remains a supported alternate)
- **NVFP4 4-bit KV cache** — the target's KV runs 4-bit via the `triton_attn_diffkv` backend
- **Xiaomi DFlash speculative decoding** — the official drafter from
  [`XiaomiMiMo/MiMo-V2.5-DFlash`](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-DFlash), `dflash/` subdir only
- **vLLM TP=2 over Ray** across Bluey + Reddie (GB10, direct RoCE fabric)

**This is the CONTEXT story.** The sibling repo,
[MiMo-V2.5-DFlash-FP8-KV-2x-DGX-Spark](https://github.com/tonyd2wild/MiMo-V2.5-DFlash-FP8-KV-2x-DGX-Spark),
is the speed story: identical patch/recipe stack, FP8 KV, 500K ctx, 980,748-token pool,
62.9 tok/s structured JSON. This variant swaps the target KV to 4-bit NVFP4 —
shipping config verified 2026-07-05: **1M context per request, 3.43 concurrent
full-1M streams, a 3,427,495-token KV pool on two Sparks — 53 tok/s single-stream
structured JSON, 68.8 tok/s aggregate at 6 streams, mean 37.8 tok/s (identical to the
500K shape and within 1% of the FP8 sibling).**

To our knowledge this is the **first public recipe running NVFP4 4-bit weights +
NVFP4 4-bit KV + DFlash speculative decoding together**.

> **Grab only the drafter, not the whole repo.** `XiaomiMiMo/MiMo-V2.5-DFlash` bundles a
> 311GB fp8 copy of the target model you do not need — the drafter itself is 2.8G:
>
> ```bash
> hf download XiaomiMiMo/MiMo-V2.5-DFlash --include "dflash/*"
> ```

## Hardware

| role | node | GPU | fabric IP | notes |
| --- | --- | --- | --- | --- |
| Ray head (rank0, runs vLLM serve) | Bluey | GB10 | 192.168.192.1 | `NCCL_IB_GID_INDEX=3` |
| Ray worker (rank1) | Reddie | GB10 | 192.168.192.2 | historically GID 5 — see DEFAULT-CONFIG gotchas |

Two NVIDIA DGX Sparks (GB10, 128G unified memory each), direct-cabled RoCE on
`enp1s0f0np0` / `rocep1s0f0`, `192.168.192.0/24`.

## How the 4-bit KV lane works

The target and the drafter deliberately run on **different attention backends with
different KV dtypes**, split by vLLM's hybrid KV cache manager:

- **Target layers** → `triton_attn_diffkv` with `--kv-cache-dtype nvfp4` (packed uint8,
  WMMA decode). MiMo-V2.5's DiffKV geometry (K=192 + V=128) compresses to a 4-bit
  cache at **3.2x the fp8 pool** (measured: 3,167,247 vs 980,748 tokens at the
  identical 500K config).
- **Drafter layers** → stock `triton_attn` with **bf16 KV in their own hybrid cache
  group** (set via `"attention_backend":"triton_attn"` inside the speculative-config
  JSON — the launcher's `SPEC_ATTENTION_BACKEND` env). The drafter's KV is SWA-1024
  bounded, so keeping it bf16 costs almost nothing — **and this split is also the fix
  for "the drafter eats the KV pool"**: without it, the global nvfp4 override clobbers
  the drafter's allocation and the engine crashes at KV init.

## Why this needed engine surgery: the 4 problems

Serving Xiaomi's DFlash drafter against the NVFP4 target on GB10 required four engine
fixes on top of the base image. All are in [`patches/`](patches/) as idempotent
apply-in-container scripts (each has a docstring explaining the mechanics).

1. **GB10 is FA2-only → no attention sinks in flash-attn.** The DFlash drafter needs
   non-causal block drafting + attention sinks + SWA. flash-attn on GB10 (FA2) has no
   sink support, so the drafter must run on the **Triton backend** — which lacked
   non-causal support. [`patch_triton_noncausal.py`](patches/patch_triton_noncausal.py)
   ports upstream-main's `USE_CAUSAL` semantics into the image's triton unified
   attention stack, and [`patch_nc_fix.py`](patches/patch_nc_fix.py) fixes two
   correctness bugs in that port (non-causal keys must be bounded by `seq_len` —
   tile-overhang junk was polluting the softmax — and non-causal SWA must skip the
   first-query V-zeroing). Before the nc fix: acceptance 2.1. A kernel probe validated
   all 8 causal/sink/SWA combinations.

2. **The hybrid KV cache manager must stay ENABLED.** The drafter runs SWA-1024 while
   the target mixes attention window types; they need separate KV groups.
   `--disable-hybrid-kv-cache-manager` breaks the deploy — don't pass it. (In this
   variant the split works even harder: it is what lets the target run 4-bit KV while
   the drafter stays bf16.)

3. **DiffKV target pages vs power-of-2 drafter pages.** MiMo-V2.5 target layers use
   DiffKV geometry (K=192 + V=128 → per-token bytes carry a factor of 5) while the
   drafter's pages are standard powers of two. Neither divides the other, so vLLM's
   stock "grow smaller pages to the max" unification fails.
   [`patch_kv_page_lcm.py`](patches/patch_kv_page_lcm.py) scales every group's block
   size to the **LCM of all page sizes** so every ratio is an integer.

4. **Aux-hidden-layer off-by-one (+1 semantics) — the big one.** DFlash's
   `target_layer_ids` mean "hidden state AFTER layer i"; vLLM's aux-hidden-state mixin
   indexes "state ENTERING layer k". The image's runner passed the ids through raw, so
   all 5 drafter feature taps were one layer early.
   [`patch_aux_layer_off_by_one.py`](patches/patch_aux_layer_off_by_one.py) backports
   upstream's +1 conversion. **This fix alone took mean acceptance 2.14 → 3.78 and
   22.6 → 35.2 tok/s.**

### The NVFP4-KV-specific extras

On top of the shared four, this variant needs three more patches plus one mod unlock
(all already in [`patches/`](patches/) and [`recipe/`](recipe/)):

- [`patch_diffkv_noncausal.py`](patches/patch_diffkv_noncausal.py) — threads
  `USE_CAUSAL` through the diffkv triton kernel so the global non-causal selector gate
  is satisfied when DFlash is active (target layers still run causal at runtime).
- [`patch_draft_cache_auto.py`](patches/patch_draft_cache_auto.py) — pins the DFlash
  drafter's KV cache dtype to `auto`: the drafter's stock-triton backend cannot consume
  the diffkv NVFP4 KV layout, and its SWA-bounded KV is tiny anyway.
- [`patch_spec_dtype_guard.py`](patches/patch_spec_dtype_guard.py) — scopes the global
  nvfp4 KV spec override to **DiffKV-backed layers only**. Without this the override
  clobbers the drafter's layers and the engine dies on a spec-vs-view size mismatch at
  KV init (92160 vs 163840 bytes/block).
- **`EXPERIMENTAL_ALLOW_DIFFKV_QUANT_KV`** — the `fix-mimo-v2-vllm` mod (applied by
  `recipe/apply-mods.sh`) is what unblocks quantized KV on the `triton_attn_diffkv`
  backend in the first place.

## Results — go-live verified 2026-07-05

Boot evidence at the 500K config (GMU 0.83, seqs 6):

```text
GPU KV cache size: 3,167,247 tokens
Maximum concurrency for 500,000 tokens per request: 6.33x
```

**3.17M tokens — 3.2x the FP8 sibling's 980,748 pool at the identical shape.**

6-category bench (512 tok, temp 0, thinking off, rp 1.0, single stream, NCCL LL
tuning) vs the FP8 sibling:

| workload | NVFP4 KV tok/s | FP8 sibling | delta |
| --- | ---: | ---: | ---: |
| structured JSON (40-object array) | 55.4 | 62.9 | −12% |
| json (short varied) | 45.0 | 42.4 | +6% |
| math (step-by-step) | 44.1 | 43.3 | ~even |
| code | 32.2 | 33.9 | −5% |
| comms (email) | 29.2 | 27.5 | +6% |
| narrative prose | 19.8 | 18.2 | +9% |
| **mean** | **37.6** | **38.0** | **−1%** |

The mean holds within 1% of FP8; only the structured-JSON peak pays a ~12% dequant
tax. **The trade: FP8 = peak speed (63), NVFP4 = 3.2x the context at the same average
speed.**

**Tool calling verified on this serve:** `tools` + `tool_choice:"auto"` returned a
clean `get_weather` tool_call via the mimo parser.

**Shipping config — 1M context (verified):** relaunched at `MAX_MODEL_LEN=1000000`,
`MAX_NUM_SEQS=6`: pool **3,427,495 tokens (3.43x full-1M streams)**, single-stream
mean **37.8** (structured JSON 53.0), aggregate **68.8 tok/s at C6**. The
million-token config costs nothing vs the 500K shape.

The honesty rule carries over from the sibling repo: DFlash speedup is
workload-shaped — **report the range (19.4–53.0 single-stream, mean 37.8, 68.8
aggregate at C6), not the peak**. Full
details and the historical 8K-config reference in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

## Files

| path | purpose |
| --- | --- |
| `DEFAULT-CONFIG.md` | byte-exact working launch flow: containers → mods → patches → Ray → serve (NVFP4-KV lane) |
| `patches/Dockerfile` | overlay image: base + vllm-main DFlash model/proposer + eagle3 wiring |
| `patches/qwen3_dflash.py` | DFlash drafter model (backported from vLLM main, Apache-2.0 SPDX kept) |
| `patches/dflash_proposer.py` | DFlash proposer (backported from vLLM main, Apache-2.0 SPDX kept) |
| `patches/patch_mimo_v2_eagle3.py` | wires SupportsEagle3 / aux-hidden-state taps into mimo_v2.py |
| `patches/patch_triton_noncausal.py` | USE_CAUSAL port into triton unified attention (problem 1) |
| `patches/patch_nc_fix.py` | non-causal correctness: seq_len bound + SWA V-zeroing skip (problem 1) |
| `patches/patch_kv_page_lcm.py` | LCM-based KV page-size unification (problem 3) |
| `patches/patch_aux_layer_off_by_one.py` | +1 aux-layer semantics — acceptance 2.14→3.78 (problem 4) |
| `patches/patch_diffkv_noncausal.py` | USE_CAUSAL in the diffkv kernel (**this variant, required**) |
| `patches/patch_draft_cache_auto.py` | drafter KV pinned to auto dtype (**this variant, required**) |
| `patches/patch_spec_dtype_guard.py` | nvfp4 spec override scoped to DiffKV layers (**this variant, required**) |
| `recipe/run-container.sh` | start the patched vLLM container (both nodes) |
| `recipe/apply-mods.sh` | apply the 6 base mods (both nodes) — registers MimoV2Config, unlocks diffkv quant KV |
| `recipe/env.sh` | full environment: serving shape, Ray/memory stability, NCCL/RoCE |
| `recipe/run-head.sh` / `recipe/run-worker.sh` | Ray bring-up with the 1 GiB object-store cap |
| `recipe/launch-dflash-fp8ckpt.sh` | **the launcher** (shared with the FP8 sibling; this variant = env overrides) |
| `recipe/launch-dflash-v2.sh` | earlier DFlash launcher (8K probe phase) |
| `recipe/launch-dflash.sh` | reference: the MTP/NVFP4-KV launcher the recipe evolved from |
| `benchmarks/dflash_bench.py` | honest per-category bench (tok/s + acceptance from /metrics deltas) |
| `benchmarks/RESULTS.md` | verified go-live numbers + FP8 comparison + historical 8K-config reference |

## Quick start

See [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md) for the byte-exact flow. Summary:

1. Both nodes: `run-container.sh` (overlay image built from `patches/Dockerfile`), then
   `apply-mods.sh`, then apply the `patch_*.py` engine patches in-container — for this
   variant the three quantized-KV patches are REQUIRED, not optional.
2. Start Ray on the **head first** (`run-head.sh` on Bluey), then join the worker
   (`run-worker.sh` on Reddie), wait for `2.0/2.0 GPU` in `ray status`.
3. On the head: `source env.sh`, export the model paths, and run
   `launch-dflash-fp8ckpt.sh` with the NVFP4-KV overrides — the ones that differ from
   the FP8 sibling are `KV_CACHE_DTYPE=nvfp4 ATTENTION_BACKEND=triton_attn_diffkv
   SPEC_ATTENTION_BACKEND=triton_attn`.
4. Verify with `benchmarks/dflash_bench.py`.

## Getting the speed: the three unlocks (carried over from the FP8 sibling)

1. **Disable thinking for throughput-critical serving** — send
   `"chat_template_kwargs": {"enable_thinking": false}`. Reasoning prose drafts at
   ~2 accept length and drags every workload down.
2. **No repetition penalty (1.0).** A penalty pushes the verifier away from
   correctly-repetitive drafts exactly on the structured output where DFlash shines
   (57.3 → 63.8 tok/s on structured JSON in the FP8 lane).
3. **`NCCL_PROTO=LL` + `NCCL_MAX_NCHANNELS=2`** on the cross-node TP2 link
   (danielgbates' insight) — targets the small-message all-reduce latency that
   dominates the ~100ms DFlash step.

## Credits

This recipe stands on prior public work:

- **Xiaomi MiMo team** — the MiMo-V2.5 model and the official DFlash drafter
  ([XiaomiMiMo/MiMo-V2.5-DFlash](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-DFlash)).
- **mitomtuna** — the drafter-matched NVFP4/MXFP8 quant of the 0703 target refresh
  ([mitomtuna/MiMo-V2.5-0703-NVFP4](https://huggingface.co/mitomtuna/MiMo-V2.5-0703-NVFP4))
- **lukealonso** — the NVFP4 quantization of the target
  ([lukealonso/MiMo-V2.5-NVFP4](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4)),
  which is what makes a 2-Spark TP=2 deploy possible at all.
- **The vLLM project** — the upstream DFlash implementation this repo backports from
  main (PRs [#45200](https://github.com/vllm-project/vllm/pull/45200),
  [#45181](https://github.com/vllm-project/vllm/pull/45181),
  [#46104](https://github.com/vllm-project/vllm/pull/46104) lineage).
  `qwen3_dflash.py` and `dflash_proposer.py` are vLLM-derived and keep their
  Apache-2.0 SPDX headers.
- **danielgbates** — independent MiMo-V2.5 + DFlash 2x-Spark recipe on the NVIDIA
  forums, and in particular the **`NCCL_PROTO=LL` + `NCCL_MAX_NCHANNELS=2`** small-message
  latency insight adopted here
  ([forums.developer.nvidia.com/t/375607](https://forums.developer.nvidia.com/t/375607)).
  His recipe runs bf16 KV — the 4-bit-KV lane in this repo is the part he did not touch.
- **z-lab** — the Qwen3.6 DFlash precedent that proved the DFlash engine path works on
  these Sparks.
- **renek** — the 60+ tok/s existence proof that set the target.
- Upstream vLLM, Triton, FlashInfer, NVIDIA Blackwell/CUDA/NCCL tooling.

Our contribution: the GB10/FA2 triton non-causal port and its correctness fixes, the
LCM page unification, the aux-layer off-by-one diagnosis and backport, and the
quantized-KV coexistence patches that make NVFP4 weights + NVFP4 KV + DFlash run
together — which we believe is a first.

## License

Repo scripts and docs: [Apache-2.0](LICENSE). The vLLM-derived files
(`patches/qwen3_dflash.py`, `patches/dflash_proposer.py`) retain their upstream
Apache-2.0 SPDX headers. Base images, model weights, and NVIDIA tooling are separate
upstream artifacts with their own licenses and terms.
