# MiMo-V2.5 + DFlash + NVFP4 4-bit KV on 2x DGX Spark — 1M context · 3.43M-token KV pool · 37.8 tok/s

> MiMo-V2.5 served with NVFP4 4-bit weights, a NVFP4 4-bit KV cache, and Xiaomi DFlash speculative decoding across two GB10 DGX Sparks (vLLM TP=2 over Ray) — 1M context per request, a 3,427,495-token KV pool (3.43 concurrent full-1M streams), mean 37.8 tok/s.

## TL;DR

- **What you get:** a self-contained two-node DGX Spark recipe serving **MiMo-V2.5** with **NVFP4 4-bit weights**, a **NVFP4 4-bit KV cache** (the target's KV runs 4-bit via the `triton_attn_diffkv` backend), and the official Xiaomi **DFlash** speculative-decoding drafter, on **vLLM TP=2 over Ray** across Bluey + Reddie (GB10, direct RoCE fabric). To our knowledge this is the **first public recipe running NVFP4 4-bit weights + NVFP4 4-bit KV + DFlash speculative decoding together**.
- **The numbers (shipping config, verified 2026-07-05):** 1M context per request, **3.43 concurrent full-1M streams**, a **3,427,495-token KV pool** on two Sparks — **53 tok/s single-stream structured JSON, 68.8 tok/s aggregate at 6 streams, mean 37.8 tok/s** (identical to the 500K shape and within 1% of the FP8 sibling). On the drafter-matched 0703 target, structured JSON clears **60 tok/s**.
- **This is the CONTEXT story.** The sibling repo, [MiMo-V2.5-DFlash-FP8-KV-2x-DGX-Spark](https://github.com/tonyd2wild/MiMo-V2.5-DFlash-FP8-KV-2x-DGX-Spark), is the speed story: identical patch/recipe stack, FP8 KV, 500K ctx, 980,748-token pool, 62.9 tok/s structured JSON. This variant swaps the target KV to 4-bit NVFP4 for **3.2x the context at the same average speed**.
- **Who it's for:** anyone reproducing long-context MiMo-V2.5 on two GB10 Sparks who wants the maximum KV pool without giving up DFlash speculative decoding.

> **Grab only the drafter, not the whole repo.** `XiaomiMiMo/MiMo-V2.5-DFlash` bundles a 311GB fp8 copy of the target model you do not need — the drafter itself is 2.8G:
>
> ```bash
> hf download XiaomiMiMo/MiMo-V2.5-DFlash --include "dflash/*"
> ```

## Hardware

Two NVIDIA DGX Sparks (GB10, 128G unified memory each), direct-cabled RoCE on `enp1s0f0np0` / `rocep1s0f0`, `192.168.192.0/24`.

| role | node | GPU | fabric IP | notes |
| --- | --- | --- | --- | --- |
| Ray head (rank0, runs vLLM serve) | Bluey | GB10 | 192.168.192.1 | `NCCL_IB_GID_INDEX=3` |
| Ray worker (rank1) | Reddie | GB10 | 192.168.192.2 | historically GID 5 — see DEFAULT-CONFIG gotchas |

## Quick start

Full byte-exact flow in [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md). Order matters — **containers on both nodes → mods → engine patches → Ray head → Ray worker → serve from the head**:

1. **Both nodes:** `recipe/run-container.sh` (overlay image built from `patches/Dockerfile`), then `recipe/apply-mods.sh`, then apply the `patch_*.py` engine patches in-container. For this variant the three quantized-KV patches are **REQUIRED**, not optional.
2. Start Ray on the **head first** (`recipe/run-head.sh` on Bluey), then join the worker (`recipe/run-worker.sh` on Reddie); wait for `2.0/2.0 GPU` in `ray status`.
3. On the head: `source recipe/env.sh`, export the model paths, and run `recipe/launch-dflash-fp8ckpt.sh` with the NVFP4-KV overrides — the ones that differ from the FP8 sibling are `KV_CACHE_DTYPE=nvfp4 ATTENTION_BACKEND=triton_attn_diffkv SPEC_ATTENTION_BACKEND=triton_attn`.
4. Smoke-test, then bench with `benchmarks/dflash_bench.py`.

```bash
# smoke test
curl http://192.168.192.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-V2.5-NVFP4-DFlash-NVFP4KV","messages":[{"role":"user","content":"Reply exactly: OK DFLASH NVFP4KV LIVE"}],"max_tokens":16,"temperature":0}'
```

## Setup (detailed)

### Weights

- **Target (recommended):** [`mitomtuna/MiMo-V2.5-0703-NVFP4`](https://huggingface.co/mitomtuna/MiMo-V2.5-0703-NVFP4) — 171G, snapshot `75f7cd8403e1285a8cf783e212fa14ce474b814c`. NVFP4/MXFP8 quant of the **drafter-matched 2026-07-03 target refresh**; **+13% on structured output** vs the April base. Drop-in: only `MODEL_PATH` changes.
- **Target (original / supported alternate):** [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4) — 171G, pinned snapshot `a147dd04...`. Both fit TP=2 across two GB10 Sparks and live in the HF cache on **both** nodes.
- **Drafter:** [`XiaomiMiMo/MiMo-V2.5-DFlash`](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-DFlash), **`dflash/` subdir only** — 2.8G, snapshot `1f58446181abcaa01030fdbde835fbd38ae9a2b1`, on **both** nodes:
  ```bash
  hf download XiaomiMiMo/MiMo-V2.5-DFlash --include "dflash/*"
  ```
  Do NOT download the full repo — it carries a 311GB fp8 copy of the target you don't need.

### Image / build

- **Base image:** `vllm-node-mimo-v25-nvfp4:latest` (same build on both Sparks; has the DFlash engine wiring + diffkv KV backends baked in). The `ghcr.io/...nvfp4kv:20260620` image has **no** dflash proposer — don't use it for this.
- **Overlay image:** `vllm-node-mimo-v25-nvfp4:dflash-mimo-20260704`, built on both nodes from [`patches/Dockerfile`](patches/Dockerfile) = base + vllm-main `qwen3_dflash.py` + guarded `dflash_proposer.py` + `patch_mimo_v2_eagle3.py`.
- **vLLM build inside:** `0.21.1rc1.dev85+gd87ee1893` (DEV build, CUDA 13.2, torch 2.11.0). Stock `pip install vllm` will not work. **Ray is required** (the mp executor is single-host).

### Launch

Same launcher as the FP8 sibling — [`recipe/launch-dflash-fp8ckpt.sh`](recipe/launch-dflash-fp8ckpt.sh) — with **three env overrides that define this variant**:

| override | value | effect |
| --- | --- | --- |
| `KV_CACHE_DTYPE` | `nvfp4` | target KV cache in 4-bit NVFP4 (packed uint8) |
| `ATTENTION_BACKEND` | `triton_attn_diffkv` | target layers on the DiffKV backend (WMMA decode) |
| `SPEC_ATTENTION_BACKEND` | `triton_attn` | drafter on stock triton with bf16 KV in its own hybrid cache group |

```bash
# DEFAULT: the drafter-matched 0703 target (recommended). To use the original
# April-base quant instead, point at models--lukealonso--MiMo-V2.5-NVFP4/snapshots/a147dd04d6cf861e43b2d783dcde23b53ab7ee68
export MODEL_PATH=/root/.cache/huggingface/hub/models--mitomtuna--MiMo-V2.5-0703-NVFP4/snapshots/75f7cd8403e1285a8cf783e212fa14ce474b814c
export DFLASH_MODEL_PATH=/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-DFlash/snapshots/1f58446181abcaa01030fdbde835fbd38ae9a2b1/dflash
export HEAD_ROCE_IP=192.168.192.1

MAX_MODEL_LEN=1000000 MAX_NUM_SEQS=6 MAX_NUM_BATCHED_TOKENS=4096 \
BLOCK_SIZE=16 GPU_MEMORY_UTILIZATION=0.83 \
KV_CACHE_DTYPE=nvfp4 ATTENTION_BACKEND=triton_attn_diffkv \
SPEC_ATTENTION_BACKEND=triton_attn \
NO_ASYNC_SCHED=0 \
SERVED_MODEL_NAME=MiMo-V2.5-NVFP4-DFlash-NVFP4KV \
bash /workspace/recipe/launch-dflash-fp8ckpt.sh
```

Boot evidence (shipping 1M config, verified 2026-07-05):

```text
GPU KV cache size: 3,427,495 tokens
Maximum concurrency for 1,000,000 tokens per request: 3.43x
```

**How the 4-bit KV lane works.** The target and the drafter deliberately run on **different attention backends with different KV dtypes**, split by vLLM's hybrid KV cache manager:

- **Target layers** → `triton_attn_diffkv` with `--kv-cache-dtype nvfp4` (packed uint8, WMMA decode). MiMo-V2.5's DiffKV geometry (K=192 + V=128) compresses to a 4-bit cache at **3.2x the fp8 pool** (measured: 3,167,247 vs 980,748 tokens at the identical 500K config).
- **Drafter layers** → stock `triton_attn` with **bf16 KV in their own hybrid cache group** (set via `"attention_backend":"triton_attn"` inside the speculative-config JSON — the launcher's `SPEC_ATTENTION_BACKEND` env). The drafter's KV is SWA-1024 bounded, so keeping it bf16 costs almost nothing — **and this split is also the fix for "the drafter eats the KV pool"**: without it, the global nvfp4 override clobbers the drafter's allocation and the engine crashes at KV init.

### Verify

```bash
python3 benchmarks/dflash_bench.py http://192.168.192.1:8000 MiMo-V2.5-NVFP4-DFlash-NVFP4KV 512
```

**Tool calling** is enabled on the serve (`--enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo`): an OpenAI-style `tools` + `tool_choice:"auto"` request returns a clean `get_weather` tool_call via the mimo parser.

### Repository layout

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
| `benchmarks/dflash_aggregate_bench.py` | concurrent mixed-workload aggregate sweep |
| `benchmarks/RESULTS.md` | verified go-live numbers + FP8 comparison + historical 8K-config reference |

## Benchmarks

Full detail — including the historical 8K-config reference — in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). Protocol: single stream, temp 0, 512 max tokens, eager, thinking off, repetition_penalty 1.0. Acceptance truth from engine-log `SpecDecoding metrics` lines — with async scheduling on, the prometheus drafts counter under-counts and inflates client-computed accept_len.

**Shipping config — 1M context (verified 2026-07-05):** `MAX_MODEL_LEN=1000000`, `MAX_NUM_SEQS=6`, GMU 0.83 — pool **3,427,495 tokens (3.43x full-1M streams)**. The million-token config costs nothing vs the 500K shape.

| workload | tok/s |
| --- | ---: |
| structured JSON (40-object array) | **53.0** |
| json (short varied) | 45.7 |
| math (step-by-step) | 45.3 |
| code | 35.8 |
| comms (email) | 27.6 |
| narrative prose | 19.4 |
| **mean** | **37.8** |

Aggregate (concurrent mixed-workload) via [`benchmarks/dflash_aggregate_bench.py`](benchmarks/dflash_aggregate_bench.py), 512 tok/stream — per-stream speculation dilutes under batched verification (C2–C4 flat), aggregate peaks at C6:

| streams | aggregate tok/s |
| --- | ---: |
| C1 | 62.1 |
| C2 | 62.3 |
| C4 | 61.2 |
| **C6** | **68.8** |

**Drafter-matched target (verified 2026-07-09, recommended):** the 0703 quant lifts structured JSON to **59.1–60.8 tok/s (+13%, repeatable)** while keeping the big pool — boot evidence `GPU KV cache size: 3,269,303 tokens` / `3.27x` (slightly smaller than the lukealonso target, which keeps attention in MXFP8). The 4-bit-KV 1M build now clears **60 tok/s** on structured output — previously FP8-only territory.

**500K shape (verified 2026-07-05)** — pool **3,167,247 tokens / 6.33x**, 3.2x the FP8 sibling's 980,748 at the identical 500K/GMU-0.83 shape:

| workload | NVFP4 KV tok/s | FP8 sibling | delta |
| --- | ---: | ---: | ---: |
| structured JSON (40-object array) | 55.4 | 62.9 | −12% |
| json (short varied) | 45.0 | 42.4 | +6% |
| math (step-by-step) | 44.1 | 43.3 | ~even |
| code | 32.2 | 33.9 | −5% |
| comms (email) | 29.2 | 27.5 | +6% |
| narrative prose | 19.8 | 18.2 | +9% |
| **mean** | **37.6** | **38.0** | **−1%** |

**vs the FP8 sibling:**

| deploy | KV | max ctx | KV pool | best structured-JSON tok/s | mean tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| [FP8 sibling](https://github.com/tonyd2wild/MiMo-V2.5-DFlash-FP8-KV-2x-DGX-Spark), go-live 2026-07-05 | fp8 | 500K | 980,748 tokens | 62.9 | 38.0 |
| this repo, 500K shape 2026-07-05 | nvfp4 (4-bit) | 500K | 3,167,247 tokens (3.2x) | 55.4 | 37.6 |
| **this repo, 1M shipping config 2026-07-05** | **nvfp4 (4-bit)** | **1M** | **3,427,495 tokens (3.5x)** | 53.0 | **37.8** |

**Honest framing:** DFlash speedup is workload-shaped — quote the shipping (1M) config as **19.4–53.0 tok/s single-stream depending on workload, mean 37.8, up to 68.8 aggregate at 6 streams**, not the peak. The mean holds within 1% of FP8; only the structured-JSON peak pays a ~12% dequant tax. **The trade: FP8 = peak speed (63), NVFP4 = 3.2x the context at the same average speed.**

## Configuration

The three shape-defining env overrides are in [Launch](#launch). The tuning knobs that get the speed (carried over from the FP8 sibling):

1. **Disable thinking for throughput-critical serving** — send `"chat_template_kwargs": {"enable_thinking": false}`. Reasoning prose drafts at ~2 accept length and drags every workload down.
2. **No repetition penalty (1.0).** A penalty pushes the verifier away from correctly-repetitive drafts exactly on the structured output where DFlash shines (57.3 → 63.8 tok/s on structured JSON in the FP8 lane).
3. **`NCCL_PROTO=LL` + `NCCL_MAX_NCHANNELS=2`** on the cross-node TP2 link (danielgbates' insight) — targets the small-message all-reduce latency that dominates the ~100ms DFlash step. `launch-dflash-fp8ckpt.sh` exports both itself (vLLM's ray_env copy propagates `NCCL_*` to the Ray workers).

Serving shape (set in the launch command): `MAX_MODEL_LEN=1000000`, `MAX_NUM_SEQS=6`, `MAX_NUM_BATCHED_TOKENS=4096`, `BLOCK_SIZE=16`, `GPU_MEMORY_UTILIZATION=0.83`, `num_speculative_tokens=7`, async scheduling on (`NO_ASYNC_SCHED=0`), `--enforce-eager` (CUDA graphs are neutral-to-negative — compile disables the custom kernels, including the WMMA decode path). `recipe/env.sh` carries the full validated environment (NVFP4 GEMM/MoE backends + WMMA decode, Ray object-store cap + `RAY_TMPDIR=/dev/shm/ray`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, RoCE/NCCL base).

## Troubleshooting

Serving Xiaomi's DFlash drafter against the NVFP4 target on GB10 required four engine fixes on top of the base image, plus three NVFP4-KV-specific patches and one mod unlock. All are in [`patches/`](patches/) as idempotent apply-in-container scripts (each has a docstring explaining the mechanics). The full live-hit gotcha list is in [`DEFAULT-CONFIG.md`](DEFAULT-CONFIG.md).

### The four base problems

1. **GB10 is FA2-only → no attention sinks in flash-attn.** The DFlash drafter needs non-causal block drafting + attention sinks + SWA. flash-attn on GB10 (FA2) has no sink support, so the drafter must run on the **Triton backend** — which lacked non-causal support. [`patch_triton_noncausal.py`](patches/patch_triton_noncausal.py) ports upstream-main's `USE_CAUSAL` semantics into the image's triton unified attention stack, and [`patch_nc_fix.py`](patches/patch_nc_fix.py) fixes two correctness bugs in that port (non-causal keys must be bounded by `seq_len` — tile-overhang junk was polluting the softmax — and non-causal SWA must skip the first-query V-zeroing). Before the nc fix: acceptance 2.1. A kernel probe validated all 8 causal/sink/SWA combinations.
2. **The hybrid KV cache manager must stay ENABLED.** The drafter runs SWA-1024 while the target mixes attention window types; they need separate KV groups. `--disable-hybrid-kv-cache-manager` breaks the deploy — don't pass it. (In this variant the split works even harder: it is what lets the target run 4-bit KV while the drafter stays bf16.)
3. **DiffKV target pages vs power-of-2 drafter pages.** MiMo-V2.5 target layers use DiffKV geometry (K=192 + V=128 → per-token bytes carry a factor of 5) while the drafter's pages are standard powers of two. Neither divides the other, so vLLM's stock "grow smaller pages to the max" unification fails. [`patch_kv_page_lcm.py`](patches/patch_kv_page_lcm.py) scales every group's block size to the **LCM of all page sizes** so every ratio is an integer.
4. **Aux-hidden-layer off-by-one (+1 semantics) — the big one.** DFlash's `target_layer_ids` mean "hidden state AFTER layer i"; vLLM's aux-hidden-state mixin indexes "state ENTERING layer k". The image's runner passed the ids through raw, so all 5 drafter feature taps were one layer early. [`patch_aux_layer_off_by_one.py`](patches/patch_aux_layer_off_by_one.py) backports upstream's +1 conversion. **This fix alone took mean acceptance 2.14 → 3.78 and 22.6 → 35.2 tok/s.**

### NVFP4-KV-specific extras (this variant, required)

- [`patch_diffkv_noncausal.py`](patches/patch_diffkv_noncausal.py) — threads `USE_CAUSAL` through the diffkv triton kernel so the global non-causal selector gate is satisfied when DFlash is active (target layers still run causal at runtime).
- [`patch_draft_cache_auto.py`](patches/patch_draft_cache_auto.py) — pins the DFlash drafter's KV cache dtype to `auto`: the drafter's stock-triton backend cannot consume the diffkv NVFP4 KV layout, and its SWA-bounded KV is tiny anyway.
- [`patch_spec_dtype_guard.py`](patches/patch_spec_dtype_guard.py) — scopes the global nvfp4 KV spec override to **DiffKV-backed layers only**. Without this the override clobbers the drafter's layers and the engine dies on a spec-vs-view size mismatch at KV init (92160 vs 163840 bytes/block).
- **`EXPERIMENTAL_ALLOW_DIFFKV_QUANT_KV`** — the `fix-mimo-v2-vllm` mod (applied by `recipe/apply-mods.sh`) is what unblocks quantized KV on the `triton_attn_diffkv` backend in the first place.

### Live-hit gotchas

- **Skipping `patch_spec_dtype_guard.py` on this lane crashes at KV init** with a spec-vs-view size mismatch (92160 vs 163840 bytes/block, LCM(10240,18432)=92160 forensics) — the global nvfp4 override hit the drafter's layers.
- **Reddie fabric IP vanishes after teardown.** If `192.168.192.2` is missing: `sudo ip addr add 192.168.192.2/24 dev enp1s0f0np0`. The netplan is fixed (moved to `enp1s0f0np0`) and applies on next boot — that fix also explains the historical "Reddie GID 5" quirk.
- **Ray object store must be capped at 1 GiB on every node** (`run-head.sh` / `run-worker.sh` do this) — uncapped Ray steals unified memory and OOMs the load.
- **Spec metrics under async scheduling:** with async on, the prometheus drafts counter under-counts, inflating client-computed accept_len — use the engine-log `SpecDecoding metrics` lines for acceptance truth.

## Credits & links

This recipe stands on prior public work:

- **Xiaomi MiMo team** — the MiMo-V2.5 model and the official DFlash drafter ([XiaomiMiMo/MiMo-V2.5-DFlash](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-DFlash)).
- **mitomtuna** — the drafter-matched NVFP4/MXFP8 quant of the 0703 target refresh ([mitomtuna/MiMo-V2.5-0703-NVFP4](https://huggingface.co/mitomtuna/MiMo-V2.5-0703-NVFP4)).
- **lukealonso** — the NVFP4 quantization of the target ([lukealonso/MiMo-V2.5-NVFP4](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4)), which is what makes a 2-Spark TP=2 deploy possible at all.
- **The vLLM project** — the upstream DFlash implementation this repo backports from main (PRs [#45200](https://github.com/vllm-project/vllm/pull/45200), [#45181](https://github.com/vllm-project/vllm/pull/45181), [#46104](https://github.com/vllm-project/vllm/pull/46104) lineage). `qwen3_dflash.py` and `dflash_proposer.py` are vLLM-derived and keep their Apache-2.0 SPDX headers.
- **danielgbates** — independent MiMo-V2.5 + DFlash 2x-Spark recipe on the NVIDIA forums, and in particular the **`NCCL_PROTO=LL` + `NCCL_MAX_NCHANNELS=2`** small-message latency insight adopted here ([forums.developer.nvidia.com/t/375607](https://forums.developer.nvidia.com/t/375607)). His recipe runs bf16 KV — the 4-bit-KV lane in this repo is the part he did not touch.
- **z-lab** — the Qwen3.6 DFlash precedent that proved the DFlash engine path works on these Sparks.
- **renek** — the 60+ tok/s existence proof that set the target.
- Upstream vLLM, Triton, FlashInfer, NVIDIA Blackwell/CUDA/NCCL tooling.

**Our contribution:** the GB10/FA2 triton non-causal port and its correctness fixes, the LCM page unification, the aux-layer off-by-one diagnosis and backport, and the quantized-KV coexistence patches that make NVFP4 weights + NVFP4 KV + DFlash run together — which we believe is a first.

## License

Repo scripts and docs: [Apache-2.0](LICENSE). The vLLM-derived files (`patches/qwen3_dflash.py`, `patches/dflash_proposer.py`) retain their upstream Apache-2.0 SPDX headers. Base images, model weights, and NVIDIA tooling are separate upstream artifacts with their own licenses and terms.
