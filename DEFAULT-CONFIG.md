# DEFAULT CONFIG — MiMo-V2.5 + DFlash + NVFP4 4-bit KV (long-context lane), 2x DGX Spark

This is the **canonical launch flow** for the NVFP4-KV variant — identical stack to the
[FP8 sibling](https://github.com/tonyd2wild/MiMo-V2.5-DFlash-FP8-KV-2x-DGX-Spark)
(same image, mods, patches, Ray bring-up, same launcher script), differing **only in
three env overrides** at launch. Order matters: **containers on both nodes → mods →
engine patches → Ray head → Ray worker → serve from the head.**

**Verified (shipping config, 2026-07-05):** TP=2 on Bluey (rank0/head) + Reddie
(rank1/worker), served `:8000`. NVFP4 4-bit target KV @ **1M max context** — **KV pool
3,427,495 tokens (3.43x for full-1M streams)**. Single-stream mean **37.8 tok/s**
(53.0 structured JSON → 19.4 narrative), identical to the 500K shape and within 1% of
the FP8 sibling; **68.8 tok/s aggregate at 6 concurrent streams**. Tool calling
verified live (mimo parser). Alternate 500K shape: pool 3,167,247 (6.33x 500K
streams). See `benchmarks/RESULTS.md`.

---

## Model + image

- **Target:** `lukealonso/MiMo-V2.5-NVFP4` (171G, pinned snapshot `a147dd04...`), in the
  HF cache on **both** nodes.
- **Drafter:** `XiaomiMiMo/MiMo-V2.5-DFlash`, **`dflash/` subdir only** (2.8G, snapshot
  `1f58446181abcaa01030fdbde835fbd38ae9a2b1`), on **both** nodes:
  ```bash
  hf download XiaomiMiMo/MiMo-V2.5-DFlash --include "dflash/*"
  ```
  Do NOT download the full repo — it carries a 311GB fp8 copy of the target you don't need.
- **Base image:** `vllm-node-mimo-v25-nvfp4:latest` (same build on both Sparks; has the
  DFlash engine wiring + diffkv KV backends baked). The
  `ghcr.io/...nvfp4kv:20260620` image has **no** dflash proposer — don't use it for this.
- **Overlay image:** `vllm-node-mimo-v25-nvfp4:dflash-mimo-20260704`, built on both
  nodes from `patches/Dockerfile` = base + vllm-main `qwen3_dflash.py` + guarded
  `dflash_proposer.py` + `patch_mimo_v2_eagle3.py`.
- vLLM build inside: `0.21.1rc1.dev85+gd87ee1893` (DEV build, CUDA 13.2, torch 2.11.0).
  Stock `pip install vllm` will not work. **Ray is required** (mp executor is single-host).

## Node / network (direct-cabled RoCE, 192.168.192.0/24)

| role | node | user | fabric IP | tailnet | GID | HF cache |
| --- | --- | --- | --- | --- | ---: | --- |
| head (rank0) | Bluey | tonyspark1 | 192.168.192.1 | 100.92.77.51 | 3 | ~/.cache/huggingface |
| worker (rank1) | Reddie | tonyspark2 | 192.168.192.2 | 100.113.138.96 | 5 (see gotcha) | ~/.cache/huggingface |

Container: `--network host --ipc host --shm-size 16g --gpus all`,
`--device /dev/infiniband:/dev/infiniband`, memlock unlimited (see `recipe/run-container.sh`).

## Step 1 — containers + mods + patches (BOTH nodes)

```bash
cd ~/mimo-v25-tp2-nvfp4/recipe            # or this repo's recipe/ dir on the node

# 1. container from the DFlash overlay image
IMAGE=vllm-node-mimo-v25-nvfp4:dflash-mimo-20260704 \
CONTAINER=vllm_mimo_dflash \
bash run-container.sh

# 2. the 6 base mods (fix-mimo-v2-vllm registers MimoV2Config — without it:
#    "model type mimo_v2 not recognized" — AND unlocks quantized KV on the
#    triton_attn_diffkv backend via EXPERIMENTAL_ALLOW_DIFFKV_QUANT_KV, which
#    this lane depends on). Mods do NOT clobber the dflash overlay files
#    (verified). A non-fatal soundfile error in the omni-audio step is
#    expected for text-only serving.
bash apply-mods.sh vllm_mimo_dflash

# 3. the engine patches (idempotent; each prints APPLIED or "already patched").
#    For THIS lane the last three are REQUIRED, not optional: they are what let
#    the nvfp4 target KV coexist with the drafter's bf16 KV.
for p in patch_triton_noncausal patch_nc_fix patch_kv_page_lcm \
         patch_aux_layer_off_by_one patch_diffkv_noncausal \
         patch_draft_cache_auto patch_spec_dtype_guard; do
  docker cp patches/$p.py vllm_mimo_dflash:/tmp/$p.py
  docker exec vllm_mimo_dflash python3 /tmp/$p.py
done
```

## Step 2 — Ray bring-up (head first, then worker)

Head (Bluey), inside the container:

```bash
source /workspace/recipe/env.sh
export HEAD_ROCE_IP=192.168.192.1
bash /workspace/recipe/run-head.sh
```

Worker (Reddie), inside its container:

```bash
source /workspace/recipe/env.sh
export HEAD_ROCE_IP=192.168.192.1
export WORKER_ROCE_IP=192.168.192.2
export NCCL_IB_GID_INDEX=5        # Reddie's GID — see gotchas
bash /workspace/recipe/run-worker.sh
```

On the head, wait for both GPUs:

```bash
until ray status 2>/dev/null | grep -qE '2\.0/2\.0 GPU|2\.0 GPU'; do sleep 2; done
```

## Step 3 — env (both nodes, on top of env.sh)

`recipe/env.sh` carries the full validated environment (NVFP4 GEMM/MoE backends + WMMA
decode, Ray object-store cap + `RAY_TMPDIR=/dev/shm/ray`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, RoCE/NCCL base). The NCCL block
that matters for DFlash step rate:

```bash
NCCL_NET=IB  NCCL_IB_DISABLE=0  NCCL_IB_HCA=rocep1s0f0  NCCL_SOCKET_IFNAME=enp1s0f0np0
NCCL_IB_GID_INDEX=3            # 5 on Reddie until its netplan fix reboots in
NCCL_CROSS_NIC=1  NCCL_CUMEM_ENABLE=0  NCCL_NVLS_ENABLE=0
NCCL_NET_GDR_LEVEL=LOC
# --- small-message latency tuning (danielgbates) — the DFlash step-rate lever:
NCCL_PROTO=LL
NCCL_MAX_NCHANNELS=2
```

`launch-dflash-fp8ckpt.sh` exports `NCCL_PROTO=LL` and `NCCL_MAX_NCHANNELS=2` itself
(vLLM's ray_env copy propagates `NCCL_*` to the Ray workers).

## Step 4 — the NVFP4-KV launch command (head node)

Same launcher as the FP8 sibling —
[`recipe/launch-dflash-fp8ckpt.sh`](recipe/launch-dflash-fp8ckpt.sh) — with **three
env overrides that define this variant**:

| override | value | effect |
| --- | --- | --- |
| `KV_CACHE_DTYPE` | `nvfp4` | target KV cache in 4-bit NVFP4 (packed uint8) |
| `ATTENTION_BACKEND` | `triton_attn_diffkv` | target layers on the DiffKV backend (WMMA decode) |
| `SPEC_ATTENTION_BACKEND` | `triton_attn` | drafter on stock triton with bf16 KV in its own hybrid cache group |

The target/drafter backend split is load-bearing: the drafter cannot consume the
diffkv NVFP4 layout, and giving it its own group is also the fix for "the drafter
eats the KV pool" (see `patch_spec_dtype_guard.py`).

```bash
export MODEL_PATH=/root/.cache/huggingface/hub/models--lukealonso--MiMo-V2.5-NVFP4/snapshots/a147dd04d6cf861e43b2d783dcde23b53ab7ee68
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

`launch-dflash-fp8ckpt.sh` expands this to (byte-exact, NVFP4-KV lane):

```
vllm serve $MODEL_PATH \
  --served-model-name MiMo-V2.5-NVFP4-DFlash-NVFP4KV \
  --trust-remote-code --dtype auto \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --load-format safetensors \
  --hf-overrides '{"architectures":["MiMoV2ForCausalLM"]}' \
  --attention-backend triton_attn_diffkv \
  --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization 0.83 \
  --max-model-len 1000000 --max-num-batched-tokens 4096 --max-num-seqs 6 \
  --block-size 16 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  --reasoning-parser mimo \
  --generation-config vllm \
  --override-generation-config '{"temperature":0,"top_p":0.95}' \
  --speculative-config '{"model":"'$DFLASH_MODEL_PATH'","method":"dflash","num_speculative_tokens":7,"attention_backend":"triton_attn"}' \
  --enforce-eager \
  --host 0.0.0.0 --port 8000
```

(with `NCCL_PROTO=LL` and `NCCL_MAX_NCHANNELS=2` exported by the script.)

Boot evidence (shipping 1M config, verified 2026-07-05):

```text
GPU KV cache size: 3,427,495 tokens
Maximum concurrency for 1,000,000 tokens per request: 3.43x
```

Alternate 500K shape (also verified): pool 3,167,247 tokens / 6.33x — 3.2x the FP8
sibling's 980,748 at the identical 500K/GMU-0.83 shape.

Key choices (all A/B verified on this stack):

- **triton_attn_diffkv target / triton_attn drafter** — the split described above.
  flash_attn is out entirely: GB10 is FA2-only, no sink support (README problem 1).
- **Hybrid KV manager stays ENABLED** — never pass `--disable-hybrid-kv-cache-manager`.
  In this lane it is what isolates the drafter's bf16 group from the nvfp4 target.
- **Async scheduling ON** (`NO_ASYNC_SCHED=0`), matching the FP8 go-live config.
  Caveat: with async on, the prometheus spec_decode drafts counter under-counts —
  see gotchas below.
- **No repetition penalty** (generation config is `temperature 0, top_p 0.95` only) —
  a penalty pushes the verifier away from correctly-repetitive drafts exactly on the
  structured output where DFlash shines (57.3 → 63.8 tok/s on structured JSON, FP8 lane).
- **Thinking off** for throughput-critical serving: send
  `"chat_template_kwargs": {"enable_thinking": false}` — reasoning prose drafts at
  ~2 accept and drags every workload.
- **`--enforce-eager`** — CUDA graphs are neutral-to-negative (compile disables the
  custom kernels, including the WMMA decode path).

Smoke test:

```bash
curl http://192.168.192.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-V2.5-NVFP4-DFlash-NVFP4KV","messages":[{"role":"user","content":"Reply exactly: OK DFLASH NVFP4KV LIVE"}],"max_tokens":16,"temperature":0}'
```

Then bench honestly (args: base-url, model, max-tokens, thinking on|off — defaults to
off, rp 1.0, includes the structured_json category):

```bash
python3 benchmarks/dflash_bench.py http://192.168.192.1:8000 MiMo-V2.5-NVFP4-DFlash-NVFP4KV 512
```

## Gotchas (all hit live)

- **Reddie fabric IP vanishes after teardown.** Historically `192.168.192.2` was bound
  to the wrong port (`enp1s0f1np1`) in `42-cx7-switch-codex.yaml`; if the address is
  missing: `sudo ip addr add 192.168.192.2/24 dev enp1s0f0np0`. The netplan is fixed
  (moved to `enp1s0f0np0`) and applies on next boot — that fix also explains the
  historical "Reddie GID 5" quirk.
- **Bluey weight-load UVM thrash** when free RAM < 2G: run a cache-dropper loop during
  load (`sync; echo 3 > /proc/sys/vm/drop_caches` whenever free < 4G).
- **Spec metrics:** use the `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total`
  counters. The `_created` variants are timestamps — do not sum them. **And with async
  scheduling ON, the prometheus drafts counter under-counts**, inflating
  client-computed accept_len (e.g. an impossible 11.77); use the engine-log
  `SpecDecoding metrics` lines for acceptance truth.
- **Skipping `patch_spec_dtype_guard.py` on this lane crashes at KV init** with a
  spec-vs-view size mismatch (92160 vs 163840 bytes/block, LCM(10240,18432)=92160
  forensics) — the global nvfp4 override hit the drafter's layers.
- **Ray object store must be capped at 1 GiB on every node** (`run-head.sh`/`run-worker.sh`
  do this) — uncapped Ray steals unified memory and OOMs the load.
- `attention_value_scale` / `num_anchors` in the drafter config are unused by Xiaomi's
  own dflash.py — not a gap in this recipe.

> **Tool calling:** the launcher enables `--enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo` (the `mimo` parser is registered by the `fix-mimo-v2-vllm` mod). Without these flags, OpenAI-style `tools`/`tool_choice:"auto"` requests return 400.
