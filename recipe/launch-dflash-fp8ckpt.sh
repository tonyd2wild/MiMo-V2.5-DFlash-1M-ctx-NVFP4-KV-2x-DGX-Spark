#!/usr/bin/env bash
# MiMo-V2.5 NVFP4 weights + DFlash + FP8 KV — "checkpoint" config.
# danielgbates-recipe shape (131K / seqs4 / batched4096 / GMU .83 / prefix+chunked)
# + NCCL LL tuning. Run INSIDE the container on the HEAD node.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
: "${DFLASH_MODEL_PATH:?set DFLASH_MODEL_PATH to the dflash/ snapshot dir}"
: "${SERVED_MODEL_NAME:=MiMo-V2.5-NVFP4-DFlash-FP8KV}"
: "${MAX_MODEL_LEN:=131072}"
: "${MAX_NUM_BATCHED_TOKENS:=4096}"
: "${MAX_NUM_SEQS:=4}"
: "${GPU_MEMORY_UTILIZATION:=0.83}"
: "${DFLASH_SPEC_TOKENS:=7}"
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${BLOCK_SIZE:=16}"
: "${KV_CACHE_DTYPE:=fp8}"
: "${ATTENTION_BACKEND:=triton_attn}"
[ "${NO_ASYNC_SCHED:-1}" = "1" ] && NO_ASYNC_SCHED_FLAG="--no-async-scheduling"
[ "${ENFORCE_EAGER:-1}" = "1" ] && EAGER_FLAG="--enforce-eager"
: "${HEAD_ROCE_IP:?set HEAD_ROCE_IP}"
export VLLM_HOST_IP="${HEAD_ROCE_IP}"

# Cross-node decode-latency levers (danielgbates / NVIDIA forum 375607).
# NCCL_* prefixes are copied to Ray workers by vLLM's ray_env copy.
export NCCL_PROTO="${NCCL_PROTO:-LL}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-2}"

if [ -n "${SPEC_ATTENTION_BACKEND:-}" ]; then
  SPECULATIVE_CONFIG="{\"model\":\"${DFLASH_MODEL_PATH}\",\"method\":\"dflash\",\"num_speculative_tokens\":${DFLASH_SPEC_TOKENS},\"attention_backend\":\"${SPEC_ATTENTION_BACKEND}\"}"
else
  SPECULATIVE_CONFIG="{\"model\":\"${DFLASH_MODEL_PATH}\",\"method\":\"dflash\",\"num_speculative_tokens\":${DFLASH_SPEC_TOKENS}}"
fi
GENERATION_CONFIG='{"temperature":0,"top_p":0.95}'
KV_CACHE_FLAG=()
[ "${KV_CACHE_DTYPE}" != "auto" ] && KV_CACHE_FLAG=(--kv-cache-dtype "${KV_CACHE_DTYPE}")

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --dtype auto \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --distributed-executor-backend ray \
  --load-format safetensors \
  --hf-overrides '{"architectures":["MiMoV2ForCausalLM"]}' \
  --attention-backend "${ATTENTION_BACKEND}" \
  "${KV_CACHE_FLAG[@]}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --block-size "${BLOCK_SIZE}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  --reasoning-parser mimo \
  ${NO_ASYNC_SCHED_FLAG:-} \
  --generation-config vllm \
  --override-generation-config "${GENERATION_CONFIG}" \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  ${EAGER_FLAG:-} \
  --host 0.0.0.0 \
  --port 8000
