#!/usr/bin/env bash
# MiMo-V2.5 (NVFP4 weights) TP=2 + Xiaomi DFlash drafter — run INSIDE the container on the HEAD node.
# Phase 1 smoke: auto KV + flash_attn. Phase 2: KV_CACHE_DTYPE=nvfp4 ATTENTION_BACKEND=triton_attn_diffkv.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
: "${DFLASH_MODEL_PATH:?set DFLASH_MODEL_PATH to the dflash/ snapshot dir}"
: "${SERVED_MODEL_NAME:=MiMo-V2.5-NVFP4-DFlash}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_BATCHED_TOKENS:=8192}"
: "${MAX_NUM_SEQS:=1}"
: "${GPU_MEMORY_UTILIZATION:=0.84}"
: "${DFLASH_SPEC_TOKENS:=7}"
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${BLOCK_SIZE:=16}"
: "${KV_CACHE_DTYPE:=auto}"
: "${ATTENTION_BACKEND:=flash_attn}"
[ "${DISABLE_HYBRID_KV:-0}" = "1" ] && DISABLE_HYBRID_KV_FLAG="--disable-hybrid-kv-cache-manager"
[ "${NO_ASYNC_SCHED:-1}" = "1" ] && NO_ASYNC_SCHED_FLAG="--no-async-scheduling"
[ "${ENFORCE_EAGER:-1}" = "1" ] && EAGER_FLAG="--enforce-eager"
: "${HEAD_ROCE_IP:?set HEAD_ROCE_IP}"
export VLLM_HOST_IP="${HEAD_ROCE_IP}"

SPECULATIVE_CONFIG="{\"model\":\"${DFLASH_MODEL_PATH}\",\"method\":\"dflash\",\"num_speculative_tokens\":${DFLASH_SPEC_TOKENS}}"
GENERATION_CONFIG='{"temperature":0,"top_p":0.95,"repetition_penalty":1.08}'
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
  ${DISABLE_HYBRID_KV_FLAG:-} \
  ${NO_ASYNC_SCHED_FLAG:-} \
  --generation-config vllm \
  --override-generation-config "${GENERATION_CONFIG}" \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  ${EAGER_FLAG:-} \
  --host 0.0.0.0 \
  --port 8000
