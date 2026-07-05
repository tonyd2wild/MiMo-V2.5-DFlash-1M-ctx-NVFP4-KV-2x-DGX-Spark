"""Fix two non-causal correctness bugs in the USE_CAUSAL port:

1. Non-causal base mask was all-True, letting tile-overhang keys
   (>= seq_len) into the softmax (causal's key<=query bound had hidden
   them). Bound non-causal keys by seq_len, like upstream main.
2. Non-causal SWA V-zeroing keyed the window off the block's FIRST
   query; for non-causal, later queries legitimately attend beyond that
   bound. Skip V-zeroing when non-causal — the per-element seq_mask is
   exact and masked loads are zero-filled.
"""
from pathlib import Path
import py_compile

P = Path("/usr/local/lib/python3.12/dist-packages/vllm")

def edit(path, old, new):
    src = path.read_text()
    if new in src and old not in src:
        print(f"  already patched: {path.name}")
        return
    assert old in src, f"ANCHOR NOT FOUND in {path}: {old[:70]!r}"
    path.write_text(src.replace(old, new, 1))

H = P / "v1/attention/ops/triton_attention_helpers.py"

# add seq_len param (runtime scalar) at the end of compute_kv_seq_mask
edit(H, """    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    USE_CAUSAL: tl.constexpr = True,
):""", """    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    USE_CAUSAL: tl.constexpr = True,
    seq_len=0,
):""")

edit(H, """    if USE_CAUSAL:
        seq_mask = seq_offset[None, :] <= query_abs_pos
    else:
        # Non-causal: all keys visible (tile_mask already bounds seq_len).
        seq_mask = (seq_offset[None, :] >= 0) & (query_abs_pos >= 0)""",
"""    if USE_CAUSAL:
        seq_mask = seq_offset[None, :] <= query_abs_pos
    else:
        # Non-causal: every VALID key is visible. Must still exclude
        # tile-overhang slots >= seq_len (causal's key<=query bound
        # implicitly hid them; without it they pollute the softmax).
        seq_mask = (seq_offset[None, :] < seq_len) & (query_abs_pos >= 0)""")

K = P / "v1/attention/ops/triton_unified_attention.py"

edit(K, """            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
            USE_CAUSAL,
        )""", """            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
            USE_CAUSAL,
            seq_len,
        )""")

edit(K, """        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            sw_dist = context_len + qpos_lo - seq_offset[:, None]
            if USE_CAUSAL:
                sw_keep = sw_dist < SLIDING_WINDOW
            else:
                sw_keep = (sw_dist < SLIDING_WINDOW) & (sw_dist > -SLIDING_WINDOW)
            V = tl.where(sw_keep, V, 0.0)""",
"""        if SLIDING_WINDOW and USE_CAUSAL:
            # Causal SWA: keys older than the window of the block's first
            # query are invalid for EVERY query in the block, so zeroing V
            # is a safe belt-and-braces cleanup. Non-causal: that bound is
            # NOT valid for later queries (they see further ahead), and the
            # per-element seq_mask is already exact with zero-filled loads,
            # so skip V-zeroing entirely.
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )""")

for f in (H, K):
    py_compile.compile(str(f), doraise=True)
print("patch_nc_fix: APPLIED + COMPILED")
