"""Port non-causal (bidirectional) attention support into the image's
triton unified attention stack, mirroring upstream vllm-main semantics
(USE_CAUSAL constexpr). Needed by the MiMo-V2.5 DFlash drafter
(non-causal block drafting + attention sinks + SWA) on GB10 where
flash-attn is FA2-only (no sink support).
"""
from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm")

def edit(path, old, new, count=1):
    src = path.read_text()
    if new in src and old not in src:
        print(f"  already patched: {path.name}: {new[:40]!r}...")
        return
    assert old in src, f"ANCHOR NOT FOUND in {path}: {old[:80]!r}"
    path.write_text(src.replace(old, new, count))

# ---------------- helpers ----------------
H = P / "v1/attention/ops/triton_attention_helpers.py"

edit(H, """    USE_MM_PREFIX: tl.constexpr,
    IS_3D: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):""", """    USE_MM_PREFIX: tl.constexpr,
    IS_3D: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    USE_CAUSAL: tl.constexpr = True,
):""")

edit(H, """    if USE_MM_PREFIX:
        # image bidirectional attention ranges require a full range
        # including q_block padding to make sure doc mask is correct
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)""",
"""    if USE_MM_PREFIX or (not USE_CAUSAL):
        # Bidirectional ranges (mm_prefix) and non-causal attention can
        # reach past the causal prefix: cover the full sequence.
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)""")

edit(H, """        last_allowed_key = context_len + qpos_hi
""", """        if USE_CAUSAL:
            last_allowed_key = context_len + qpos_hi
        else:
            # Non-causal SWA: keys may be AHEAD of the query within the window.
            last_allowed_key = context_len + qpos_hi + SLIDING_WINDOW - 1
""")

edit(H, """    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):""", """    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    USE_CAUSAL: tl.constexpr = True,
):""")

edit(H, """    # Compute attention mask: causal by default (key <= query)
    seq_mask = seq_offset[None, :] <= query_abs_pos""",
"""    # Compute attention mask: causal by default (key <= query)
    if USE_CAUSAL:
        seq_mask = seq_offset[None, :] <= query_abs_pos
    else:
        # Non-causal: all keys visible (tile_mask already bounds seq_len).
        seq_mask = (seq_offset[None, :] >= 0) & (query_abs_pos >= 0)""")

edit(H, """    elif SLIDING_WINDOW > 0:
        seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)""",
"""    elif SLIDING_WINDOW > 0:
        if USE_CAUSAL:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
        else:
            seq_mask = (
                seq_mask
                & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
                & ((query_abs_pos - seq_offset) > -SLIDING_WINDOW)
            )""")

# ---------------- unified attention kernel ----------------
K = P / "v1/attention/ops/triton_unified_attention.py"
src = K.read_text()
if "USE_CAUSAL" not in src:
    # kernel signature: insert before the closing "):" of kernel_unified_attention
    i = src.index("def kernel_unified_attention(")
    j = src.index("\n):", i)
    src = src[:j] + "\n    USE_CAUSAL: tl.constexpr = True,  # bool" + src[j:]
    K.write_text(src)

edit(K, """        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
    )""", """        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
        USE_CAUSAL,
    )""")

edit(K, """            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
        )""", """            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
            USE_CAUSAL,
        )""")

edit(K, """        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )""", """        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            sw_dist = context_len + qpos_lo - seq_offset[:, None]
            if USE_CAUSAL:
                sw_keep = sw_dist < SLIDING_WINDOW
            else:
                sw_keep = (sw_dist < SLIDING_WINDOW) & (sw_dist > -SLIDING_WINDOW)
            V = tl.where(sw_keep, V, 0.0)""")

edit(K, """    assert causal, "Only causal attention is supported"
""", """    use_causal = bool(causal)
""")

edit(K, """        SLIDING_WINDOW=(1 + window_size[0]),
""", """        SLIDING_WINDOW=(1 + window_size[0]),
        USE_CAUSAL=use_causal,
""")

# ---------------- triton_attn backend ----------------
B = P / "v1/attention/backends/triton_attn.py"

edit(B, """    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None
""", """    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None
    causal: bool = True
""")

edit(B, """            slot_mapping=slot_mapping,
            use_cascade=use_cascade,""", """            slot_mapping=slot_mapping,
            causal=common_attn_metadata.causal,
            use_cascade=use_cascade,""")

edit(B, """    @classmethod
    def supports_sink(cls) -> bool:
        return True
""", """    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True
""")

edit(B, """            softmax_scale=self.scale,
            causal=True,""", """            softmax_scale=self.scale,
            causal=attn_metadata.causal,""")

# ---------------- diffkv backend: keep declaring non-causal unsupported ----------------
D = P / "v1/attention/backends/triton_attn_diffkv.py"
edit(D, """class TritonAttentionDiffKVBackend(TritonAttentionBackend):
""", """class TritonAttentionDiffKVBackend(TritonAttentionBackend):
    @classmethod
    def supports_non_causal(cls) -> bool:
        # Non-causal not yet ported to the diffkv kernel variant.
        return False

""")

import py_compile
for f in (H, K, B, D):
    py_compile.compile(str(f), doraise=True)
print("patch_triton_noncausal: ALL EDITS APPLIED + COMPILED")
