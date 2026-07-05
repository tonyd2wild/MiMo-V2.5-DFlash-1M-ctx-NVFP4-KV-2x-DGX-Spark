"""Thread USE_CAUSAL through the diffkv triton kernel (mirrors the stock
triton port). Target layers still run causal at runtime; this satisfies the
global non-causal selector gate under DFlash and applies the same
correctness semantics (seq_len bound, SWA handling) if ever used non-causally."""
from pathlib import Path
import py_compile

P = Path("/usr/local/lib/python3.12/dist-packages/vllm")
K = P / "v1/attention/ops/triton_unified_attention_diffkv.py"
B = P / "v1/attention/backends/triton_attn_diffkv.py"

def edit(path, old, new, count=1):
    s = path.read_text()
    if new in s and old not in s:
        print(f"  already: {path.name}")
        return
    assert old in s, f"ANCHOR NOT FOUND in {path.name}: {old[:70]!r}"
    path.write_text(s.replace(old, new, count))

src = K.read_text()
if "USE_CAUSAL" not in src:
    i = src.index("def kernel_unified_attention_diffkv(")
    j = src.index("\n):", i)
    src = src[:j] + "\n    USE_CAUSAL: tl.constexpr = True,  # bool" + src[j:]
    K.write_text(src)

edit(K, """        SLIDING_WINDOW,
        False,  # USE_MM_PREFIX
        IS_3D,
    )""", """        SLIDING_WINDOW,
        False,  # USE_MM_PREFIX
        IS_3D,
        -1,  # CHUNK_LOOKBACK
        -1,  # CHUNK_SIZE
        USE_CAUSAL,
    )""")

edit(K, """            SLIDING_WINDOW,
            False,  # USE_MM_PREFIX
            0,  # MAX_MM_RANGES
        )""", """            SLIDING_WINDOW,
            False,  # USE_MM_PREFIX
            0,  # MAX_MM_RANGES
            -1,  # CHUNK_LOOKBACK
            -1,  # CHUNK_SIZE
            USE_CAUSAL,
            seq_len,
        )""")

edit(K, """        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )""", """        if SLIDING_WINDOW and USE_CAUSAL:
            # See stock triton port: the qpos_lo bound is only valid for
            # causal SWA; non-causal relies on the exact per-element mask.
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )""")

edit(K, """    assert causal, "Only causal attention is supported"
""", """    use_causal = bool(causal)
""")

edit(K, """        SLIDING_WINDOW=sliding_window_val,
""", """        SLIDING_WINDOW=sliding_window_val,
        USE_CAUSAL=use_causal,
""")

# backend: un-pin non-causal, honor metadata causal
edit(B, """    @classmethod
    def supports_non_causal(cls) -> bool:
        # Non-causal not yet ported to the diffkv kernel variant.
        return False
""", """    @classmethod
    def supports_non_causal(cls) -> bool:
        return True
""")
edit(B, """            causal=True,""", """            causal=getattr(attn_metadata, "causal", True),""")

for f in (K, B):
    py_compile.compile(str(f), doraise=True)
print("patch_diffkv_noncausal: APPLIED + COMPILED")
