"""Scope the nvfp4-kv-diffkv mod's global spec override to DiffKV-backed
layers only. The mod forces every layer's KV spec to nvfp4/uint8 when the
global cache dtype is nvfp4 — correct for the all-diffkv MTP stack, but it
clobbers the DFlash drafter's layers, which run on the stock triton backend
(auto KV) and then crash on a spec-vs-view size mismatch at KV init."""
from pathlib import Path
import py_compile

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py")
s = p.read_text()
old = """        if getattr(vllm_config.cache_config, 'cache_dtype', None) == 'nvfp4':
            quant_mode = get_kv_quant_mode('nvfp4')  # nvfp4-kv-diffkv: live dtype not stale self
            import torch as _t_nv; self.kv_cache_torch_dtype = _t_nv.uint8  # nvfp4 packed uint8 cache"""
new = """        if (
            getattr(vllm_config.cache_config, 'cache_dtype', None) == 'nvfp4'
            and 'DiffKV' in getattr(self.attn_backend, '__name__', '')
        ):
            # nvfp4-kv-diffkv: live dtype not stale self — but ONLY for
            # DiffKV-backed layers. Non-diffkv layers (e.g. the DFlash
            # drafter on stock triton) keep their own kv_cache_dtype.
            quant_mode = get_kv_quant_mode('nvfp4')
            import torch as _t_nv; self.kv_cache_torch_dtype = _t_nv.uint8  # nvfp4 packed uint8 cache"""
if new in s:
    print("already patched")
else:
    assert old in s, "spec override anchor not found"
    p.write_text(s.replace(old, new, 1))
    py_compile.compile(str(p), doraise=True)
    print("patch_spec_dtype_guard: APPLIED + COMPILED")
