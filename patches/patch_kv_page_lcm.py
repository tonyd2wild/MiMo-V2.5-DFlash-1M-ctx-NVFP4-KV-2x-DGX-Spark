"""LCM-based KV page-size unification.

MiMo-V2.5 target layers use DiffKV geometry (K=192 + V=128, 4 kv heads ->
per-token bytes carry a factor of 5) while the DFlash drafter layers are
standard (256 x 4 heads -> power of two). Neither page divides the other,
so the stock "grow smaller pages to the max" unification fails. Instead,
scale every group's block size to reach the LCM of all page sizes --
every ratio is then an integer and no padding/strided views are needed.
"""
import math
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py")
src = p.read_text()

old = """    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec"""

new = """    max_page_size = max(page_sizes)
    target_page_size = max_page_size
    if any(max_page_size % p != 0 for p in page_sizes):
        # Non-divisible mix (e.g. DiffKV 320B/token target vs 512B/token
        # drafter = 5:8): unify to the LCM so every ratio is an integer.
        import math as _math
        target_page_size = _math.lcm(*page_sizes)
        if target_page_size > 64 * max_page_size:
            raise NotImplementedError(
                f"KV page sizes {sorted(page_sizes)} cannot be unified: "
                f"LCM {target_page_size} exceeds 64x the max page size."
            )
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == target_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            assert target_page_size % layer_page_size == 0
            ratio = target_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == target_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec"""

if new in src:
    print("already patched")
else:
    assert old in src, "unify_kv_cache_spec_page_size anchor not found"
    p.write_text(src.replace(old, new, 1))
    import py_compile
    py_compile.compile(str(p), doraise=True)
    print("patch_kv_page_lcm: APPLIED + COMPILED")
