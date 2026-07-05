"""Backport upstream fix: DFlash target_layer_ids use 'hidden state AFTER
layer i' semantics; vLLM's aux-hidden-state mixin indexes 'state ENTERING
layer k' (k=0 is embeddings). Upstream main adds +1 when converting; the
image's runner passed the ids through raw, so all 5 drafter feature taps
were one layer early."""
from pathlib import Path
import py_compile

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py")
src = p.read_text()
old = """            dflash_config = getattr(hf_config, "dflash_config", None)
            if dflash_config and isinstance(dflash_config, dict):
                layer_ids = dflash_config.get("target_layer_ids")"""
new = """            dflash_config = getattr(hf_config, "dflash_config", None)
            if dflash_config and isinstance(dflash_config, dict):
                # Add 1 to convert DFlash's aux layer id semantics
                layer_ids = [
                    i + 1 for i in (dflash_config.get("target_layer_ids") or [])
                ]"""
if new in src:
    print("already patched")
else:
    assert old in src, "anchor not found"
    p.write_text(src.replace(old, new, 1))
    py_compile.compile(str(p), doraise=True)
    print("patch_aux_layer_off_by_one: APPLIED + COMPILED")
