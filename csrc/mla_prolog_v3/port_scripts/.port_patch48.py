"""Auto-detect weight_quant_mode in the shim.

SGLang's forward_mlaprolog hardcodes `weight_quant_mode=1` (uq_qr quantised). For the
GLM-4.7-Flash MXFP8 checkpoint the serving path dequantises the three projections to
bf16 (mode 0), so passing mode 1 together with bf16 weights makes the tiling reject the
call. Derive the mode from the actual dtypes and drop the scale arguments that the
derived mode does not use.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''_libs = []


def _forward(*args, **kwargs):
    return torch.ops.npu.mla_prolog_v3(*args, **kwargs)
'''
NEW = '''_libs = []

# scale arguments that only make sense for a quantised weight mode
_WEIGHT_SCALE_ARGS = (
    "dequant_scale_x",
    "dequant_scale_w_dq",
    "dequant_scale_w_uq_qr",
    "dequant_scale_w_dkv_kr",
)


def infer_weight_quant_mode(kwargs):
    """Derive ``weight_quant_mode`` from the projection dtypes, or None if unknown.

    * bf16 (or other float16/32) weights  -> 0 (no quant)
    * float8_e4m3fn weights with an e8m0 scale -> 3 (MXFP8 full quant)
    * int8 weights -> None (caller's 1/2 decides, we cannot tell 1 from 2)
    """
    w = kwargs.get("weight_uq_qr")
    if w is None:
        return None
    if w.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return 0
    if w.dtype == torch.float8_e4m3fn and kwargs.get("dequant_scale_w_uq_qr") is not None:
        return 3
    return None


def _forward(*args, **kwargs):
    mode = infer_weight_quant_mode(kwargs)
    if mode is not None and kwargs.get("weight_quant_mode") != mode:
        kwargs["weight_quant_mode"] = mode
        if mode == 0:
            for name in _WEIGHT_SCALE_ARGS:
                kwargs.pop(name, None)
    return torch.ops.npu.mla_prolog_v3(*args, **kwargs)
'''

assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("auto weight_quant_mode added")
