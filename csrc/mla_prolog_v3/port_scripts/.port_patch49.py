"""Fix the shim: torch passes tensors positionally to library impls."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''def infer_weight_quant_mode(kwargs):
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

NEW = '''def infer_weight_quant_mode(weight_uq_qr, weight_scale=None):
    """Derive the weight quant mode from the projection dtypes, or None if unknown.

    * float16/32/bfloat16 weights                  -> 0 (no quant)
    * float8_e4m3fn weights with an e8m0 scale     -> 3 (MXFP8 full quant)
    * int8 weights                                 -> None (the caller's 1/2 decides)
    """
    if weight_uq_qr is None:
        return None
    if weight_uq_qr.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return 0
    if weight_uq_qr.dtype == torch.float8_e4m3fn and weight_scale is not None:
        return 3
    return None


def _forward(*args, **kwargs):
    # torch hands tensors to a library impl positionally: (token_x, weight_dq,
    # weight_uq_qr, ...), while keyword-only arguments arrive as kwargs.
    weight_uq_qr = kwargs.get("weight_uq_qr")
    if weight_uq_qr is None and len(args) > 2:
        weight_uq_qr = args[2]
    mode = infer_weight_quant_mode(weight_uq_qr, kwargs.get("dequant_scale_w_uq_qr"))
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
print("shim fixed for positional dispatch")
