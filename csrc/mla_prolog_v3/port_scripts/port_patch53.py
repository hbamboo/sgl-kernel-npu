"""Make the shim dequantise MXFP8 projection weights on first use.

The only numerically correct configuration for GLM-4.7-Flash with this kernel is bf16
weights with ``weight_quant_mode=0`` (native MXFP8 mis-maps the ``weight_uq_qr`` per-32
scales at Hcq=768).  SGLang hands over the checkpoint's fp8 weights plus e8m0 scales,
so the shim dequantises them once (cached) and switches to mode 0 — the serving stack
needs no changes at all.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''# scale arguments that only make sense for a quantised weight mode'''
NEW = '''_BF16_CACHE = {}


def clear_dequant_cache():
    """Drop the cached MXFP8 -> bf16 dequantised weights."""
    _BF16_CACHE.clear()


def _e8m0_scale(scale):
    """uint8 e8m0 -> float32 multiplier (2^(e-127); e == 0 is treated as 0)."""
    if scale.dtype == torch.float32:
        return scale
    e = scale.to(torch.int32)
    val = torch.pow(2.0, (e - 127).to(torch.float32))
    return torch.where(e == 0, torch.zeros_like(val), val)


def dequant_mxfp8(weight, scale):
    """[out, in] float8_e4m3 + [out, in/32] e8m0 scale -> bf16 [out, in]."""
    w = weight.to(torch.float32)
    mult = _e8m0_scale(scale).repeat_interleave(32, dim=1)[:, : w.shape[1]]
    return (w * mult).to(torch.bfloat16)


def ensure_bf16(weight, scale, block=32):
    """Return (bf16 weight, True) when ``weight`` is MXFP8; cached by identity."""
    if weight is None or scale is None:
        return weight, False
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        return weight, False
    if scale.numel() == 0 or scale.shape[-1] * block != weight.shape[-1]:
        return weight, False
    key = (weight.data_ptr(), tuple(weight.shape), scale.data_ptr(), tuple(scale.shape))
    cached = _BF16_CACHE.get(key)
    if cached is None:
        cached = dequant_mxfp8(weight, scale)
        _BF16_CACHE[key] = cached
    return cached, True


# scale arguments that only make sense for a quantised weight mode'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = '''    mode = infer_weight_quant_mode(weight_uq_qr, kwargs.get("dequant_scale_w_uq_qr"))
    if mode is not None and kwargs.get("weight_quant_mode") != mode:'''
NEW2 = '''    # MXFP8 -> bf16 (the only numerically safe configuration for Hcq=768), cached
    args = list(args)
    for w_idx, s_name in ((1, "dequant_scale_w_dq"), (2, "dequant_scale_w_uq_qr"), (4, "dequant_scale_w_dkv_kr")):
        if w_idx < len(args):
            converted, done = ensure_bf16(args[w_idx], kwargs.get(s_name))
            if done:
                args[w_idx] = converted
    weight_uq_qr = args[2] if len(args) > 2 else kwargs.get("weight_uq_qr")

    mode = infer_weight_quant_mode(weight_uq_qr, kwargs.get("dequant_scale_w_uq_qr"))
    if mode is not None and kwargs.get("weight_quant_mode") != mode:'''
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)

OLD3 = '''    # the kernel needs FRACTAL_NZ for the three projection matrices
    args = list(args)
    for idx in (1, 2, 4):  # weight_dq, weight_uq_qr, weight_dkv_kr'''
NEW3 = '''    # the kernel needs FRACTAL_NZ for the three projection matrices
    for idx in (1, 2, 4):  # weight_dq, weight_uq_qr, weight_dkv_kr'''
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)

# after dequantising, the tensors are bf16 -> infer says mode 0, and the scale args are
# dropped by the existing branch; make sure the dtypes drive that decision
old_head = '''    weight_uq_qr = kwargs.get("weight_uq_qr")
    if weight_uq_qr is None and len(args) > 2:
        weight_uq_qr = args[2]
'''
assert s.count(old_head) == 1
s = s.replace(old_head, "")
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("MXFP8 auto-dequant added to the shim")
