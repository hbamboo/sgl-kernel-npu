"""Fix orientation handling in the shim's MXFP8 dequant.

The op's ABI takes weights as [in, out] and e8m0 scales as [out, in/32], so the scale
has to be expanded along the *rows* for that layout (and along the columns for the
checkpoint's [out, in] layout). Accept both and verify each path.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''def dequant_mxfp8(weight, scale):
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
        return weight, False'''

NEW = '''def dequant_mxfp8(weight, scale, block=32):
    """float8_e4m3 weight + e8m0 per-``block`` scale -> bf16, layout aware.

    Supports both the op ABI (weight ``[in, out]`` with scale ``[out, in/block]``) and
    the checkpoint layout (weight ``[out, in]`` with scale ``[out, in/block]``).
    """
    w = weight.to(torch.float32)
    mult = _e8m0_scale(scale)
    if mult.shape[-1] * block == w.shape[0] and mult.shape[0] == w.shape[1]:
        # op ABI: expand the scale down the rows
        m = mult.t().repeat_interleave(block, dim=0)[: w.shape[0]]
    else:
        m = mult.repeat_interleave(block, dim=1)[:, : w.shape[1]]
    return (w * m).to(torch.bfloat16)


def _is_mxfp8(weight, scale, block=32):
    if weight is None or scale is None:
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2 or scale.numel() == 0:
        return False
    return scale.shape[-1] * block in (weight.shape[0], weight.shape[1])


def ensure_bf16(weight, scale, block=32):
    """Return (bf16 weight, True) when ``weight`` is MXFP8; cached by identity."""
    if not _is_mxfp8(weight, scale, block):
        return weight, False'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("dequant orientation fixed")
