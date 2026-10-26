"""_prepare must also reorient checkpoint weights ([out, in] -> the kernel's [in, out])."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''def _prepare(weight, scale):
    """(fp8 weight, e8m0 scale) -> (kernel-ready weight, mode hint)."""
    converted, done = ensure_bf16(weight, scale)
    if done:
        return ensure_nz(converted), 0
    return ensure_nz(weight), None'''
NEW = '''def _prepare(weight, scale):
    """(fp8 weight, e8m0 scale) -> (kernel-ready weight, mode hint).

    Dequantises MXFP8 to bf16 if needed and reorients the checkpoint's ``[out, in]``
    weight to the kernel's ``[in, out]`` ABI (detected from the e8m0 scale's rows).
    """
    converted, done = ensure_bf16(weight, scale)
    if not done:
        return ensure_nz(weight), None
    if (converted.dim() == 2 and scale is not None and scale.dim() == 2
            and converted.shape[0] == scale.shape[0] and converted.shape[1] != scale.shape[0]):
        converted = converted.t().contiguous()  # [out, in] -> [in, out]
    return ensure_nz(converted), 0'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("orientation handling added")
