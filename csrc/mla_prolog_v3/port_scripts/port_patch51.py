"""Make the shim NZ-cast the projection weights when the caller passes ND.

SGLang's `mlaprolog_preprocess_weight` only NZ-casts the fused qkv_a projection; it
passes `q_b_proj.weight` (ND, and the MXFP8 scheme deliberately keeps it ND) and
`w_kc` straight through. The ported kernel requires FRACTAL_NZ for the three
projection matrices (ND produces garbage: cosine 0.005), so the shim casts them and
caches the converted tensor (weights are static after loading).
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''_libs = []

# scale arguments that only make sense for a quantised weight mode'''
NEW = '''_libs = []

# FRACTAL_NZ format enum for torch_npu
_NZ_FORMAT = 29

# ND -> NZ conversions, keyed by (data_ptr, shape, dtype, device).  The projection
# weights are static after loading, so one conversion per tensor is enough; call
# clear_format_cache() if a weight is re-loaded in place.
_NZ_CACHE = {}


def clear_format_cache():
    """Drop the cached NZ conversions (call after re-loading weights in place)."""
    _NZ_CACHE.clear()


def ensure_nz(t):
    """Return ``t`` in FRACTAL_NZ, converting and caching it if needed."""
    if t is None:
        return None
    try:
        fmt = torch_npu.get_npu_format(t)
    except Exception:  # noqa: BLE001 - not an NPU tensor / unknown format
        return t
    if fmt == _NZ_FORMAT:
        return t
    key = (t.data_ptr(), tuple(t.shape), t.dtype, t.device.index)
    cached = _NZ_CACHE.get(key)
    if cached is not None:
        return cached
    converted = torch_npu.npu_format_cast(t, _NZ_FORMAT)
    _NZ_CACHE[key] = converted
    return converted


# scale arguments that only make sense for a quantised weight mode'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = '''    mode = infer_weight_quant_mode(weight_uq_qr, kwargs.get("dequant_scale_w_uq_qr"))
    if mode is not None and kwargs.get("weight_quant_mode") != mode:
        kwargs["weight_quant_mode"] = mode
        if mode == 0:
            for name in _WEIGHT_SCALE_ARGS:
                kwargs.pop(name, None)
    return torch.ops.npu.mla_prolog_v3(*args, **kwargs)'''
NEW2 = '''    mode = infer_weight_quant_mode(weight_uq_qr, kwargs.get("dequant_scale_w_uq_qr"))
    if mode is not None and kwargs.get("weight_quant_mode") != mode:
        kwargs["weight_quant_mode"] = mode
        if mode == 0:
            for name in _WEIGHT_SCALE_ARGS:
                kwargs.pop(name, None)
    # the kernel needs FRACTAL_NZ for the three projection matrices
    args = list(args)
    for idx in (1, 2, 4):  # weight_dq, weight_uq_qr, weight_dkv_kr
        if idx < len(args):
            args[idx] = ensure_nz(args[idx])
    return torch.ops.npu.mla_prolog_v3(*args, **kwargs)'''
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("NZ auto-cast added to the shim")
