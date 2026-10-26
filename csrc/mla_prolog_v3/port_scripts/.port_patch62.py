"""Handle both weight orientations in prepare_mlapolog_bf16_weights.

Measured module state with the MXFP8 GLM checkpoint:
  * fused qkv_a projection: [out_total, He] fp8 with scale_inv [He/64, out, 2]  (NOT transposed)
  * q_b projection:         [Hcq, out] fp8 with scale_inv [Hcq/64, out, 2]      (transposed)
So the helper must detect the orientation from the scale's ``out`` dim instead of
assuming [in, out].
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''def with_kv_b_proj_ignored(ignore):'''
NEW = '''def dequant_to_in_out(weight, scale_inv):
    """FP8 weight (+ MXFP8 ``weight_scale_inv``) -> bf16 ``[in, out]``, or None.

    Detects the weight orientation from the scale's output dimension, so it handles
    both the scheme's transposed ``[in, out]`` and a plain ``[out, in]`` weight.
    """
    if weight is None or scale_inv is None or weight.dtype != torch.float8_e4m3fn:
        return None
    nd = to_nd(weight)
    if scale_inv.dim() != 3:
        return None
    out_dim = scale_inv.shape[1]
    if nd.shape[1] == out_dim:               # [in, out]
        scale = _scale_from_inv(scale_inv, out_dim, nd.shape[0])
        return dequant_mxfp8(nd, scale) if scale is not None else None
    if nd.shape[0] == out_dim:               # [out, in] -> transpose
        scale = _scale_from_inv(scale_inv, out_dim, nd.shape[1])
        if scale is None:
            return None
        return dequant_mxfp8(nd, scale).t().contiguous()
    return None


def with_kv_b_proj_ignored(ignore):'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = '''    # --- fused q/kv down projection -> split into q_a and kv_a
    if fused_w is not None and hcq:
        nd = to_nd(fused_w)
        if nd.dtype == torch.float8_e4m3fn:
            scale = _scale_from_inv(getattr(fused, "weight_scale_inv", None), nd.shape[1], nd.shape[0])
            nd = dequant_mxfp8(nd, scale) if scale is not None else None
        if nd is not None and nd.dtype == torch.bfloat16 and nd.dim() == 2 and nd.shape[1] > hcq:
            module.q_a_proj_weight = torch_npu.npu_format_cast(nd[:, :hcq].contiguous(), _NZ_FORMAT)
            module.kv_a_proj_weight = torch_npu.npu_format_cast(nd[:, hcq:].contiguous(), _NZ_FORMAT)

    # --- q_b up projection -> bf16 NZ behind a holder (the call site reads .weight)
    if q_b_w is not None and q_b_w.dtype == torch.float8_e4m3fn:
        nd = to_nd(q_b_w)
        scale = _scale_from_inv(getattr(q_b, "weight_scale_inv", None), nd.shape[1], nd.shape[0])
        if scale is not None:
            module.q_b_proj = _WeightHolder(torch_npu.npu_format_cast(dequant_mxfp8(nd, scale), _NZ_FORMAT))
    elif q_b_w is not None and q_b_w.dtype == torch.bfloat16:
        module.q_b_proj = _WeightHolder(ensure_nz(q_b_w))
    return module'''

NEW2 = '''    # --- fused q/kv down projection -> split into q_a and kv_a
    if fused_w is not None and hcq:
        nd = dequant_to_in_out(fused_w, getattr(fused, "weight_scale_inv", None))
        if nd is None:
            plain = to_nd(fused_w)
            nd = plain if plain is not None and plain.dtype == torch.bfloat16 else None
        if nd is not None and nd.dim() == 2 and nd.shape[1] > hcq:
            module.q_a_proj_weight = torch_npu.npu_format_cast(nd[:, :hcq].contiguous(), _NZ_FORMAT)
            module.kv_a_proj_weight = torch_npu.npu_format_cast(nd[:, hcq:].contiguous(), _NZ_FORMAT)

    # --- q_b up projection -> bf16 NZ behind a holder (the call site reads .weight)
    if q_b_w is not None:
        nd = dequant_to_in_out(q_b_w, getattr(q_b, "weight_scale_inv", None))
        if nd is None:
            plain = to_nd(q_b_w)
            nd = plain if plain is not None and plain.dtype == torch.bfloat16 else None
        if nd is not None:
            module.q_b_proj = _WeightHolder(torch_npu.npu_format_cast(nd, _NZ_FORMAT))
    return module'''
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("orientation-aware dequant added")
