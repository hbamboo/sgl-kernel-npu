"""Bypass SGLang's MLAPO weight preprocessing for MXFP8 checkpoints.

With an MXFP8 checkpoint SGLang's `mlaprolog_preprocess_weight` produces wrong tensors:
the scheme already transposed the fused projection to [in, out] and the preprocess
transposes it back.  Instead of fixing that in place, the patch recomputes the three
projections as bf16 FRACTAL_NZ straight from the fp8 weights plus the scheme's
`weight_scale_inv`, which is what the ported kernel wants.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''def prepare_mlapolog_bf16_weights(module):
    """Convert an ``NPUFusedMLAPreprocess``'s three projections to bf16 NZ in place.

    Must run after ``mlaprolog_preprocess_weight()`` (which produces
    ``q_a_proj_weight`` / ``kv_a_proj_weight``) and after the quant scheme's
    ``process_weights_after_loading`` (which produces ``weight_scale_inv``).
    """
    fused = getattr(module, "qkv_a_proj", None)
    q_b = getattr(module, "q_b_proj", None)

    # down projections: q_a -> [He, Hcq], kv_a -> [He, Hckv+Dr], split by q_lora_rank
    fused_scale = getattr(fused, "weight_scale_inv", None) if fused is not None else None
    if fused_scale is not None and fused_scale.dim() == 3 and fused_scale.shape[1] > 0:
        out_total = fused_scale.shape[1]
        in_dim = fused_scale.shape[0] * 64
        scale_fused = _scale_from_inv(fused_scale, out_total, in_dim)
        if scale_fused is not None:
            hcq = module.q_lora_rank
            q_a = _dequant_nz(module.q_a_proj_weight, scale_fused[:hcq].contiguous())
            kv_a = _dequant_nz(module.kv_a_proj_weight, scale_fused[hcq:].contiguous())
            if q_a is not None:
                module.q_a_proj_weight = q_a
            if kv_a is not None:
                module.kv_a_proj_weight = kv_a
'''

NEW = '''def prepare_mlapolog_bf16_weights(module):
    """Recompute the module's three projections as bf16 FRACTAL_NZ, in place.

    Reads the fused projection straight from ``qkv_a_proj.weight`` (the MXFP8 scheme
    leaves it as [in, out] fp8 with ``weight_scale_inv``) so it does not depend on what
    SGLang's ``mlaprolog_preprocess_weight`` produced, then splits it into
    ``q_a_proj_weight`` / ``kv_a_proj_weight`` and wraps ``q_b_proj`` with a bf16 NZ
    weight.  No-op for checkpoints whose weights are already bf16.
    """
    fused = getattr(module, "qkv_a_proj", None)
    q_b = getattr(module, "q_b_proj", None)

    fused_w = getattr(fused, "weight", None) if fused is not None else None
    if fused_w is not None and fused_w.dtype == torch.float8_e4m3fn:
        in_dim, out_total = fused_w.shape
        scale = _scale_from_inv(getattr(fused, "weight_scale_inv", None), out_total, in_dim)
        bf16 = _dequant_nz(fused_w, scale)
        if bf16 is not None:
            nd = bf16
            try:
                if torch_npu.get_npu_format(nd) != 2:
                    nd = torch_npu.npu_format_cast(nd, 2)
            except Exception:  # noqa: BLE001
                nd = nd
            hcq = module.q_lora_rank
            module.q_a_proj_weight = torch_npu.npu_format_cast(nd[:, :hcq].contiguous(), _NZ_FORMAT)
            module.kv_a_proj_weight = torch_npu.npu_format_cast(nd[:, hcq:].contiguous(), _NZ_FORMAT)

    # q_b: [in, out] already transposed by the MXFP8 scheme
    if q_b is not None and getattr(q_b, "weight", None) is not None:
        w = q_b.weight
        if w.dtype == torch.float8_e4m3fn:
            out_dim, in_dim = w.shape[1], w.shape[0]
            scale = _scale_from_inv(getattr(q_b, "weight_scale_inv", None), out_dim, in_dim)
            bf16 = _dequant_nz(w, scale)
            if bf16 is not None:
                module.q_b_proj = _WeightHolder(bf16)
    return module
'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = '''        mla_cls.__init__ = patched_ctor
        mla_cls._mlaprolog_ctor_patched = True
'''
NEW2 = '''        mla_cls.__init__ = patched_ctor
        mla_cls._mlaprolog_ctor_patched = True

        # recompute bf16 NZ weights after the module's own preprocessing
        pre = mla_cls.mlaprolog_preprocess_weight

        def patched_preprocess(self, *args, **kwargs):
            out = pre(self, *args, **kwargs)
            try:
                prepare_mlapolog_bf16_weights(self)
            except Exception as exc:  # noqa: BLE001
                print("[dsh-patch] prepare_mlapolog_bf16_weights failed:", exc, flush=True)
            return out

        mla_cls.mlaprolog_preprocess_weight = patched_preprocess
        mla_cls._mlaprolog_prep_patched = True
'''
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("forward-level bf16 weight bypass wired")
