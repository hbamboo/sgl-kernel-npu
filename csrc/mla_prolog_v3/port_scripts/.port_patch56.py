"""Module-level bf16 weight preparation for SGLang's MLAPO path (MXFP8 checkpoints).

SGLang's `NPUFusedMLAPreprocess` assumes a W8A8-int8 quantisation scheme:
  * it reads `q_b_proj.weight_scale`, which the MXFP8 scheme deletes after renaming to
    `weight_scale_inv` (and it has already transposed the weight to [in, out]);
  * `forward_mlaprolog` only forwards the q_b scale, so nothing can dequantise the two
    down-projection weights at call time.

This helper dequantises the three MLA projections to bf16 (FRACTAL_NZ) once, so the
ported kernel can run them in its numerically-verified configuration. It is validated
offline against the CPU golden in `tests/test_mlapolog_weight_prep.py`.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''def mla_prolog_v3_out(*args, **kwargs):'''
NEW = '''def _scale_from_inv(scale_inv, out_dim, in_dim, block=32):
    """Rebuild the [out, in/block] e8m0 scale from the MXFP8 scheme's [in/2b, out, 2] view."""
    if scale_inv is None:
        return None
    if scale_inv.dim() == 3:
        # [in/2b, out, 2] (or [out, in/2b, 2] for an unprocessed module)
        if scale_inv.shape[1] == out_dim:
            scale = scale_inv.permute(1, 0, 2).reshape(out_dim, -1)
        else:
            scale = scale_inv.reshape(out_dim, -1)
    else:
        scale = scale_inv.reshape(out_dim, -1)
    want = in_dim // block
    if scale.shape[1] != want:
        return None
    return scale.contiguous()


def _dequant_nz(fp8_weight, scale, block=32):
    """fp8 (any format) + [out, in/block] e8m0 scale -> bf16 FRACTAL_NZ."""
    if fp8_weight is None or scale is None:
        return None
    nd = fp8_weight
    fmt = None
    try:
        fmt = torch_npu.get_npu_format(nd)
    except Exception:  # noqa: BLE001
        fmt = None
    if fmt is not None and fmt != 2:  # 2 == ACL_FORMAT_ND
        nd = torch_npu.npu_format_cast(nd, 2)
    bf16 = dequant_mxfp8(nd, scale)
    return torch_npu.npu_format_cast(bf16, _NZ_FORMAT)


class _WeightHolder:
    """Minimal stand-in exposing ``.weight`` (what forward_mlaprolog reads)."""

    def __init__(self, weight):
        self.weight = weight


def prepare_mlapolog_bf16_weights(module):
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
            module.q_a_proj_weight = _dequant_nz(
                module.q_a_proj_weight, scale_fused[:hcq].contiguous())
            module.kv_a_proj_weight = _dequant_nz(
                module.kv_a_proj_weight, scale_fused[hcq:].contiguous())

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


def mla_prolog_v3_out(*args, **kwargs):'''

assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

# make the MLAPO constructor tolerate a missing weight_scale (MXFP8 / float q_b)
OLD2 = '''    cls.__init__ = patched_init
    cls._mlaprolog_patched = True
    return True'''
NEW2 = '''    cls.__init__ = patched_init

    # MXFP8 checkpoints have no `q_b_proj.weight_scale` (the scheme renames it to
    # weight_scale_inv), so the MLAPO constructor would raise.  Provide an empty
    # placeholder: the ported kernel never consumes it (the shim works in bf16).
    if not getattr(cls, "_mlaprolog_ctor_patched", False):
        ctor = cls.__init__

        def patched_ctor(self, *args, **kwargs):
            try:
                ctor(self, *args, **kwargs)
            except AttributeError as exc:
                if "weight_scale" not in str(exc):
                    raise
                q_b = kwargs.get("q_b_proj")
                if q_b is None:
                    q_b = args[3]
                q_b.weight_scale = torch.empty(0, dtype=torch.float32)
                ctor(self, *args, **kwargs)

        cls.__init__ = patched_ctor
        cls._mlaprolog_ctor_patched = True

    cls._mlaprolog_patched = True
    return True'''
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("module-level bf16 weight prep added")
