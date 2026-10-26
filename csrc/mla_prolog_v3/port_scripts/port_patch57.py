"""Fix two bugs in the MLAPO patch helpers.

1. The constructor patch was applied to ``ModelSlimConfig`` instead of
   ``NPUFusedMLAPreprocess`` (so the missing ``weight_scale`` still crashed).
2. ``prepare_mlapolog_bf16_weights`` assigned ``None`` when a weight had no block
   scale (already bf16), wiping the projection instead of leaving it alone.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''    # MXFP8 checkpoints have no `q_b_proj.weight_scale` (the scheme renames it to
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

NEW = '''    # MXFP8 checkpoints have no `q_b_proj.weight_scale` (the scheme renames it to
    # weight_scale_inv), so NPUFusedMLAPreprocess.__init__ raises.  Provide an empty
    # placeholder: the ported kernel never consumes it (the shim works in bf16).
    try:
        from sglang.srt.hardware_backend.npu.attention import mla_preprocess as _mla
    except Exception:  # noqa: BLE001
        _mla = None
    if _mla is not None and not getattr(_mla.NPUFusedMLAPreprocess, "_mlaprolog_ctor_patched", False):
        mla_cls = _mla.NPUFusedMLAPreprocess
        ctor = mla_cls.__init__

        def patched_ctor(self, *args, **kwargs):
            try:
                ctor(self, *args, **kwargs)
            except AttributeError as exc:
                if "weight_scale" not in str(exc):
                    raise
                q_b = kwargs.get("q_b_proj")
                if q_b is None and len(args) > 3:
                    q_b = args[3]
                if q_b is not None:
                    q_b.weight_scale = torch.empty(0, dtype=torch.float32)
                ctor(self, *args, **kwargs)

        mla_cls.__init__ = patched_ctor
        mla_cls._mlaprolog_ctor_patched = True

    cls._mlaprolog_patched = True
    return True'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = '''        if scale_fused is not None:
            hcq = module.q_lora_rank
            module.q_a_proj_weight = _dequant_nz(
                module.q_a_proj_weight, scale_fused[:hcq].contiguous())
            module.kv_a_proj_weight = _dequant_nz(
                module.kv_a_proj_weight, scale_fused[hcq:].contiguous())'''
NEW2 = '''        if scale_fused is not None:
            hcq = module.q_lora_rank
            q_a = _dequant_nz(module.q_a_proj_weight, scale_fused[:hcq].contiguous())
            kv_a = _dequant_nz(module.kv_a_proj_weight, scale_fused[hcq:].contiguous())
            if q_a is not None:
                module.q_a_proj_weight = q_a
            if kv_a is not None:
                module.kv_a_proj_weight = kv_a'''
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("patch helpers fixed")
