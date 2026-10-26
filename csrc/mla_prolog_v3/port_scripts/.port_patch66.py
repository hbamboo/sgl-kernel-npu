"""Drop the forced forward() diversion: it desynchronises the caller's unpacking.

`npu_mla_preprocess` decides by itself whether to unpack 7 or 8 return values (based
on `quant_config.ignore`); forcing `forward_mlaprolog` when the caller chose the 7-value
path raises `too many values to unpack`.  Diverting is only safe when the caller agrees,
so the shim keeps the standard MLAPO wiring instead of hijacking it.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''    orig_forward = mla_cls.forward

    def patched_forward(self, positions, hidden_states, forward_batch, zero_allocator):
        # forward() prefers the int8 forward_mlapo whenever the quant method reports
        # modelslim - MXFP8 also does - but that path needs W8A8 `input_offset`s.
        for attr in ("qkv_a_proj", "q_b_proj"):
            mod = getattr(self, attr, None)
            w = getattr(mod, "weight", None)
            if w is not None and w.dtype == torch.float8_e4m3fn:
                return self.forward_mlaprolog(positions, hidden_states, forward_batch)
        return orig_forward(self, positions, hidden_states, forward_batch, zero_allocator)

    mla_cls.forward = patched_forward
    return True'''
NEW = '''    # NOTE: do not divert forward() to forward_mlaprolog here.  The caller
    # (`npu_mla_preprocess`) decides how many values to unpack from
    # `quant_config.ignore`; hijacking the return shape raises "too many values to
    # unpack".  SGLang's intended MLAPO configuration (W8A8 int8 projections with an
    # unquantised kv_b_proj) already takes that branch by itself.
    return True'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("forced diversion removed")
