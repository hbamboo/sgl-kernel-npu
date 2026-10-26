"""Dispatch MXFP8 modules to forward_mlaprolog instead of the int8 forward_mlapo.

`NPUFusedMLAPreprocess.forward` prefers `forward_mlapo` whenever the projection's quant
method reports the modelslim config name - true for MXFP8 too - and that path needs the
W8A8 int8 offsets (`input_offset`), which MXFP8 checkpoints do not have.  Route fp8
modules to `forward_mlaprolog`, which is the path the ported kernel implements.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''        mla_cls.mlaprolog_preprocess_weight = patched_preprocess
        mla_cls._mlaprolog_prep_patched = True
'''
NEW = '''        mla_cls.mlaprolog_preprocess_weight = patched_preprocess
        mla_cls._mlaprolog_prep_patched = True

        # forward() prefers the int8 `forward_mlapo` whenever the quant method reports
        # modelslim - which MXFP8 also does - but that path needs `input_offset`.
        # Route fp8 projections to `forward_mlaprolog` instead.
        orig_forward = mla_cls.forward

        def patched_forward(self, positions, hidden_states, forward_batch, zero_allocator):
            for attr in ("qkv_a_proj", "q_b_proj"):
                mod = getattr(self, attr, None)
                w = getattr(mod, "weight", None)
                if w is not None and w.dtype == torch.float8_e4m3fn:
                    return self.forward_mlaprolog(positions, hidden_states, forward_batch)
            return orig_forward(self, positions, hidden_states, forward_batch, zero_allocator)

        mla_cls.forward = patched_forward
        mla_cls._mlaprolog_forward_patched = True
'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("forward dispatch patch added")
