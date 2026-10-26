"""Return a legal (dummy) dequant scale so SGLang's MLAPO tail does not crash.

With bf16 weights the kernel returns an empty `dequant_scale_q_norm` (nothing is
quantised), but SGLang's `forward_mlaprolog` unconditionally does
`dequant_q_norm.view(hidden_states.shape[0])`.  Hand it a per-token ones vector: the
value is only consumed by the W8A8 path, which is not active when the projections are
bf16.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''    try:
        return torch.ops.npu.mla_prolog_v3(*args, **kwargs)
    except Exception:
        if _DEBUG():
            _dump_args(args, kwargs)
        raise'''
NEW = '''    try:
        out = torch.ops.npu.mla_prolog_v3(*args, **kwargs)
    except Exception:
        if _DEBUG():
            _dump_args(args, kwargs)
        raise

    # SGLang's forward_mlaprolog does `dequant_q_norm.view(hidden_states.shape[0])`
    # unconditionally; with bf16 weights the kernel emits an empty scale (nothing is
    # quantised), so hand back a per-token ones vector instead of crashing.
    if (isinstance(out, tuple) and len(out) >= 5 and kwargs.get("query_norm_flag")
            and isinstance(out[4], torch.Tensor) and out[4].numel() == 0):
        tokens = out[3].shape[0] if isinstance(out[3], torch.Tensor) and out[3].dim() > 0 else None
        if tokens:
            scale = torch.ones(tokens, dtype=torch.float32, device=out[0].device)
            out = out[:4] + (scale,)
    return out'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("dummy dequant scale added")
