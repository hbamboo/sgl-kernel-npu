"""allocate_outputs() must produce exactly what the allocating entry produces.

The `_out` entry validates the buffers, and their shapes are subtle (empty 1-D where nothing
is quantised, [T, Hcq/32] for an MXFP8 query scale), so the helper is the supported way to
build them.  This checks the helper against `mla_prolog_v3` for several quantisation modes.
"""
import sys

import torch
import torch_npu

sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")
import sgl_kernel_npu  # noqa: F401,E402
from sgl_kernel_npu.mla_prolog_v3 import allocate_outputs  # noqa: E402

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 4
torch.manual_seed(0)

nz = lambda t: torch_npu.npu_format_cast(t, 29)  # noqa: E731
w_dq = nz(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02)
w_uk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T)
freq = torch.outer(pos.float(), 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr)))
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)

fails = []
cases = [
    ("bf16, query_norm on, no kv quant", dict(query_norm_flag=True, weight_quant_mode=0, kv_cache_quant_mode=0)),
    ("bf16, query_norm off", dict(query_norm_flag=False, weight_quant_mode=0, kv_cache_quant_mode=0)),
    ("bf16, kv cache quantised", dict(query_norm_flag=True, weight_quant_mode=0, kv_cache_quant_mode=1)),
    ("mxfp8 (mode 3)", dict(query_norm_flag=True, weight_quant_mode=3, kv_cache_quant_mode=0)),
    ("int8 uq_qr (mode 1)", dict(query_norm_flag=True, weight_quant_mode=1, kv_cache_quant_mode=0)),
]

expected = {
    "bf16, query_norm on, no kv quant": [(T, N, Hckv), (T, N, Dr), (0,), (T, Hcq), (0,)],
    "bf16, query_norm off": [(T, N, Hckv), (T, N, Dr), (0,), (0,), (0,)],
    "bf16, kv cache quantised": [(T, N, Hckv), (T, N, Dr), (T, N, 1), (T, Hcq), (0,)],
    "mxfp8 (mode 3)": [(T, N, Hckv), (T, N, Dr), (0,), (T, Hcq), (T, Hcq // 32)],
    "int8 uq_qr (mode 1)": [(T, N, Hckv), (T, N, Dr), (0,), (T, Hcq), (T, 1)],
}

for name, kw in cases:
    got = [tuple(t.shape) for t in allocate_outputs(x, w_dq, w_uk, sin, **kw)]
    dtypes = [t.dtype for t in allocate_outputs(x, w_dq, w_uk, sin, **kw)]
    ok = got == expected[name]
    print("%-38s %-58s %s" % (name, str(got), "OK" if ok else "FAIL expected %s" % (expected[name],)))
    if not ok:
        fails.append(name)

print()
print("RESULT:", "OK" if not fails else "FAILED: %s" % fails)
sys.exit(0 if not fails else 1)
