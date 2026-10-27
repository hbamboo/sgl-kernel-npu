"""One case per process: does raising the per-step token count above 128 overflow in mm3/mm4?

Usage: python3 probe_step_overflow.py <T> <N> <D> <step> <qn:0|1>
A device error (507015) leaves the context unusable, so each case must run in its own process.
"""
import os
import sys

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

T, N, D, step, qn = (int(v) for v in sys.argv[1:6])
os.environ["MLA_PROLOG_V3_MAX_STEP"] = str(step)

dev = "npu:0"
He, Hcq, Hckv, Dr = 2048, 768, 512, 64
nz = lambda t: torch_npu.npu_format_cast(t, 29)  # noqa: E731
torch.manual_seed(0)

Wdq = nz(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02)
Wuqqr = nz(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02)
Wdkv = nz(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02)
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
gq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
gkv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T)
freq = torch.outer(pos.float(), 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr)))
cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
blocks = (T + 127) // 128 + 1
kv = torch.zeros(blocks, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
kr = torch.zeros(blocks, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

try:
    out = torch.ops.npu.mla_prolog_v3(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kv, kr,
                                      cache_index=idx, cache_mode="PA_BSND", query_norm_flag=bool(qn),
                                      weight_quant_mode=0, kv_cache_quant_mode=0)
    torch_npu.npu.synchronize()
    print("T=%-4d N=%-4d D=%-4d step=%-4d qn=%d -> OK" % (T, N, D, step, qn))
    sys.exit(0)
except Exception as exc:  # noqa: BLE001
    msg = str(exc)
    code = "507015" if "507015" in msg else "other"
    print("T=%-4d N=%-4d D=%-4d step=%-4d qn=%d -> FAIL(%s)" % (T, N, D, step, qn, code))
    sys.exit(1)
