"""Cross-check the ported kernel against the CANN built-in MlaPrologV3 over several
geometries, including the ones the built-in binary supports natively.

The GLM adaptation (Hcq=768, D=192, MXFP8) only changed the *supported* parameter space, so
the original DeepSeek-like geometries must stay bit-identical.  This covers Hcq 1536/2048 and
N 8/16/32/128, with D 128 and 192.
"""
import sys

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
T = 8
Hckv, Dr = 512, 64

# (He, Hcq, D, N) - all accepted by the built-in operator
GEOMS = [
    (7168, 1536, 128, 16),
    (7168, 2048, 128, 32),
    (4096, 1536, 192, 8),
    (7168, 2048, 128, 128),
]

fails = []
for gi, (He, Hcq, D, N) in enumerate(GEOMS):
    torch.manual_seed(gi)
    gq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
    gkv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
    Wdq = torch_npu.npu_format_cast(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02, 29)
    Wuqqr = torch_npu.npu_format_cast(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02, 29)
    Wdkv = torch_npu.npu_format_cast(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02, 29)
    Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
    x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
    pos = torch.arange(T)
    inv = 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr))
    freq = torch.outer(pos.float(), inv)
    cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
    sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
    blocks = (T + 127) // 128 + 1
    idx = torch.arange(T, dtype=torch.int64, device=dev)
    kw = dict(cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=0)

    def run(which):
        kvc = torch.zeros(blocks, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
        krc = torch.zeros(blocks, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
        if which == "ported":
            out = torch.ops.npu.mla_prolog_v3(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
                                              kv_cache_quant_mode=0, **kw)
        else:
            out = torch_npu.npu_mla_prolog_v3(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc, **kw)
        torch_npu.npu.synchronize()
        return out, kvc, krc

    p, pkv, pkr = run("ported")
    b, bkv, bkr = run("builtin")

    def diff(a, c):
        a, c = a.float(), c.float()
        return (a - c).abs().max().item()

    ds = [diff(p[0], b[0]), diff(p[1], b[1]), diff(p[3], b[3]), diff(pkv, bkv), diff(pkr, bkr)]
    ok = max(ds) == 0.0
    print("He=%-5d Hcq=%-5d D=%-4d N=%-4d  max|d| = %s  %s"
          % (He, Hcq, D, N, ["%.3e" % d for d in ds], "OK" if ok else "MISMATCH"))
    if not ok:
        fails.append((He, Hcq, D, N, ds))
    del Wdq, Wuqqr, Wdkv, Wuk
    torch_npu.npu.empty_cache()

print()
print("RESULT:", "OK (bit-identical on every geometry)" if not fails else "FAILED: %s" % fails)
sys.exit(0 if not fails else 1)
