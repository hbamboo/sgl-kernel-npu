"""Attribute prefill cost by sweeping He (mm1/mm2 contraction dim).

At T=256 the kernel is ~94 us.  mm1 (x@W_DQ) and mm2 (x@W_DKV/W_KR) scale linearly
with He while mm3/mm4 do not, so the slope tells us their share of the time.
"""
import statistics

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
Hcq, Hckv, D, Dr, N = 768, 512, 192, 64, 20
T = 256


def timeit(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch_npu.npu.synchronize()
    evs = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(iters)]
    for i in range(iters):
        evs[i][0].record()
        fn()
        evs[i][1].record()
    torch_npu.npu.synchronize()
    return statistics.median([s.elapsed_time(e) * 1e3 for s, e in evs])


torch.manual_seed(0)
print("%6s %10s %10s %12s" % ("He", "us", "weights_MB", "mm1+2_flops_G"))
for He in (1024, 2048, 4096, 8192):
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
    kvc = torch.zeros(blocks, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
    krc = torch.zeros(blocks, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
    idx = torch.arange(T, dtype=torch.int64, device=dev)

    def call():
        torch.ops.npu.mla_prolog_v3(
            x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
            cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
            weight_quant_mode=0, kv_cache_quant_mode=0)

    us = timeit(call)
    wbytes = (He * Hcq + He * (Hckv + Dr) + Hcq * N * (D + Dr) + N * D * Hckv) * 2
    mm12 = 2 * T * He * (Hcq + Hckv + Dr)
    print("%6d %10.2f %10.2f %12.3f" % (He, us, wbytes / 2 ** 20, mm12 / 1e9))
    del Wdq, Wuqqr, Wdkv, Wuk
    torch_npu.npu.empty_cache()
