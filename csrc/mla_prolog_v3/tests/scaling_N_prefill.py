"""Decompose the prefill kernel cost: how much depends on N (mm3/mm4) vs not (mm1/mm2+fixed)?

Combined with the He sweep (which moves mm1/mm2 only), this separates
    t(T) ≈ fixed + a * He + b * N
and tells whether removing a phase (route A) can matter at all.
"""
import statistics

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr = 2048, 768, 512, 192, 64


def bench(fn, warmup=10, iters=30):
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


for T in (256, 1024):
    print("== T=%d" % T)
    print("%6s %10s %12s %12s" % ("N", "us", "mm3+mm4_GB", "mm3+mm4_FLOP_G"))
    for N in (4, 8, 12, 20):
        torch.manual_seed(1)
        gq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
        gkv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
        Wdq = torch_npu.npu_format_cast(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02, 29)
        Wuqqr = torch_npu.npu_format_cast(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02, 29)
        Wdkv = torch_npu.npu_format_cast(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02, 29)
        Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
        x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
        pos = torch.arange(T)
        freq = torch.outer(pos.float(), 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr)))
        cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
        sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
        blocks = (T + 127) // 128 + 1
        kvc = torch.zeros(blocks, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
        krc = torch.zeros(blocks, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
        idx = torch.arange(T, dtype=torch.int64, device=dev)

        def call():
            torch.ops.npu.mla_prolog_v3(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
                                        cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
                                        weight_quant_mode=0, kv_cache_quant_mode=0)

        us = bench(call)
        bytes_ = (Hcq * N * (D + Dr) + N * D * Hckv) * 2
        flops = 2 * T * (Hcq * N * (D + Dr) + N * D * Hckv)
        print("%6d %10.2f %12.2f %12.3f" % (N, us, bytes_ / 2 ** 30, flops / 1e9))
        del Wdq, Wuqqr, Wdkv, Wuk
        torch_npu.npu.empty_cache()
