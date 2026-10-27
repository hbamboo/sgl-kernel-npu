"""At large T, is prefill bound by weight traffic (re-streamed every 128-token step) or by
compute?  Sweeping He changes both, but the slopes tell them apart:

  * compute-bound  -> time grows ~linearly with He (FLOPs grow linearly)
  * weight-bound   -> time grows ~linearly with He as well, BUT the per-token cost is
                      T-independent; the diagnostic is whether time at T=2048 is ~8x that at
                      T=256 (pure scaling) or whether the fixed per-step weight streaming
                      dominates.
"""
import statistics

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
Hcq, Hckv, D, Dr, N = 768, 512, 192, 64, 20


def bench(fn, warmup=5, iters=20):
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


print("%6s %8s %10s %12s %12s" % ("He", "T", "us", "us per 1k tok", "us per MB wgt"))
for He in (1024, 2048, 4096):
    for T in (256, 1024, 2048):
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
        us = bench(lambda: torch.ops.npu.mla_prolog_v3(
            x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kv, kr, cache_index=idx,
            cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=0, kv_cache_quant_mode=0))
        wmb = (He * (Hcq + Hckv + Dr) + Hcq * N * (D + Dr) + N * D * Hckv) * 2 / 2 ** 20
        print("%6d %8d %10.2f %12.2f %12.2f" % (He, T, us, us / T * 1000, us / wmb))
        del Wdq, Wuqqr, Wdkv, Wuk
        torch_npu.npu.empty_cache()
