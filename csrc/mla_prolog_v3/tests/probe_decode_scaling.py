"""How does the decode kernel cost split between its fixed overhead and the weight streaming?

Sweeps He (moves mm1/mm2 bytes) and N (moves mm3/mm4 bytes) at T=1 so the intercept of
    t = fixed + a * weight_bytes
can be read off; that intercept is what a leaner decode kernel would have to attack.
"""
import statistics

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
Hcq, Hckv, D, Dr = 768, 512, 192, 64


def bench(fn, warmup=20, iters=60):
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


T = 1
print("%6s %4s %10s %10s" % ("He", "N", "MB", "us"))
rows = []
for He in (1024, 2048):
    for N in (2, 20):
        torch.manual_seed(1)
        nz = lambda t: torch_npu.npu_format_cast(t, 29)  # noqa: E731
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
        kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
        krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
        idx = torch.arange(T, dtype=torch.int64, device=dev)
        mb = (He * (Hcq + Hckv + Dr) + Hcq * N * (D + Dr) + N * D * Hckv) * 2 / 2 ** 20
        us = bench(lambda: torch.ops.npu.mla_prolog_v3(
            x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc, cache_index=idx,
            cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=0, kv_cache_quant_mode=0))
        rows.append((mb, us))
        print("%6d %4d %10.2f %10.2f" % (He, N, mb, us))
        del Wdq, Wuqqr, Wdkv, Wuk
        torch_npu.npu.empty_cache()

(mb1, t1), (mb2, t2) = rows[0], rows[-1]
slope = (t2 - t1) / (mb2 - mb1)
print()
print("fit: t = %.1f us + %.2f us/MB (bytes %.1f..%.1f MB)" % (t1 - slope * mb1, slope, mb1, mb2))
print("at the real weight size (%.1f MB): %.1f us of which %.1f us is the fixed part"
      % (mb2, t2, t2 - slope * mb2))
