"""Performance baseline for the ported MlaPrologV3 (GLM-4.7-Flash, bf16 path)."""
import statistics

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20


def timeit(fn, warmup=10, iters=50):
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


def measure_peak_bw():
    n = 256 * 1024 * 1024
    a = torch.empty(n, dtype=torch.float16, device=dev)
    b = torch.empty_like(a)
    med = timeit(lambda: b.copy_(a), warmup=5, iters=20)
    return med, 2 * n * 2 / (med * 1e-6) / 1e12


bcopy_us, peak_tbs = measure_peak_bw()
print("device copy: %.1f us for 256MiB -> practical peak ~%.2f TB/s" % (bcopy_us, peak_tbs))

weight_bytes = (He * Hcq + He * (Hckv + Dr) + Hcq * N * (D + Dr) + N * D * Hckv) * 2
flops_per_token = 2 * (He * Hcq + He * (Hckv + Dr) + Hcq * N * (D + Dr) + N * D * Hckv)
print("weights = %.2f MiB, flops/token = %.3f MFLOP\n" % (weight_bytes / 2 ** 20, flops_per_token / 1e6))

torch.manual_seed(0)
gq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
gkv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
Wdq = torch_npu.npu_format_cast(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wuqqr = torch_npu.npu_format_cast(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wdkv = torch_npu.npu_format_cast(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02

print("%6s %10s %12s %12s %8s" % ("T", "median_us", "GB/s(eff)", "TFLOP/s", "BW%"))
for T in [1, 2, 4, 8, 16, 64, 256, 512, 1024, 2048]:
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

    med = timeit(call)
    act_bytes = T * He * 2 + T * (Hckv + Dr) * 2 + T * N * (Hckv + Dr) * 2
    total_bytes = weight_bytes + act_bytes
    gbs = total_bytes / (med * 1e-6) / 1e9
    tflops = flops_per_token * T / (med * 1e-6) / 1e12
    print("%6d %10.2f %12.1f %12.4f %8.1f" % (T, med, gbs, tflops, gbs / (peak_tbs * 1000) * 100))
