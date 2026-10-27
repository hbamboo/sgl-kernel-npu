"""What does this card actually deliver? Practical peak for bf16 GEMM and HBM copy.

The roofline figures in the report use the datasheet numbers (424 TFLOP/s bf16, 3.27 TB/s).
This measures what a well-tuned large GEMM and a large copy actually reach on this device, so
the operator's utilisation can be judged against a practical ceiling as well.
"""
import statistics

import torch
import torch_npu

dev = "npu:0"


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch_npu.npu.synchronize()
    evs = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(iters)]
    for i in range(iters):
        evs[i][0].record()
        fn()
        evs[i][1].record()
    torch_npu.npu.synchronize()
    return statistics.median([s.elapsed_time(e) for s, e in evs]) * 1e-3  # seconds


print("bf16 GEMM (torch.matmul, the practical cube ceiling):")
for n in (2048, 4096, 8192):
    a = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    b = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    t = bench(lambda: torch.matmul(a, b))
    tflops = 2 * n ** 3 / t / 1e12
    print("  %5d^3 : %8.3f ms  -> %7.1f TFLOP/s  (%.0f%% of 424)" % (n, t * 1e3, tflops, tflops / 424 * 100))
    del a, b
    torch_npu.npu.empty_cache()

print()
print("HBM copy (torch.clone of a 4 GiB bf16 buffer):")
numel = 2 ** 31  # 4 GiB in bf16
src = torch.empty(numel, dtype=torch.bfloat16, device=dev)
t = bench(lambda: src.clone())
gbs = 2 * numel * 2 / t / 1e9
print("  %.3f ms -> %.0f GB/s (%.0f%% of 3270)" % (t * 1e3, gbs, gbs / 3270 * 100))
