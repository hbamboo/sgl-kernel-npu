"""Calibrate the practical ceiling for the *prefill* GEMM shapes.

Prefill runs m<=128 (L1 bound) with N 768/576/5120 and K 768/2048.  Measuring torch.matmul on
exactly those shapes (queue full, so launch overhead is amortised) gives the practical ceiling
for our shape; comparing it with the operator's 54 TFLOP/s says whether the gap is the shape or
the kernel.
"""
import time

import torch
import torch_npu

dev = "npu:0"
torch.manual_seed(0)


def steady(fn, calls=47, rounds=10):
    for _ in range(5):
        fn()
    torch_npu.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for _ in range(calls):
            fn()
    torch_npu.npu.synchronize()
    return (time.perf_counter() - t0) / (rounds * calls) * 1e6


print("%-34s %10s %12s %10s" % ("shape (m x k @ k x n)", "us", "TFLOP/s", "% of 424"))
cases = [
    (128, 2048, 768, "prefill mm1  m=128"),
    (256, 2048, 768, "mm1 m=256"),
    (512, 2048, 768, "mm1 m=512"),
    (1024, 2048, 768, "mm1 m=1024"),
    (2048, 2048, 768, "whole prefill mm1 m=2048"),
    (128, 768, 5120, "prefill mm3  m=128"),
    (512, 768, 5120, "mm3 m=512"),
    (2048, 768, 5120, "mm3 m=2048"),
]
for m, k, n, label in cases:
    a = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    b = torch.randn(k, n, dtype=torch.bfloat16, device=dev)
    t = steady(lambda: torch.matmul(a, b))
    tf = 2 * m * k * n / (t * 1e-6) / 1e12
    print("%-34s %10.2f %12.1f %9.0f%%   %s" % ("(%d x %d @ %d x %d)" % (m, k, k, n), t, tf, tf / 424 * 100, label))
    del a, b
    torch_npu.npu.empty_cache()

print()
print("operator prefill reference: 89.7 us for T=256 (2 x 128-token steps), 54 TFLOP/s")
