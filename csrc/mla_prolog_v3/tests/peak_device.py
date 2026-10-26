"""Practical peak measurement on this card: bf16 matmul TFLOP/s."""
import statistics

import torch
import torch_npu

dev = "npu:0"


def timeit(fn, warmup=5, iters=15):
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


for n in (2048, 4096):
    a = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    b = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    us = timeit(lambda: torch.matmul(a, b))
    print("matmul %5d^3 bf16: %9.1f us -> %7.1f TFLOP/s" % (n, us, 2 * n ** 3 / (us * 1e-6) / 1e12))
    del a, b
    torch_npu.npu.empty_cache()
