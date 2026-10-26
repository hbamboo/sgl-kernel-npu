"""Where do the ~15 us of host time in the PyBind path go?

Times: empty binding call, the real call, and a call whose kernel is already resident.
The difference isolates pybind casting + our host code from the kernel launch itself.
"""
import statistics
import sys

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib"
sys.path.insert(0, LIB)
torch.ops.load_library(LIB + "/libsgl_kernel_npu.so")
import libsgl_kernel_npu as _C  # noqa: E402

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 1
torch.manual_seed(0)


def timeit(fn, warmup=50, iters=400):
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
kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

q = torch.empty(T, N, Hckv, dtype=torch.bfloat16, device=dev)
qr = torch.empty(T, N, Dr, dtype=torch.bfloat16, device=dev)
dqn = torch.empty(0, dtype=torch.float32, device=dev)
qn = torch.empty(T, Hcq, dtype=torch.bfloat16, device=dev)
dqnorm = torch.empty(0, dtype=torch.float32, device=dev)
common = dict(cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
              weight_quant_mode=0, kv_cache_quant_mode=0)

import time


def host_cost(fn, iters=2000):
    """Wall-clock per call with the queue kept busy (no sync inside the loop)."""
    for _ in range(200):
        fn()
    torch_npu.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch_npu.npu.synchronize()
    return (t1 - t0) / iters * 1e6


noop = host_cost(lambda: _C.noop()) if hasattr(_C, "noop") else 0.0
py = host_cost(lambda: _C.mla_prolog_v3_out(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
                                            q, qr, dqn, qn, dqnorm, **common))
ops = host_cost(lambda: torch.ops.npu.mla_prolog_v3_out(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
                                                       q, qr, dqn, qn, dqnorm, **common))
print("empty pybind call        : %6.2f us" % noop)
print("pybind kernel call       : %6.2f us" % py)
print("torch.ops kernel call    : %6.2f us" % ops)
print("=> our host code         : %6.2f us   (kernel ~18.2 us; dispatcher overhead = %.2f us)"
      % (py - noop, ops - py))
