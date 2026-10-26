"""Split the 23.4 us per-call floor into dispatcher vs launch cost."""
import statistics

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"


def timeit(fn, warmup=50, iters=500):
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


x = torch.randn(64, dtype=torch.float32, device=dev)
y = torch.empty_like(x)
print("dispatcher only  (version op) : %6.2f us" % timeit(lambda: torch.ops.npu.sgl_kernel_npu_version()))
print("tiny kernel      (helloworld) : %6.2f us" % timeit(lambda: torch.ops.npu.helloworld(x, y)))
print("2 tiny kernels               : %6.2f us"
      % timeit(lambda: (torch.ops.npu.helloworld(x, y), torch.ops.npu.helloworld(x, y))))
print("torch elementwise add        : %6.2f us" % timeit(lambda: torch.add(x, x)))
