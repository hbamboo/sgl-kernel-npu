"""Break down the per-call host cost: dispatcher + RunOpApi floor vs our op."""
import statistics

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"


def timeit(fn, warmup=20, iters=200):
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
print("helloworld (tiny kernel)     : %6.2f us" % timeit(lambda: torch.ops.npu.helloworld(x, y)))

qkv = torch.zeros(4, 3 * 128, dtype=torch.bfloat16, device=dev)
g = torch.ones(128, dtype=torch.bfloat16, device=dev)
cos = torch.ones(4, 128, dtype=torch.bfloat16, device=dev)
sin = torch.zeros(4, 128, dtype=torch.bfloat16, device=dev)
idx = torch.arange(4, dtype=torch.int64, device=dev)
q_out = torch.empty(4, 128, dtype=torch.bfloat16, device=dev)
kc = torch.zeros(1, 128 // 16, 128, 16, dtype=torch.bfloat16, device=dev)
vc = torch.zeros(1, 128 // 16, 128, 16, dtype=torch.bfloat16, device=dev)
print("qkv_rms_norm_rope_cache tiny : %6.2f us"
      % timeit(lambda: torch.ops.npu.qkv_rms_norm_rope_cache(qkv, g, g, cos, sin, idx, q_out, kc, vc, 1, 1, 1, 1e-5)))

a = torch.randn(1024, 1024, dtype=torch.bfloat16, device=dev)
b = torch.randn(1024, 1024, dtype=torch.bfloat16, device=dev)
print("torch.matmul 1024^3 bf16     : %6.2f us" % timeit(lambda: torch.matmul(a, b), warmup=5, iters=50))
