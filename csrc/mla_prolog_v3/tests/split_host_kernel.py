"""Split host launch overhead from device kernel time for MlaPrologV3."""
import statistics
import time

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 1
torch.manual_seed(0)
gq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
gkv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
Wdq = torch_npu.npu_format_cast(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wuqqr = torch_npu.npu_format_cast(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wdkv = torch_npu.npu_format_cast(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T)
inv = 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr))
freq = torch.outer(pos.float(), inv)
cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)


def call():
    torch.ops.npu.mla_prolog_v3(
        x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
        cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
        weight_quant_mode=0, kv_cache_quant_mode=0)


for _ in range(20):
    call()
torch_npu.npu.synchronize()

N_ITER = 200
t0 = time.perf_counter()
for _ in range(N_ITER):
    call()
t_host = (time.perf_counter() - t0) / N_ITER * 1e6
torch_npu.npu.synchronize()
t_total = (time.perf_counter() - t0) / N_ITER * 1e6
print("host-side per call (queue only): %.1f us" % t_host)
print("end-to-end per call (queued):    %.1f us" % t_total)

# same but with a torch.empty of the same workspace size to isolate allocator cost
ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=dev)
t0 = time.perf_counter()
for _ in range(N_ITER):
    torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=dev)
t_alloc = (time.perf_counter() - t0) / N_ITER * 1e6
print("at::empty(16MiB) per call:       %.1f us" % t_alloc)
