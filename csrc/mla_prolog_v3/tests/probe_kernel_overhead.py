"""Compare our kernel's duration with a trivial kernel's, to separate launch ramp from work.

A "trivial" Mix kernel (the repo's helloworld) shows how much of a small kernel's device
time is the launch/ramp itself; the difference to our decode kernel is what a leaner
implementation could actually remove.
"""
import statistics

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
T, He, Hcq, Hckv, D, Dr, N = 1, 2048, 768, 512, 192, 64, 20
torch.manual_seed(0)


def bench(fn, warmup=50, iters=200):
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

mla = lambda: torch.ops.npu.mla_prolog_v3(  # noqa: E731
    x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc, cache_index=idx,
    cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=0, kv_cache_quant_mode=0)

hw = None
for name in ("npu_helloworld", "helloworld"):
    op = getattr(torch.ops.npu, name, None)
    if op is not None:
        a = torch.zeros(8, dtype=torch.float32, device=dev)
        b = torch.zeros(8, dtype=torch.float32, device=dev)
        try:
            op(a, b)
            hw = lambda: op(a, b)  # noqa: E731
        except Exception:  # noqa: BLE001
            hw = lambda: op(a)  # noqa: E731
        print("using torch.ops.npu.%s as the trivial kernel" % name)
        break

if hw is None:
    print("no trivial kernel available; only our number below")
else:
    th = bench(hw)
    print("trivial kernel : %6.2f us" % th)
print("mla decode     : %6.2f us" % bench(mla))
if hw is not None:
    print("difference     : %6.2f us (what a leaner kernel could target)" % (bench(mla) - th))
