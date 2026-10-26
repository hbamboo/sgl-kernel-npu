"""Measure the PyBind path against torch.ops (the dispatcher costs ~16.7 us/call)."""
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
torch.manual_seed(0)


def timeit(fn, warmup=50, iters=300):
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


for T in (1, 8):
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
    ref = torch.ops.npu.mla_prolog_v3(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc, **common)
    _C.mla_prolog_v3_out(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc, q, qr, dqn, qn, dqnorm, **common)
    torch_npu.npu.synchronize()
    ok = torch.equal(q, ref[0]) and torch.equal(qr, ref[1]) and torch.equal(qn, ref[3])

    t_alloc = timeit(lambda: torch.ops.npu.mla_prolog_v3(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc, **common))
    t_torch = timeit(lambda: torch.ops.npu.mla_prolog_v3_out(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
                                                            q, qr, dqn, qn, dqnorm, **common))
    t_py = timeit(lambda: _C.mla_prolog_v3_out(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
                                               q, qr, dqn, qn, dqnorm, **common))
    print("T=%d  torch.ops alloc=%6.2f  torch.ops out=%6.2f  pybind out=%6.2f  (-%.1f%% vs torch out)  identical=%s"
          % (T, t_alloc, t_torch, t_py, (t_torch - t_py) / t_torch * 100, ok))
    del Wdq, Wuqqr, Wdkv, Wuk
    torch_npu.npu.empty_cache()
