"""Steady-state cost per call when 47 layers are issued back to back (as in serving).

Single-call latency mixes host cost and kernel time; a serving step issues one prologue per
layer, so what matters is the pipelined throughput.  This measures 47 * repeats back to back
without intermediate synchronisation, for the three entry points.
"""
import sys
import time

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib"
sys.path.insert(0, LIB)
torch.ops.load_library(LIB + "/libsgl_kernel_npu.so")
import libsgl_kernel_npu as _C  # noqa: E402

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
LAYERS = 47
torch.manual_seed(0)

nz = lambda t: torch_npu.npu_format_cast(t, 29)  # noqa: E731
w_dq = nz(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02)
w_uq_qr = nz(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02)
w_dkv_kr = nz(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02)
w_uk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
g_cq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
g_ckv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5


def bench(fn, rounds=20):
    for _ in range(5):
        fn()
    torch_npu.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(rounds):
        fn()
    torch_npu.npu.synchronize()
    return (time.perf_counter() - t0) / rounds * 1e6


for T in (1, 8, 64):
    x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
    pos = torch.arange(T)
    freq = torch.outer(pos.float(), 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr)))
    cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
    sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
    kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
    krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
    idx = torch.arange(T, dtype=torch.int64, device=dev)
    kw = dict(cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
              weight_quant_mode=0, kv_cache_quant_mode=0)
    bufs = (torch.empty(T, N, Hckv, dtype=torch.bfloat16, device=dev),
            torch.empty(T, N, Dr, dtype=torch.bfloat16, device=dev),
            torch.empty(0, dtype=torch.float32, device=dev),
            torch.empty(T, Hcq, dtype=torch.bfloat16, device=dev),
            torch.empty(0, dtype=torch.float32, device=dev))

    def layers_torch_out():
        for _ in range(LAYERS):
            torch.ops.npu.mla_prolog_v3_out(x, w_dq, w_uq_qr, w_uk, w_dkv_kr, g_cq, g_ckv, sin, cos,
                                            kvc, krc, *bufs, **kw)

    def layers_pybind():
        for _ in range(LAYERS):
            _C.mla_prolog_v3_out(x, w_dq, w_uq_qr, w_uk, w_dkv_kr, g_cq, g_ckv, sin, cos,
                                 kvc, krc, *bufs, **kw)

    r1 = bench(layers_torch_out, rounds=5)   # microseconds per 47-layer round
    r2 = bench(layers_pybind, rounds=5)
    print("T=%-3d  per call: torch.ops %6.2f us  pybind %6.2f us | per decode step (47 layers): "
          "%.3f ms / %.3f ms"
          % (T, r1 / LAYERS, r2 / LAYERS, r1 / 1000.0, r2 / 1000.0))
