"""Compare the ported kernel with the CANN built-in MlaPrologV3 on a geometry the
built-in binary supports (DeepSeek-like), to see whether the port regressed.
"""
import statistics

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 7168, 1536, 512, 128, 64, 16


def timeit(fn, warmup=10, iters=50):
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


torch.manual_seed(0)
for T in (1, 8, 64):
    gq = torch.ones(Hcq, dtype=torch.bfloat16, device=dev)
    gkv = torch.ones(Hckv, dtype=torch.bfloat16, device=dev)
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
    blocks = (T + 127) // 128 + 1
    kvc = torch.zeros(blocks, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
    krc = torch.zeros(blocks, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
    idx = torch.arange(T, dtype=torch.int64, device=dev)

    ported = timeit(lambda: torch.ops.npu.mla_prolog_v3(
        x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
        cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
        weight_quant_mode=0, kv_cache_quant_mode=0))

    def builtin():
        torch_npu.npu_mla_prolog_v3(
            x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
            cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
            weight_quant_mode=0)

    try:
        built = timeit(builtin)
        print("T=%3d  ported=%7.2f us   builtin=%7.2f us   ratio=%.2f"
              % (T, ported, built, ported / built))
    except Exception as exc:  # noqa: BLE001
        print("T=%3d  ported=%7.2f us   builtin failed: %s" % (T, ported, str(exc).splitlines()[0][:100]))
