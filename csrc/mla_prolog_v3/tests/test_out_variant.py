"""Verify the out-variant against the allocating one and measure the gain."""
import statistics

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
torch.manual_seed(0)


def timeit(fn, warmup=20, iters=100):
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


for T in (1, 8, 256):
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
    blocks = (T + 127) // 128 + 1
    kvc = torch.zeros(blocks, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
    krc = torch.zeros(blocks, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
    idx = torch.arange(T, dtype=torch.int64, device=dev)

    args = [x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc]
    kw = dict(cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
              weight_quant_mode=0, kv_cache_quant_mode=0)

    ref = torch.ops.npu.mla_prolog_v3(*args, **kw)
    torch_npu.npu.synchronize()

    q = torch.empty(T, N, Hckv, dtype=torch.bfloat16, device=dev)
    qr = torch.empty(T, N, Dr, dtype=torch.bfloat16, device=dev)
    dqn = torch.empty(0, N, 1, dtype=torch.float32, device=dev)
    qn = torch.empty(T, Hcq, dtype=torch.bfloat16, device=dev)
    dqnorm = torch.empty(0, dtype=torch.float32, device=dev)

    torch.ops.npu.mla_prolog_v3_out(*args, q, qr, dqn, qn, dqnorm, **kw)
    torch_npu.npu.synchronize()
    ok = all(torch.allclose(a.float(), b.float(), atol=0, rtol=0)
             for a, b in zip((q, qr, qn), (ref[0], ref[1], ref[3])))

    t_alloc = timeit(lambda: torch.ops.npu.mla_prolog_v3(*args, **kw))
    t_out = timeit(lambda: torch.ops.npu.mla_prolog_v3_out(*args, q, qr, dqn, qn, dqnorm, **kw))
    print("T=%3d  alloc=%7.2f us  out=%7.2f us  gain=%.1f%%  bit-identical=%s"
          % (T, t_alloc, t_out, (t_alloc - t_out) / t_alloc * 100, ok))
    del Wdq, Wuqqr, Wdkv, Wuk
    torch_npu.npu.empty_cache()
