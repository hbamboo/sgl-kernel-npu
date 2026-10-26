"""Re-test ACL graph capture with the out-variant (stable buffers, no allocations)."""
import statistics

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

q = torch.empty(T, N, Hckv, dtype=torch.bfloat16, device=dev)
qr = torch.empty(T, N, Dr, dtype=torch.bfloat16, device=dev)
dqn = torch.empty(0, dtype=torch.float32, device=dev)
qn = torch.empty(T, Hcq, dtype=torch.bfloat16, device=dev)
dqnorm = torch.empty(0, dtype=torch.float32, device=dev)

args = [x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc, q, qr, dqn, qn, dqnorm]
kw = dict(cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
          weight_quant_mode=0, kv_cache_quant_mode=0)


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


eager = timeit(lambda: torch.ops.npu.mla_prolog_v3_out(*args, **kw))
ref = q.clone()
print("eager _out: %.2f us" % eager)

g = torch.npu.NPUGraph()
try:
    with torch.npu.graph(g):
        torch.ops.npu.mla_prolog_v3_out(*args, **kw)
    torch_npu.npu.synchronize()
    replay = timeit(lambda: g.replay())
    g.replay()
    torch_npu.npu.synchronize()
    cs = torch.nn.functional.cosine_similarity(q.float().flatten(), ref.float().flatten(), 0).item()
    print("graph replay: %.2f us  (eager/replay = %.2f)  cosine vs eager = %.7f"
          % (replay, eager / replay, cs))
except Exception as exc:  # noqa: BLE001
    print("capture failed: %s" % str(exc).splitlines()[0][:200])
