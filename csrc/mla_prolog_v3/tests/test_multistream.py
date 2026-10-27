"""Two streams must not share the scratch buffer.

SGLang issues the MLA prologue on a side stream (alt_stream) to overlap it with the rest of
the layer, so the same thread can have two in-flight launches with identical shapes.  Before
the fix the workspace cache was keyed per (tiling, device) only, so the two launches would
write into the same scratch buffer.  This test runs the operator concurrently on two streams
and compares the results with sequential execution.
"""
import sys

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib"
sys.path.insert(0, LIB)
torch.ops.load_library(LIB + "/libsgl_kernel_npu.so")
import libsgl_kernel_npu as _C  # noqa: E402

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 8
torch.manual_seed(0)

nz = lambda t: torch_npu.npu_format_cast(t, 29)  # noqa: E731
w_dq = nz(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02)
w_uq_qr = nz(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02)
w_dkv_kr = nz(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02)
w_uk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
g_cq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
g_ckv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
pos = torch.arange(T)
freq = torch.outer(pos.float(), 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr)))
cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
kw = dict(cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=0, kv_cache_quant_mode=0)


def make_inputs(seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(T, He, generator=g) * 0.1).to(torch.bfloat16).to(dev)
    kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
    krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
    idx = torch.arange(T, dtype=torch.int64, device=dev)
    return x, kvc, krc, idx


def run(x, kvc, krc, idx):
    q = torch.empty(T, N, Hckv, dtype=torch.bfloat16, device=dev)
    qr = torch.empty(T, N, Dr, dtype=torch.bfloat16, device=dev)
    dqn = torch.empty(0, dtype=torch.float32, device=dev)
    qn = torch.empty(T, Hcq, dtype=torch.bfloat16, device=dev)
    dqn2 = torch.empty(0, dtype=torch.float32, device=dev)
    _C.mla_prolog_v3_out(x, w_dq, w_uq_qr, w_uk, w_dkv_kr, g_cq, g_ckv, sin, cos, kvc, krc,
                         q, qr, dqn, qn, dqn2, cache_index=idx, **kw)
    return q, qr, qn


A = make_inputs(11)
B = make_inputs(22)

# sequential reference
refA = [t.clone() for t in run(*A)]
refB = [t.clone() for t in run(*B)]
torch_npu.npu.synchronize()

# concurrent on two streams
s1, s2 = torch.npu.Stream(), torch.npu.Stream()
outA = [torch.empty_like(t) for t in refA]
outB = [torch.empty_like(t) for t in refB]
with torch.npu.stream(s1):
    a = run(*A)
with torch.npu.stream(s2):
    b = run(*B)
torch_npu.npu.current_stream().wait_stream(s1)
torch_npu.npu.current_stream().wait_stream(s2)
torch_npu.npu.synchronize()

okA = all(torch.equal(x, y) for x, y in zip(a, refA))
okB = all(torch.equal(x, y) for x, y in zip(b, refB))
print("stream A identical to sequential:", okA)
print("stream B identical to sequential:", okB)
print("RESULT:", "OK" if okA and okB else "FAILED")
sys.exit(0 if (okA and okB) else 1)
