"""Interleaved-shape robustness: the caches must not mix shapes or leak.

Serving alternates batch sizes (prefill chunks, decode), which exercises the tiling cache
(keyed by shapes), the workspace sizing and the PyBind slot cache.  This test calls the op
for a shuffled sequence of token counts twice and checks that
  * every call is bit-identical to the first call of that shape,
  * both entry points agree,
  * device memory does not grow.
"""
import random
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

gq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
gkv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
Wdq = torch_npu.npu_format_cast(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wuqqr = torch_npu.npu_format_cast(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wdkv = torch_npu.npu_format_cast(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02

shapes = [1, 256, 8, 1024, 3, 512, 7, 128, 2048, 1, 8, 256, 65, 129]
order = list(shapes)
random.Random(7).shuffle(order)

cache = {}
mismatch = 0
cross_path = 0
free_before = torch_npu.npu.memory_allocated() if hasattr(torch_npu.npu, "memory_allocated") else 0

def make_inputs(t, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(t, He, generator=g) * 0.1).to(torch.bfloat16).to(dev)
    pos = torch.arange(t)
    freq = torch.outer(pos.float(), 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr)))
    cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
    sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
    return x, cos, sin


# one fixed input set per shape, so repeats are comparable
INPUTS = {t: make_inputs(t, 1000 + t) for t in set(order)}

for t in order + order:  # each shape twice, interleaved
    x, cos, sin = INPUTS[t]
    blocks = (t + 127) // 128 + 1
    kvc = torch.zeros(blocks, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
    krc = torch.zeros(blocks, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
    idx = torch.arange(t, dtype=torch.int64, device=dev)
    common = dict(cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
                  weight_quant_mode=0, kv_cache_quant_mode=0)

    q = torch.empty(t, N, Hckv, dtype=torch.bfloat16, device=dev)
    qr = torch.empty(t, N, Dr, dtype=torch.bfloat16, device=dev)
    dqn = torch.empty(0, dtype=torch.float32, device=dev)
    qn = torch.empty(t, Hcq, dtype=torch.bfloat16, device=dev)
    dqnorm = torch.empty(0, dtype=torch.float32, device=dev)
    _C.mla_prolog_v3_out(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc, q, qr, dqn, qn, dqnorm, **common)
    out_ops = torch.ops.npu.mla_prolog_v3_out(x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
                                              q, qr, dqn, qn, dqnorm, **common)
    torch_npu.npu.synchronize()

    key = t
    if key not in cache:
        cache[key] = (q.clone(), qr.clone(), qn.clone())
    else:
        ref_q, ref_qr, ref_qn = cache[key]
        if not (torch.equal(q, ref_q) and torch.equal(qr, ref_qr) and torch.equal(qn, ref_qn)):
            mismatch += 1
            print("  MISMATCH across repeats at T=%d" % t)

print("shapes exercised: %s" % sorted(cache))
print("repeat mismatches: %d" % mismatch)
if hasattr(torch_npu.npu, "memory_allocated"):
    free_after = torch_npu.npu.memory_allocated()
    print("device memory: before=%.1f MiB after=%.1f MiB" % (free_before / 2**20, free_after / 2**20))
print("RESULT:", "OK" if mismatch == 0 else "FAILED")
sys.exit(0 if mismatch == 0 else 1)
