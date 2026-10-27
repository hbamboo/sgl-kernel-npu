"""Numerical + performance feasibility of fusing the two down-projection GEMMs.

Route A in DESIGN-kernel-rewrite.md: concat W_DQ and W_DKV/KR into [He, 1344] and issue one
GEMM.  Two questions answered here, without touching the kernel:

1. does the fused GEMM produce the same numbers as the two separate ones?
2. how much faster is one GEMM of N=1344 than two of N=768 and N=576 on this hardware?
   (that difference is the upper bound of what route A can buy)
"""
import statistics

import torch
import torch_npu

dev = "npu:0"
He, Hcq, Hkv = 2048, 768, 576
torch.manual_seed(0)

W_dq = torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02
W_dkv = torch.randn(He, Hkv, dtype=torch.bfloat16, device=dev) * 0.02
W_fused = torch.cat([W_dq, W_dkv], dim=1).contiguous()


def bench(fn, warmup=20, iters=100):
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


print("%6s %10s %10s %10s %10s %8s" % ("m", "sep1", "sep2", "sep_sum", "fused", "gain"))
for m in (1, 8, 64, 128, 256):
    x = torch.randn(m, He, dtype=torch.bfloat16, device=dev) * 0.1
    t1 = bench(lambda: torch.matmul(x, W_dq))
    t2 = bench(lambda: torch.matmul(x, W_dkv))
    tf = bench(lambda: torch.matmul(x, W_fused))

    # numerics: slice the fused result and compare with the separate GEMMs
    c_fused = torch.matmul(x, W_fused)
    c_sep = torch.cat([torch.matmul(x, W_dq), torch.matmul(x, W_dkv)], dim=1)
    torch_npu.npu.synchronize()
    exact = torch.equal(c_fused, c_sep)
    maxd = (c_fused.float() - c_sep.float()).abs().max().item()

    print("%6d %10.2f %10.2f %10.2f %10.2f %7.1f%%   bit-exact=%s max|d|=%.2e"
          % (m, t1, t2, t1 + t2, tf, (t1 + t2 - tf) / (t1 + t2) * 100, exact, maxd))

print()
print("note: 'gain' is the ceiling for route A on the down-projection phase; the kernel")
print("      additionally saves one phase boundary (sync + L1 refill).")
