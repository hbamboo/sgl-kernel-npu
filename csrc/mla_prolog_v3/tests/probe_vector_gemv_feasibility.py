"""Feasibility probe for a vector-only decode path: how fast can the memory system do the
shape of work a vector GEMV needs?

For c_Q = x @ W_dq (He=2048, Hcq=768) a vector implementation streams the 3.1 MB weight once
and does a multiply+reduce per output row.  `(Wt * x).sum(1)` on the transposed layout is
exactly that pattern, so its time is a reasonable lower bound for a hand-written vector GEMV -
and it is compared with the cost of the same projection inside the current cube kernel.
"""
import statistics
import time

import torch
import torch_npu

dev = "npu:0"
He, Hcq, Hckv, HcqKr = 2048, 768, 576, 576
Dr = 64
torch.manual_seed(0)


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


x = torch.randn(He, dtype=torch.bfloat16, device=dev) * 0.1

def steady(fn, calls=47, rounds=10):
    """Per-call time when the queue stays full (as in serving)."""
    for _ in range(5):
        fn()
    torch_npu.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for _ in range(calls):
            fn()
    torch_npu.npu.synchronize()
    return (time.perf_counter() - t0) / (rounds * calls) * 1e6


for rows, name in ((Hcq, "c_Q   [768, 2048]"), (HcqKr, "c_KV  [576, 2048]")):
    Wt = torch.randn(rows, He, dtype=torch.bfloat16, device=dev)          # transposed layout
    mb = rows * He * 2 / 2 ** 20
    for label, fn in (("mul+reduce(row)", lambda: (Wt * x).sum(dim=1)),
                      ("torch.mv", lambda: torch.mv(Wt, x))):
        t = steady(fn)
        print("%-18s %6.2f MB  %-16s %6.2f us/call -> %5.0f GB/s (latency %.1f us)"
              % (name, mb, label, t, mb / 1024 / (t * 1e-6), bench(fn, iters=30)))
    del Wt
    torch_npu.npu.empty_cache()

# reference: what the cube path spends for the same projection (mm1 phase share is ~4-5 us)
print()
print("current cube kernel spends ~4-5 us per phase at decode (total 18.2 us for 4 phases)")
