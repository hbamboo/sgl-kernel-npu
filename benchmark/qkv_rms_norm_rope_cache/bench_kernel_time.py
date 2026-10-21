"""Split host overhead vs device kernel time for the two implementations.

Exports a torch_npu chrome trace and aggregates device-side events per call, so the two
kernels can be compared on device time regardless of Python/launch overhead.
"""

import json
import tempfile
import time
from collections import defaultdict

import torch
import torch_npu  # noqa: F401
from torch_npu.profiler import ProfilerActivity, profile

import sgl_kernel_npu  # noqa: F401
from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope

D = 128
T, NQ, NK, NV, BS = 34460, 32, 8, 8, 128
ITERS = 10


def make_inputs():
    g = torch.Generator(device="cpu").manual_seed(0)
    qkv = torch.randn(T, (NQ + NK + NV) * D, generator=g).to(torch.bfloat16).npu()
    qg = torch.randn(D, generator=g).to(torch.bfloat16).npu()
    kg = torch.randn(D, generator=g).to(torch.bfloat16).npu()
    cos = torch.randn(T, 1, 1, D, generator=g).clamp(-1, 1).to(torch.bfloat16).npu()
    sin = torch.randn(T, 1, 1, D, generator=g).clamp(-1, 1).to(torch.bfloat16).npu()
    bn = (T + BS - 1) // BS
    idx = torch.arange(T, dtype=torch.int64, device="npu")
    q = torch.zeros(T, NQ * D, dtype=torch.bfloat16, device="npu")
    kc = torch.zeros(bn, NK * D // 16, BS, 16, dtype=torch.bfloat16, device="npu")
    vc = torch.zeros(bn, NV * D // 16, BS, 16, dtype=torch.bfloat16, device="npu")
    return qkv, qg, kg, cos, sin, idx, q, kc, vc


def host_us(fn, iters=20):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) / iters * 1e6, (t2 - t0) / iters * 1e6


def trace_of(fn, iters=ITERS):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    path = tempfile.mktemp(suffix=".json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
    prof.export_chrome_trace(path)
    with open(path) as f:
        return json.load(f)


def summarize(trace, iters, tag):
    agg = defaultdict(lambda: [0, 0.0])
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        key = (e.get("cat", ""), e.get("name", ""))
        agg[key][0] += 1
        agg[key][1] += float(e["dur"])
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])[:10]
    print(f"\n== {tag} -- top device/host events (per call, iters={iters}) ==")
    for (cat, name), (cnt, dur) in rows:
        print(f"  {dur/iters:9.1f} us x{cnt:<4d} [{cat:12s}] {name[:64]}")


def main():
    qkv, qg, kg, cos, sin, idx, q, kc, vc = make_inputs()
    cos2, sin2 = cos.view(T, D), sin.view(T, D)

    def run_ascendc():
        torch.ops.npu.qkv_rms_norm_rope_cache(
            qkv, qg, kg, cos2, sin2, idx, q, kc, vc, NQ, NK, NV, 1e-6
        )

    def run_triton():
        split_qkv_rmsnorm_rope(
            qkv, sin, cos, NQ * D, NK * D, D, eps=1e-6, q_weight=qg, k_weight=kg
        )

    for name, fn in (("AscendC", run_ascendc), ("Triton", run_triton)):
        enq, total = host_us(fn)
        print(f"\n### {name}: host enqueue {enq:.1f} us/call, host+device {total:.1f} us/call")
        summarize(trace_of(fn), ITERS, name)


if __name__ == "__main__":
    main()
