"""Full-path event timing: installed npu_kv_rmsnorm_rope_cache on real unified-576
512/64 slices vs torch.ops.npu.kv_rmsnorm_unified_cache.

Both arms include input preparation, output allocation, launch and cache update.
Cache tensors are allocated/cloned once outside the measured intervals. Arms are
alternated each repetition; warmup + multiple rounds reported as median/mean/p95.
"""
import json
import os
import statistics
import sys

import torch
import torch_npu

LIB = os.environ.get("KV_RMSNORM_UNIFIED_LIB")
torch.ops.load_library(LIB)
torch.npu.set_device(0)
torch.manual_seed(1234)

DTYPE = torch.bfloat16
EPS = 1e-5
WARMUP = 20


def make_state(T, pages, strided=False, dtype=DTYPE):
    g = torch.Generator(device="cpu").manual_seed(T * 131 + pages)
    gamma = (torch.randn(512, generator=g) * 0.5 + 1.0).to(dtype).npu()
    cos = torch.randn(T, 1, 1, 64, generator=g).to(dtype).npu()
    sin = torch.randn(T, 1, 1, 64, generator=g).to(dtype).npu()
    idx = torch.randperm(pages * 128, generator=g)[:T].to(torch.int64).npu()
    if strided:
        qkv = torch.randn(T, 1344, generator=g).to(dtype).npu()
        kv2d = qkv[:, 768:]
    else:
        kv2d = torch.randn(T, 576, generator=g).to(dtype).npu()
    return kv2d, gamma, cos, sin, idx


def baseline_call(kv2d, gamma, cos, sin, idx, cache, pages):
    kb = cache[..., :512]
    vb = cache[..., 512:]
    T = kv2d.shape[0]
    r = torch_npu.npu_kv_rmsnorm_rope_cache(
        kv2d.view(T, 1, 1, 576), gamma, cos, sin, idx, vb, kb,
        k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
        epsilon=EPS, cache_mode="PA_BNSD", is_output_kv=True,
    )
    return r


def candidate_call(kv2d, gamma, cos, sin, idx, cache):
    T = kv2d.shape[0]
    dt = kv2d.dtype
    ko = torch.empty(T, 1, 1, 64, dtype=dt, device="npu")
    vo = torch.empty(T, 1, 1, 512, dtype=dt, device="npu")
    torch.ops.npu.kv_rmsnorm_unified_cache(kv2d.view(T, 1, 1, 576), gamma, cos, sin, idx, cache, ko, vo, EPS)
    return ko, vo


def time_once(fn, ev0, ev1):
    ev0.record()
    out = fn()
    ev1.record()
    ev1.synchronize()
    # torch.npu.Event.elapsed_time returns milliseconds.
    return ev0.elapsed_time(ev1) * 1000.0, out


def bench(T, pages, reps=50, strided=False):
    kv2d, gamma, cos, sin, idx = make_state(T, pages, strided=strided)
    cache_a = torch.full((pages, 128, 1, 576), -7.0, dtype=DTYPE, device="npu")
    cache_b = cache_a.clone()
    ka = lambda: baseline_call(kv2d, gamma, cos, sin, idx, cache_a, pages)
    kb = lambda: candidate_call(kv2d, gamma, cos, sin, idx, cache_b)
    for _ in range(WARMUP):
        ka(); kb()
    torch.npu.synchronize()
    ev0, ev1 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    a, b = [], []
    # Outputs stay alive through each timed interval (time_once returns them after
    # the end-event sync) but are dropped before the next repetition, mirroring the
    # framework which discards the old op's cache returns immediately.
    for r in range(reps):
        if r % 2 == 0:
            ta, oa = time_once(ka, ev0, ev1); tb, ob = time_once(kb, ev0, ev1)
        else:
            tb, ob = time_once(kb, ev0, ev1); ta, oa = time_once(ka, ev0, ev1)
        a.append(ta); b.append(tb); del oa, ob
    torch.npu.synchronize()

    def stats(x):
        xs = sorted(x)
        return dict(median=statistics.median(xs), mean=statistics.mean(xs),
                    p95=xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))],
                    stdev=statistics.pstdev(xs), min=xs[0], max=xs[-1])

    sa, sb = stats(a), stats(b)
    return dict(T=T, pages=pages, strided=strided,
                baseline_us=sa, candidate_us=sb,
                net_saved_us=sa["median"] - sb["median"],
                ratio=sa["median"] / sb["median"],
                baseline_samples_us=a, candidate_samples_us=b)


def main(out_path):
    cases = [
        dict(T=3787, pages=7194, reps=50),
        dict(T=15148, pages=7194, reps=40),
        dict(T=3787, pages=128, reps=50),
        dict(T=3787, pages=15148, reps=50),
        dict(T=3787, pages=7194, reps=50, strided=True),
        dict(T=15148, pages=15148, reps=40),
    ]
    results = []
    for c in cases:
        r = bench(**c)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if not k.endswith("samples_us")}, indent=1))
        print("  samples baseline", [round(x, 1) for x in r["baseline_samples_us"][:8]])
        print("  samples candidate", [round(x, 1) for x in r["candidate_samples_us"][:8]])
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print("WROTE", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/kv_rmsnorm_bench.json")
