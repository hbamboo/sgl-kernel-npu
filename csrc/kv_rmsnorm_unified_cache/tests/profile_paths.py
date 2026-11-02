"""Kernel-sequence profiling for old vs candidate at representative cache sizes.

Run separately for each arm so the CSV attribution is unambiguous.  Evidence:
old path = 2 x aclnnContiguous_Slice + KvRmsNormRopeCache + 2 x aclnnViewCopy;
candidate = only sgl_kv_rmsnorm_unified_cache.  Both arm windows are warmed up
before recording, so only steady-state launches are captured.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import torch
import torch_npu
import torch_npu.profiler as P

LIB = os.environ["KV_RMSNORM_UNIFIED_LIB"]
torch.ops.load_library(LIB)
torch.npu.set_device(0)
torch.manual_seed(7)
EPS = 1e-5


def state(T, pages, dtype):
    g = torch.Generator().manual_seed(T + pages)
    kv = torch.randn(T, 576, generator=g).to(dtype).npu()
    gamma = (torch.randn(512, generator=g) * 0.5 + 1.0).to(dtype).npu()
    cos = torch.randn(T, 1, 1, 64, generator=g).to(dtype).npu()
    sin = torch.randn(T, 1, 1, 64, generator=g).to(dtype).npu()
    idx = torch.randperm(pages * 128, generator=g)[:T].to(torch.int64).npu()
    cache = torch.full((pages, 128, 1, 576), -7.0, dtype=dtype, device="npu")
    return kv, gamma, cos, sin, idx, cache


def run_arm(arm, T, pages, outdir, dtype=torch.bfloat16):
    kv, gamma, cos, sin, idx, cache = state(T, pages, dtype)
    iters = 3

    if arm == "old":
        def call():
            return torch_npu.npu_kv_rmsnorm_rope_cache(
                kv.view(T, 1, 1, 576), gamma, cos, sin, idx, cache[..., 512:], cache[..., :512],
                k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                epsilon=EPS, cache_mode="PA_BNSD", is_output_kv=True)
    else:
        def call():
            ko = torch.empty(T, 1, 1, 64, dtype=dtype, device="npu")
            vo = torch.empty(T, 1, 1, 512, dtype=dtype, device="npu")
            torch.ops.npu.kv_rmsnorm_unified_cache(
                kv.view(T, 1, 1, 576), gamma, cos, sin, idx, cache, ko, vo, EPS)
            return ko, vo

    for _ in range(5):
        call()
    torch.npu.synchronize()
    trace_root = f"/tmp/kvprof_{arm}_{T}_{pages}"
    shutil.rmtree(trace_root, ignore_errors=True)
    exp = P._ExperimentalConfig(export_type=P.ExportType.Text)
    with P.profile(activities=[P.ProfilerActivity.CPU, P.ProfilerActivity.NPU],
                   experimental_config=exp,
                   on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_root)):
        for _ in range(iters):
            call()
        torch.npu.synchronize()
    csvs = glob.glob(trace_root + "/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
    assert csvs, f"no kernel_details.csv under {trace_root}"
    dst = os.path.join(outdir, f"profile_{arm}_T{T}_P{pages}")
    os.makedirs(dst, exist_ok=True)
    shutil.copy(csvs[0], os.path.join(dst, "kernel_details.csv"))
    opcsv = glob.glob(trace_root + "/*/ASCEND_PROFILER_OUTPUT/operator_details.csv")
    if opcsv:
        shutil.copy(opcsv[0], os.path.join(dst, "operator_details.csv"))

    rows = []
    with open(csvs[0]) as f:
        hdr = f.readline().strip().split(",")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            rec = dict(zip(hdr, parts))
            rows.append({"name": rec.get("Name", ""), "duration_us": float(rec.get("Duration(us)", "0")),
                         "block_num": rec.get("Block Num", ""), "start_us": rec.get("Start Time(us)", "")})
    classes = {"slice": 0, "viewcopy": 0, "kv_rmsnorm_cache": 0, "unified": 0}
    for r in rows:
        n = r["name"].lower()
        if "slice" in n:
            classes["slice"] += 1
        if "viewcopy" in n:
            classes["viewcopy"] += 1
        if n.startswith("kvrmsnormropecache"):
            classes["kv_rmsnorm_cache"] += 1
        if n.startswith("sgl_kv_rmsnorm_unified_cache"):
            classes["unified"] += 1
    info = {
        "arm": arm, "T": T, "pages": pages, "dtype": str(dtype),
        "cache_shape": list(cache.shape), "cache_bytes": cache.numel() * cache.element_size(),
        "kv_input_shape": list(kv.shape),
        "kernels": rows, "kernel_class_counts_per_window": classes,
        "iters": iters,
    }
    with open(os.path.join(dst, "summary.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "kernels"}, indent=1))
    for r in rows[:16]:
        print("  ", r["name"], r["duration_us"], "us block", r["block_num"])
    return info


if __name__ == "__main__":
    outdir = sys.argv[1]
    T = int(sys.argv[2]); pages = int(sys.argv[3])
    os.makedirs(outdir, exist_ok=True)
    a = run_arm("old", T, pages, outdir)
    b = run_arm("new", T, pages, outdir)
    print("SUMMARY_JSON", json.dumps({"old": a["kernel_class_counts_per_window"],
                                      "new": b["kernel_class_counts_per_window"]}))
