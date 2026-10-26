"""Kernel-level timing for MlaPrologV3 via the torch_npu profiler (no schedule)."""
import glob
import os

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = int(os.environ.get("T", "1"))
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


def call():
    torch.ops.npu.mla_prolog_v3(
        x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
        cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
        weight_quant_mode=0, kv_cache_quant_mode=0)


for _ in range(20):
    call()
torch_npu.npu.synchronize()

exp = "/tmp/prof_mla2"
os.system("rm -rf " + exp)
with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
                                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(exp)):
    for _ in range(20):
        call()
    torch_npu.npu.synchronize()

files = glob.glob(os.path.join(exp, "*", "kernel_details.csv"))
print("csv:", files[:1])
if files:
    import csv

    durs = []
    with open(files[0]) as f:
        for row in csv.DictReader(f):
            name = row.get("Name", "")
            if "mla" in name.lower():
                durs.append(float(row.get("Duration(us)", row.get("Duration", 0))))
    if durs:
        durs.sort()
        print("mla_prolog_v3 kernel: n=%d median=%.2f us min=%.2f max=%.2f"
              % (len(durs), durs[len(durs) // 2], durs[0], durs[-1]))
    else:
        names = set()
        with open(files[0]) as f:
            for row in csv.DictReader(f):
                names.add(row.get("Name", ""))
        print("no mla kernel rows; kernels seen:", list(names)[:10])
