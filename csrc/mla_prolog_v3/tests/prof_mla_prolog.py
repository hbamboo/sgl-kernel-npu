"""Profile the ported MlaPrologV3: split kernel time from host overhead."""
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
blocks = (T + 127) // 128 + 1
kvc = torch.zeros(blocks, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(blocks, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)


def call():
    torch.ops.npu.mla_prolog_v3(
        x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
        cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
        weight_quant_mode=0, kv_cache_quant_mode=0)


for _ in range(10):
    call()
torch_npu.npu.synchronize()

exp = "/tmp/prof_mla"
activities = [torch_npu.profiler.ProfilerActivity.NPU]
with torch_npu.profiler.profile(activities=activities, schedule=torch_npu.profiler.schedule(wait=0, warmup=2, active=5, repeat=1),
                                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(exp),
                                experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1)):
    for _ in range(10):
        call()
        torch_npu.npu.synchronize()

files = glob.glob(os.path.join(exp, "*", "kernel_details.csv"))
print("kernel_details:", files)
if files:
    import csv

    tot = 0.0
    rows = []
    with open(files[0]) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    for r in rows[:6]:
        print({k: r[k] for k in list(r)[:8]})
    names = {}
    for r in rows:
        names[r.get("Name", "?")] = names.get(r.get("Name", "?"), 0) + 1
    print("kernel counts:", names)
