"""Reproducibility probe for the MXFP8 query-branch explosion.

Runs the GLM geometry twice in a row (same process) and also a DeepSeek case in
between, printing the query magnitude each time.  Distinguishes a
geometry-dependent bug from run-order/state-dependent corruption.
"""
import sys

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
T = 4


def e8m0(n):
    return torch.full(n, 127, dtype=torch.uint8, device=dev)


def run(tag, He, Hcq, Hckv, D, Dr, N):
    torch.manual_seed(0)
    x = (torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1).to(torch.float8_e4m3fn)
    Wdq = torch_npu.npu_format_cast((torch.randn(He, Hcq, device=dev) * 0.02).to(torch.float8_e4m3fn), 29)
    Wuqqr = torch_npu.npu_format_cast((torch.randn(Hcq, N * (D + Dr), device=dev) * 0.02).to(torch.float8_e4m3fn), 29)
    Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
    Wdkv = torch_npu.npu_format_cast((torch.randn(He, Hckv + Dr, device=dev) * 0.02).to(torch.float8_e4m3fn), 29)
    gq = torch.ones(Hcq, dtype=torch.bfloat16, device=dev)
    gkv = torch.ones(Hckv, dtype=torch.bfloat16, device=dev)
    sin = torch.zeros(T, Dr, dtype=torch.bfloat16, device=dev)
    cos = torch.ones(T, Dr, dtype=torch.bfloat16, device=dev)
    kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
    krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
    idx = torch.arange(T, dtype=torch.int64, device=dev)
    q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(
        x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
        cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
        weight_quant_mode=3, kv_cache_quant_mode=0,
        dequant_scale_x=e8m0((T, He // 32)),
        dequant_scale_w_dq=e8m0((Hcq, He // 32)),
        dequant_scale_w_uq_qr=e8m0((N * (D + Dr), Hcq // 32)),
        dequant_scale_w_dkv_kr=e8m0((Hckv + Dr, He // 32)))
    torch_npu.npu.synchronize()
    qf = q.float()
    print("%-28s |q|max=%.3e |q|mean=%.3e inf=%d/%d"
          % (tag, float(qf.abs().max()), float(qf.abs().mean()),
             int(torch.isinf(qf).sum()), qf.numel()), flush=True)


order = sys.argv[1] if len(sys.argv) > 1 else "glm-first"
if order == "glm-first":
    run("GLM #1", 2048, 768, 512, 192, 64, 20)
    run("GLM #2", 2048, 768, 512, 192, 64, 20)
    run("DS", 7168, 1536, 512, 128, 64, 16)
    run("GLM #3", 2048, 768, 512, 192, 64, 20)
else:
    run("DS #1", 7168, 1536, 512, 128, 64, 16)
    run("GLM #1", 2048, 768, 512, 192, 64, 20)
