"""MXFP8 sanity sweep: geometry x query_norm_flag.

Prints whether the query branch is numerically sane (|q| magnitude) for:
  * GLM-4.7-Flash geometry (He=2048, Hcq=768, D=192, N=20)
  * DeepSeek geometry     (He=7168, Hcq=1536, D=128, N=16)
with query_norm_flag on/off.
"""
import itertools

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
T = 4
torch.manual_seed(0)


def e8m0(n):
    return torch.full(n, 127, dtype=torch.uint8, device=dev)


def run(tag, He, Hcq, Hckv, D, Dr, N, qnf):
    tok = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
    x = tok.to(torch.float8_e4m3fn)
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
    try:
        q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(
            x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
            cache_index=idx, cache_mode="PA_BSND", query_norm_flag=qnf,
            weight_quant_mode=3, kv_cache_quant_mode=0,
            dequant_scale_x=e8m0((T, He // 32)),
            dequant_scale_w_dq=e8m0((Hcq, He // 32)),
            dequant_scale_w_uq_qr=e8m0((N * (D + Dr), Hcq // 32)),
            dequant_scale_w_dkv_kr=e8m0((Hckv + Dr, He // 32)))
        torch_npu.npu.synchronize()
        qf = q.float()
        print("%-34s qnf=%-5s |q|max=%.3e  |q|mean=%.3e  finite=%.2f  kv|.|mean=%.3e"
              % (tag, qnf, float(qf.abs().max()), float(qf.abs().mean()),
                 float(torch.isfinite(qf).float().mean()), float(kvc.float().abs().mean())))
    except Exception as exc:  # noqa: BLE001
        print("%-34s qnf=%-5s FAILED: %s" % (tag, qnf, str(exc).splitlines()[0][:120]))


for hcq in (768, 1024, 1152, 1536):
    run("He=2048 Hcq=%d D192 N20" % hcq, 2048, hcq, 512, 192, 64, 20, True)
