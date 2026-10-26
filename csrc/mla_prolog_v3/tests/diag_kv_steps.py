"""Locate which tokens' KV cache entries are wrong at T=256 (stepBatch=64)."""
import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
EPS = 1e-5
torch.manual_seed(3)


def rmsnorm(v, gamma):
    return v / torch.sqrt(v.pow(2).mean(-1, keepdim=True) + EPS) * gamma


def run(T):
    x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
    Wdq = torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02
    Wuqqr = torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02
    Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
    Wdkv = torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02
    gq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
    gkv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
    pos = torch.arange(T)
    inv = 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr))
    freq = torch.outer(pos.float(), inv)
    cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
    sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
    blocks = (T + 127) // 128 + 1
    kvc = torch.zeros(blocks, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
    krc = torch.zeros(blocks, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
    idx = torch.arange(T, dtype=torch.int64, device=dev)
    torch.ops.npu.mla_prolog_v3(
        x, torch_npu.npu_format_cast(Wdq.clone(), 29), torch_npu.npu_format_cast(Wuqqr.clone(), 29), Wuk,
        torch_npu.npu_format_cast(Wdkv.clone(), 29), gq, gkv, sin, cos, kvc, krc,
        cache_index=idx, cache_mode="PA_BSND", query_norm_flag=False,
        weight_quant_mode=0, kv_cache_quant_mode=0)
    torch_npu.npu.synchronize()

    ckv = (x.cpu().float() @ Wdkv.cpu().float()[:, :Hckv])
    ckv = rmsnorm(ckv, gkv.cpu().float())
    got = kvc.cpu().float().view(-1, Hckv)[:T]
    per_tok = (got - ckv).abs().mean(dim=1)
    bad = (per_tok > 0.05).nonzero().flatten().tolist()
    print("T=%d blocks=%d  mean err=%.5f  bad tokens=%d first_bad=%s"
          % (T, blocks, per_tok.mean().item(), len(bad), bad[:8]))
    if bad:
        print("   err by step (64):", [round(float(per_tok[i:i + 64].mean()), 4) for i in range(0, T, 64)])


import os

for t in (64, 128, 192, 256, 320):
    run(t)
