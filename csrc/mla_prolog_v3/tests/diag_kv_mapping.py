"""Where do the KV cache rows actually land at T=192 (multi-step)? Find the mapping."""
import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
EPS = 1e-5
T = 192
torch.manual_seed(3)


def rmsnorm(v, gamma):
    return v / torch.sqrt(v.pow(2).mean(-1, keepdim=True) + EPS) * gamma


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
got = kvc.cpu().float().view(-1, Hckv)          # all slots
slots = got.shape[0]
print("slots=%d tokens=%d" % (slots, T))

# normalise for cosine matching
gn = got / got.norm(dim=1, keepdim=True).clamp_min(1e-6)
cn = ckv / ckv.norm(dim=1, keepdim=True).clamp_min(1e-6)
sim = gn @ cn.T                                  # (slots, T)
best = sim.argmax(dim=1)
bestv = sim.max(dim=1).values
print("slot -> best token match (first 12):", [(i, int(best[i]), round(float(bestv[i]), 3)) for i in range(12)])
print("token 0 expected value found at slot(s):", [int(i) for i in range(slots) if int(best[i]) == 0][:6])
print("slots with low match (<0.9):", int((bestv < 0.9).sum()), "/", slots)
nonzero = (got.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
print("nonzero slots:", len(nonzero), "first:", nonzero[:6], "last:", nonzero[-6:])
