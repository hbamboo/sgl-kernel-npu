"""Full CPU golden vs kernel for MlaPrologV3 on GLM-4.7-Flash geometry (bf16 path).

Weight layout (verified by probe): W_UQ_QR columns are per-head interleaved,
  head h occupies columns [h*(D+Dr), (h+1)*(D+Dr)), first D = W_UQ, next Dr = W_QR.

RoPE (verified by probe, "interleave-half"):
  y[2i]     = x[2i]*cos[i]      - x[2i+1]*sin[i]
  y[2i+32]  = x[2i]*sin[i+32]   + x[2i+1]*cos[i+32]
"""
import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 8
EPS = 1e-5
torch.manual_seed(3)


def rope(x, cos, sin):
    """interleave-half rope: lower half = even*cos - odd*sin, upper half = even*sin + odd*cos."""
    half = x.shape[-1] // 2
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., :half] = xe * cos[..., :half] - xo * sin[..., :half]
    out[..., half:] = xe * sin[..., half:] + xo * cos[..., half:]
    return out


def rmsnorm(v, gamma, eps=EPS):
    return v / torch.sqrt(v.pow(2).mean(-1, keepdim=True) + eps) * gamma


x = (torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1)
Wdq = torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02
Wuqqr = torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
Wdkv = torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02
gq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
gkv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5

pos = torch.arange(T)
inv = 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr))
freq = torch.outer(pos.float(), inv)
cos_half = freq.cos()
sin_half = freq.sin()
cos = torch.cat([cos_half, cos_half], dim=1).to(torch.bfloat16).to(dev)  # duplicated halves (SGLang style)
sin = torch.cat([sin_half, sin_half], dim=1).to(torch.bfloat16).to(dev)

kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

Wdq_nz = torch_npu.npu_format_cast(Wdq.clone(), 29)
Wuqqr_nz = torch_npu.npu_format_cast(Wuqqr.clone(), 29)
Wdkv_nz = torch_npu.npu_format_cast(Wdkv.clone(), 29)

q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(
    x, Wdq_nz, Wuqqr_nz, Wuk, Wdkv_nz, gq, gkv, sin, cos, kvc, krc,
    cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
    weight_quant_mode=0, kv_cache_quant_mode=0)
torch_npu.npu.synchronize()

# ---------------- golden ----------------
xf = x.cpu().float()
Wdq_c, Wuqqr_c, Wuk_c, Wdkv_c = Wdq.cpu().float(), Wuqqr.cpu().float(), Wuk.cpu().float(), Wdkv.cpu().float()
gq_c, gkv_c = gq.cpu().float(), gkv.cpu().float()
cq = rmsnorm(xf @ Wdq_c, gq_c)                      # (T, Hcq)
wuqqr = Wuqqr_c.view(Hcq, N, D + Dr)                      # per-head blocks
q_gold = torch.empty(T, N, Hckv)
qr_gold = torch.empty(T, N, Dr)
for h in range(N):
    qc = cq @ wuqqr[:, h, :D]                                   # (T, D)
    q_gold[:, h, :] = qc @ Wuk_c[h]                             # (T, Hckv)
    qr_gold[:, h, :] = rope(cq @ wuqqr[:, h, D:], cos.cpu().float(), sin.cpu().float())

ckv = rmsnorm(xf @ Wdkv_c[:, :Hckv], gkv_c)
kr = rope(xf @ Wdkv_c[:, Hckv:], cos.cpu().float(), sin.cpu().float())

kvc_gold = kvc.cpu().float().view(-1, Hckv)[idx.cpu()]
krc_gold = krc.cpu().float().view(-1, Dr)[idx.cpu()]


def cmp(name, got, ref):
    got = got.cpu().float()
    ref = ref.cpu().float()
    cos_sim = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), 0).item()
    rel = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).mean().item()
    print("%-12s cosine=%.6f  max|d|=%.5f  mean rel=%.4f  ref|.|mean=%.4f"
          % (name, cos_sim, (got - ref).abs().max().item(), rel, ref.abs().mean().item()))


cmp("query", q, q_gold)
cmp("query_rope", qr, qr_gold)
cmp("query_norm", qn, cq)
cmp("kv_cache", kvc_gold, ckv)
cmp("kr_cache", krc_gold, kr)
print("dequant_scale_q_nope:", tuple(dqn.shape), " dequant_scale_q_norm:", tuple(dqnorm.shape))
