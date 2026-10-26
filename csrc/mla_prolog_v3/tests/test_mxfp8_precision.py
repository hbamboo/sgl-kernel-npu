"""MXFP8 numerics for GLM geometry: kernel vs CPU golden on dequantised weights.

fp8 e4m3 values times e8m0 (power-of-two) scales are exactly representable, so the
reference is simply the bf16/fp32 GEMM on the dequantised tensors.
"""
import os

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = int(os.environ.get("T", "8"))
EPS = 1e-5
torch.manual_seed(5)


def e8m0_const(shape, exp=127):
    return torch.full(shape, exp, dtype=torch.uint8, device=dev)


def deq_x(t_fp8, scale_u8):
    """activation: [rows, in] with scale [rows, in/32]."""
    v = t_fp8.float()
    s = torch.pow(2.0, scale_u8.to(torch.int32).float() - 127.0)
    return v * s.repeat_interleave(32, dim=-1)[..., : v.shape[-1]]


def deq_w(w_fp8, scale_u8):
    """op weight layout: [in, out] with scale [out, in/32]."""
    v = w_fp8.float()
    s = torch.pow(2.0, scale_u8.to(torch.int32).float() - 127.0)     # [out, in/32]
    s = s.t().repeat_interleave(32, dim=0)[: v.shape[0]]             # [in, out]
    return v * s


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., :half] = xe * cos[..., :half] - xo * sin[..., :half]
    out[..., half:] = xe * sin[..., half:] + xo * cos[..., half:]
    return out


x_bf = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
x = x_bf.to(torch.float8_e4m3fn)
Wdq = (torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02).to(torch.float8_e4m3fn)
Wuqqr = (torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02).to(torch.float8_e4m3fn)
Wdkv = (torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02).to(torch.float8_e4m3fn)
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
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

sx = e8m0_const((T, He // 32))
sdq = e8m0_const((Hcq, He // 32))
suq = e8m0_const((N * (D + Dr), Hcq // 32))
sdkv = e8m0_const((Hckv + Dr, He // 32))

q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(
    x, torch_npu.npu_format_cast(Wdq.clone(), 29), torch_npu.npu_format_cast(Wuqqr.clone(), 29), Wuk,
    torch_npu.npu_format_cast(Wdkv.clone(), 29), gq, gkv, sin, cos, kvc, krc,
    cache_index=idx, cache_mode="PA_BSND", query_norm_flag=False,
    weight_quant_mode=3, kv_cache_quant_mode=0,
    dequant_scale_x=sx, dequant_scale_w_dq=sdq, dequant_scale_w_uq_qr=suq, dequant_scale_w_dkv_kr=sdkv)
torch_npu.npu.synchronize()

# golden on the dequantised tensors
xd = deq_x(x, sx).cpu()
dq = deq_w(Wdq, sdq).cpu()
uq = deq_w(Wuqqr, suq).cpu().view(Hcq, N, D + Dr)
dkv = deq_w(Wdkv, sdkv).cpu()
cq = xd @ dq
cq = cq / torch.sqrt(cq.pow(2).mean(-1, keepdim=True) + EPS) * gq.cpu().float()
q_gold = torch.empty(T, N, Hckv)
qr_gold = torch.empty(T, N, Dr)
for h in range(N):
    q_gold[:, h, :] = (cq @ uq[:, h, :D]) @ Wuk.cpu().float()[h]
    qr_gold[:, h, :] = rope(cq @ uq[:, h, D:], cos.cpu().float(), sin.cpu().float())
ckv = xd @ dkv[:, :Hckv]
ckv = ckv / torch.sqrt(ckv.pow(2).mean(-1, keepdim=True) + EPS) * gkv.cpu().float()
kr = rope(xd @ dkv[:, Hckv:], cos.cpu().float(), sin.cpu().float())
kvc_gold = kvc.cpu().float().view(-1, Hckv)[idx.cpu()]
krc_gold = krc.cpu().float().view(-1, Dr)[idx.cpu()]


def cmp(name, got, ref):
    got = got.cpu().float()
    ref = ref.cpu().float()
    cs = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), 0).item()
    print("%-12s cosine=%.7f  max|d|=%.5f  ref|.|mean=%.5f"
          % (name, cs, (got - ref).abs().max().item(), ref.abs().mean().item()))


cores = os.environ.get("MLA_PROLOG_V3_CORES", "auto")
print("T=%d cores=%s" % (T, cores))
cmp("query", q, q_gold)
cmp("query_rope", qr, qr_gold)
cmp("kv_cache", kvc_gold, ckv)
cmp("kr_cache", krc_gold, kr)
