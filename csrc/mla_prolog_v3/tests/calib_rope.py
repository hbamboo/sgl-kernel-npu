"""Calibrate the RoPE convention used by MlaPrologV3 (interleave-half VF).

Builds W_QR as an identity selector so that q_rope[head 0] is exactly rope(c_Q[:64])
and compares the kernel output against several candidate formulas.
"""
import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 2
torch.manual_seed(7)

x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
Wdq = torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02
# W_UQ_QR: columns [0:N*D] = W_UQ, columns [N*D:] = W_QR (head-major, Dr per head)
Wuqqr = torch.zeros(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev)
for h in range(N):  # keep W_UQ = 0 so q_nope is zero, only rope matters
    pass
# W_QR for head 0: identity on c_Q[0:64]; other heads zero
# try interleaved per-head layout: head 0 rope block starts at offset D
Wuqqr[:Dr, D : D + Dr] = torch.eye(Dr, dtype=torch.bfloat16, device=dev)
Wuk = torch.zeros(N, D, Hckv, dtype=torch.bfloat16, device=dev)
Wdkv = torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02
gq = torch.ones(Hcq, dtype=torch.bfloat16, device=dev)
gkv = torch.ones(Hckv, dtype=torch.bfloat16, device=dev)

pos = torch.arange(T)
inv = 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr))
freq = torch.outer(pos.float(), inv)           # (T, Dr/2)
cos = freq.cos().repeat_interleave(2, dim=1)   # (T, Dr)
sin = freq.sin().repeat_interleave(2, dim=1)
cos = cos.to(torch.bfloat16).to(dev)
sin = sin.to(torch.bfloat16).to(dev)

kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

Wdq_nz = torch_npu.npu_format_cast(Wdq.clone(), 29)
Wuqqr_nz = torch_npu.npu_format_cast(Wuqqr.clone(), 29)
Wdkv_nz = torch_npu.npu_format_cast(Wdkv.clone(), 29)

q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(
    x, Wdq_nz, Wuqqr_nz, Wuk, Wdkv_nz, gq, gkv, sin, cos, kvc, krc,
    cache_index=idx, cache_mode="PA_BSND", query_norm_flag=False,
    weight_quant_mode=0, kv_cache_quant_mode=0)
torch_npu.npu.synchronize()

# golden c_Q (verified earlier against the kernel)
cq = (x.float() @ Wdq.float())
cq = cq / torch.sqrt(cq.pow(2).mean(-1, keepdim=True) + 1e-5) * 1.0
# the rope input for head 0 is c_Q[:64] (identity selector)
xin = cq[:, :Dr]                        # (T, Dr)
print("q_rope per-head norms:", [round(float(qr[0, h].float().abs().mean()), 5) for h in range(4)])
got = qr[:, 0, :].float()               # (T, Dr)
c = cos.float()
s = sin.float()


def cand_rotate_half(xx, cc, ss):
    x1, x2 = xx[:, : Dr // 2], xx[:, Dr // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return xx * cc + rot * ss


def cand_pairwise_scalar(xx, cc, ss):
    out = torch.empty_like(xx)
    out[:, 0::2] = xx[:, 0::2] * cc[:, 0::2] - xx[:, 1::2] * ss[:, 1::2]
    out[:, 1::2] = xx[:, 0::2] * ss[:, 0::2] + xx[:, 1::2] * cc[:, 1::2]
    return out


def cand_pairwise_half(xx, cc, ss):
    half = Dr // 2
    xe, xo = xx[:, 0::2], xx[:, 1::2]
    out = torch.empty_like(xx)
    out[:, 0::2] = xe * cc[:, :half] - xo * ss[:, :half]
    out[:, 1::2] = xe * ss[:, half:] + xo * cc[:, half:]
    return out


def cand_pairwise_half2(xx, cc, ss):
    half = Dr // 2
    xe, xo = xx[:, 0::2], xx[:, 1::2]
    out = torch.empty_like(xx)
    out[:, 0::2] = xe * cc[:, :half] + xo * ss[:, :half]
    out[:, 1::2] = xe * ss[:, half:] + xo * cc[:, half:]
    return out


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0).item()


print("rope input |.|mean =", xin.abs().mean().item(), " kernel out |.|mean =", got.abs().mean().item())
for name, fn in [("rotate_half", cand_rotate_half), ("pairwise_scalar", cand_pairwise_scalar),
                 ("pairwise_half(-,+)", cand_pairwise_half), ("pairwise_half(+,+)", cand_pairwise_half2)]:
    ref = fn(xin, c, s)
    print("%-20s cosine=%.6f maxdiff=%.4f" % (name, cos_sim(got, ref), (got - ref).abs().max().item()))

# also check the kr_cache (k_R = rope(x @ W_KR))
kr = krc.float().view(-1, Dr)[idx]
xkr = x.float() @ Wdkv.float()[:, Hckv:]
for name, fn in [("rotate_half", cand_rotate_half), ("pairwise_scalar", cand_pairwise_scalar),
                 ("pairwise_half(-,+)", cand_pairwise_half), ("pairwise_half(+,+)", cand_pairwise_half2)]:
    ref = fn(xkr, c, s)
    print("kr %-17s cosine=%.6f maxdiff=%.4f" % (name, cos_sim(kr, ref), (kr - ref).abs().max().item()))
