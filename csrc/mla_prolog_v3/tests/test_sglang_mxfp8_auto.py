"""MXFP8 auto-dequant in the shim: both weight layouts, and an end-to-end run.

The second half reproduces the GLM serving path: SGLang-style preprocessing of the
fused qkv_a projection (transpose + split + NZ) plus fp8 q_b / kv_a with e8m0 scales,
and compares against a CPU golden computed from the dequantised weights.
"""
import json
import os
import sys

import torch
import torch_npu
from safetensors import safe_open

sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")
import sgl_kernel_npu  # noqa: F401,E402
from sgl_kernel_npu.mla_prolog_v3_compat import (  # noqa: E402
    dequant_mxfp8,
    register_custom_alias,
)

dev = "npu:0"
BLOCK = 32

# ---- 1) orientation unit checks (CPU)
torch.manual_seed(0)
out_f, in_f = 6, 64
w_ckpt = (torch.randn(out_f, in_f) * 0.1).to(torch.float8_e4m3fn)      # [out, in]
s_ckpt = torch.randint(120, 131, (out_f, in_f // BLOCK), dtype=torch.uint8)
ref = dequant_mxfp8(w_ckpt, s_ckpt)

w_abi = w_ckpt.t().contiguous()                                        # [in, out]
got_abi = dequant_mxfp8(w_abi, s_ckpt)
print("checkpoint layout dequant finite:", bool(torch.isfinite(ref).all()))
print("op-ABI layout matches checkpoint layout:", torch.equal(got_abi, ref.t().contiguous()))

# ---- 2) real checkpoint, serving-like path
MODEL = "/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
EPS = 1e-5
T = 8
LAYER = 2

index = json.load(open(os.path.join(MODEL, "quant_model_weights.safetensors.index.json")))
wm = index["weight_map"]


def load(name):
    with safe_open(os.path.join(MODEL, wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


p = "model.layers.%d.self_attn." % LAYER
q_a_fp8 = load(p + "q_a_proj.weight")            # [Hcq, He]
q_a_sc = load(p + "q_a_proj.weight_scale")       # [Hcq, He/32]
q_b_fp8 = load(p + "q_b_proj.weight")            # [N*(D+Dr), Hcq]
q_b_sc = load(p + "q_b_proj.weight_scale")
kv_fp8 = load(p + "kv_a_proj_with_mqa.weight")   # [Hckv+Dr, He]
kv_sc = load(p + "kv_a_proj_with_mqa.weight_scale")
gamma_cq = load(p + "q_a_layernorm.weight").to(torch.bfloat16)
gamma_ckv = load(p + "kv_a_layernorm.weight").to(torch.bfloat16)
kv_b = load(p + "kv_b_proj.weight").to(torch.bfloat16)
w_kc = kv_b.view(N, D + 256, Hckv)[:, :D, :].contiguous()

# SGLang stores the fused qkv_a projection as [Hcq+Hckv+Dr, He] and transposes it
fused = torch.cat([q_a_fp8, kv_fp8], dim=0)      # [1344, He]
fused_t = fused.t().contiguous()                 # [He, 1344]  (mlaprolog_preprocess_weight)
q_dq_abi = fused_t[:, :Hcq].contiguous()         # [He, Hcq]  -> needs dequant
kv_dq_abi = fused_t[:, Hcq:].contiguous()        # [He, 576]
q_b_abi = q_b_fp8.t().contiguous()               # [Hcq, 5120]

print("alias registered:", register_custom_alias())
nz = lambda t: torch_npu.npu_format_cast(t.to(dev), 29)

x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T)
inv = 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr))
freq = torch.outer(pos.float(), inv)
cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

# hand the shim the *fp8* weights plus their e8m0 scales, in the op ABI orientation
got = torch.ops.custom.npu_mla_prolog_v3(
    token_x=x,
    weight_dq=q_dq_abi.to(dev),
    weight_uq_qr=q_b_abi.to(dev),
    weight_uk=w_kc.to(dev),
    weight_dkv_kr=kv_dq_abi.to(dev),
    rmsnorm_gamma_cq=gamma_cq.to(dev),
    rmsnorm_gamma_ckv=gamma_ckv.to(dev),
    rope_sin=sin,
    rope_cos=cos,
    kv_cache=kvc,
    kr_cache=krc,
    cache_index=idx,
    dequant_scale_w_dq=q_a_sc.to(dev),
    dequant_scale_w_uq_qr=q_b_sc.to(dev),
    dequant_scale_w_dkv_kr=kv_sc.to(dev),
    rmsnorm_epsilon_cq=EPS,
    rmsnorm_epsilon_ckv=EPS,
    cache_mode="PA_BSND",
    query_norm_flag=True,
    weight_quant_mode=3,
)
torch_npu.npu.synchronize()

# golden on the dequantised weights
qa = dequant_mxfp8(q_a_fp8, q_a_sc).t().float()
qb = dequant_mxfp8(q_b_fp8, q_b_sc).t().float()
kva = dequant_mxfp8(kv_fp8, kv_sc).t().float()


def rope(v, c, s):
    half = v.shape[-1] // 2
    xe, xo = v[..., 0::2], v[..., 1::2]
    out = torch.empty_like(v)
    out[..., :half] = xe * c[..., :half] - xo * s[..., :half]
    out[..., half:] = xe * s[..., half:] + xo * c[..., half:]
    return out


xf = x.cpu().float()
cq = xf @ qa
cq = cq / torch.sqrt(cq.pow(2).mean(-1, keepdim=True) + EPS) * gamma_cq.cpu().float()
wuqqr = qb.view(Hcq, N, D + Dr)
q_gold = torch.empty(T, N, Hckv)
qr_gold = torch.empty(T, N, Dr)
for h in range(N):
    q_gold[:, h, :] = (cq @ wuqqr[:, h, :D]) @ w_kc.cpu().float()[h]
    qr_gold[:, h, :] = rope(cq @ wuqqr[:, h, D:], cos.cpu().float(), sin.cpu().float())
ckv = xf @ kva[:, :Hckv]
ckv = ckv / torch.sqrt(ckv.pow(2).mean(-1, keepdim=True) + EPS) * gamma_ckv.cpu().float()
kr = rope(xf @ kva[:, Hckv:], cos.cpu().float(), sin.cpu().float())
kvc_gold = kvc.cpu().float().view(-1, Hckv)[idx.cpu()]
krc_gold = krc.cpu().float().view(-1, Dr)[idx.cpu()]


def cmp(name, a, b):
    a, b = a.cpu().float(), b.cpu().float()
    cs = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0).item()
    print("%-11s cosine=%.7f  max|d|=%.5f" % (name, cs, (a - b).abs().max().item()))


print("-- MXFP8 weights handed to the shim, mode inferred to 0:")
cmp("query", got[0], q_gold)
cmp("query_rope", got[1], qr_gold)
cmp("query_norm", got[3], cq)
cmp("kv_cache", kvc_gold, ckv)
cmp("kr_cache", krc_gold, kr)
