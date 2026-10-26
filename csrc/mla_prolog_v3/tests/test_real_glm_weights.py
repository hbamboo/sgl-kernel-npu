"""End-to-end check with real GLM-4.7-Flash MXFP8 weights (layer 2).

Dequantises the msmodelslim MXFP8 weights (float8_e4m3 * per-32 e8m0 scale) to
bf16, runs MlaPrologV3 with weight_quant_mode=0 and compares against a CPU
golden computed from the same dequantised weights.
"""
import json
import math
import os

import torch
import torch_npu
from safetensors import safe_open

MODEL = "/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8"
LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
LAYER = 2
T = 8
EPS = 1e-5
torch.manual_seed(0)

index = json.load(open(os.path.join(MODEL, "quant_model_weights.safetensors.index.json")))
wm = index["weight_map"]


def load(name):
    path = os.path.join(MODEL, wm[name])
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def e8m0_to_float(u8):
    # value = 2^(e - 127); e == 0 is the zero/subnormal case (treat as 0)
    e = u8.to(torch.int32)
    val = torch.pow(2.0, (e - 127).float())
    return torch.where(e == 0, torch.zeros_like(val), val)


def dequant(w_fp8, scale_u8):
    """w: [out, in] float8_e4m3, scale: [out, in/32] uint8 (e8m0)."""
    w = w_fp8.to(torch.float32)
    s = e8m0_to_float(scale_u8)              # (out, in/32)
    s = s.repeat_interleave(32, dim=1)[:, : w.shape[1]]
    return (w * s).to(torch.bfloat16)


p = "model.layers.%d.self_attn." % LAYER
# checkpoint stores [out, in]; the op wants [in, out] (k, n)
q_a = dequant(load(p + "q_a_proj.weight"), load(p + "q_a_proj.weight_scale")).t().contiguous()
q_b = dequant(load(p + "q_b_proj.weight"), load(p + "q_b_proj.weight_scale")).t().contiguous()
kv_a = dequant(load(p + "kv_a_proj_with_mqa.weight"), load(p + "kv_a_proj_with_mqa.weight_scale")).t().contiguous()
gamma_cq = load(p + "q_a_layernorm.weight").to(torch.bfloat16)
gamma_ckv = load(p + "kv_a_layernorm.weight").to(torch.bfloat16)
kv_b = load(p + "kv_b_proj.weight").to(torch.bfloat16)      # unquantised
w_kc = kv_b.view(N, D + 256, Hckv)[:, :D, :].contiguous()   # (N, D, Hckv)

print("shapes:", tuple(q_a.shape), tuple(q_b.shape), tuple(kv_a.shape), tuple(kv_b.shape), tuple(w_kc.shape))
print("weight |.|mean: q_a=%.5f q_b=%.5f kv_a=%.5f"
      % (q_a.float().abs().mean(), q_b.float().abs().mean(), kv_a.float().abs().mean()))

x = (torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1)
pos = torch.arange(T)
inv = 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr))
freq = torch.outer(pos.float(), inv)
cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

nz = lambda t: torch_npu.npu_format_cast(t.clone().to(dev), 29)

# exactly the kwargs SGLang's forward_mlaprolog builds
mlaprolog_args = dict(
    token_x=x,
    weight_dq=nz(q_a),
    weight_uq_qr=nz(q_b),
    weight_uk=w_kc.to(dev),
    weight_dkv_kr=nz(kv_a),
    rmsnorm_gamma_cq=gamma_cq.to(dev),
    rmsnorm_gamma_ckv=gamma_ckv.to(dev),
    rope_sin=sin,
    rope_cos=cos,
    kv_cache=kvc,
    kr_cache=krc,
    cache_index=idx,
    cache_mode="PA_BSND",
    query_norm_flag=True,
    weight_quant_mode=0,
)

if os.environ.get("SGLANG_ALIAS"):
    import sys

    sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")
    from sgl_kernel_npu.mla_prolog_v3_compat import register_custom_alias

    print("registered custom::npu_mla_prolog_v3:", register_custom_alias())
    q, qr, dqn, qn, dqnorm = torch.ops.custom.npu_mla_prolog_v3(**mlaprolog_args)
else:
    q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(**mlaprolog_args)
torch_npu.npu.synchronize()


def rope(v, c, s):
    half = v.shape[-1] // 2
    xe, xo = v[..., 0::2], v[..., 1::2]
    out = torch.empty_like(v)
    out[..., :half] = xe * c[..., :half] - xo * s[..., :half]
    out[..., half:] = xe * s[..., half:] + xo * c[..., half:]
    return out


xf = x.cpu().float()
cq = xf @ q_a.float()
cq = cq / torch.sqrt(cq.pow(2).mean(-1, keepdim=True) + EPS) * gamma_cq.cpu().float()
wuqqr = q_b.cpu().float().view(Hcq, N, D + Dr)
q_gold = torch.empty(T, N, Hckv)
qr_gold = torch.empty(T, N, Dr)
for h in range(N):
    q_gold[:, h, :] = (cq @ wuqqr[:, h, :D]) @ w_kc.cpu().float()[h]
    qr_gold[:, h, :] = rope(cq @ wuqqr[:, h, D:], cos.cpu().float(), sin.cpu().float())
ckv = xf @ kv_a.cpu().float()[:, :Hckv]
ckv = ckv / torch.sqrt(ckv.pow(2).mean(-1, keepdim=True) + EPS) * gamma_ckv.cpu().float()
kr = rope(xf @ kv_a.cpu().float()[:, Hckv:], cos.cpu().float(), sin.cpu().float())
kvc_gold = kvc.cpu().float().view(-1, Hckv)[idx.cpu()]
krc_gold = krc.cpu().float().view(-1, Dr)[idx.cpu()]


def cmp(name, got, ref):
    got = got.cpu().float()
    ref = ref.cpu().float()
    cs = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), 0).item()
    print("%-12s cosine=%.7f  max|d|=%.5f  ref|.|mean=%.5f"
          % (name, cs, (got - ref).abs().max().item(), ref.abs().mean().item()))


cmp("query", q, q_gold)
cmp("query_rope", qr, qr_gold)
cmp("query_norm", qn, cq)
cmp("kv_cache", kvc_gold, ckv)
cmp("kr_cache", krc_gold, kr)
