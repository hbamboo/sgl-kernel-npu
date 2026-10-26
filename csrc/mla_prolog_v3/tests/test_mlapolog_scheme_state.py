"""Check prepare_mlapolog_bf16_weights against the raw module weights (MXFP8 shapes).

Simulates the *scheme output* (fused weight already [in, out], scale 'weight_scale_inv'
with the [in/64, out, 2] layout) and verifies the helper produces projections that the
kernel reproduces the golden with - i.e. it does not depend on SGLang's preprocess.
"""
import json
import os
import sys
import types

import torch
import torch_npu
from safetensors import safe_open

sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")
import sgl_kernel_npu  # noqa: F401,E402
from sgl_kernel_npu.mla_prolog_v3_compat import (  # noqa: E402
    dequant_mxfp8,
    prepare_mlapolog_bf16_weights,
    register_custom_alias,
)

MODEL = "/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8"
dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
EPS, T, LAYER = 1e-5, 8, 2

index = json.load(open(os.path.join(MODEL, "quant_model_weights.safetensors.index.json")))
wm = index["weight_map"]


def load(name):
    with safe_open(os.path.join(MODEL, wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


p = "model.layers.%d.self_attn." % LAYER
q_a_fp8, q_a_sc = load(p + "q_a_proj.weight"), load(p + "q_a_proj.weight_scale")
q_b_fp8, q_b_sc = load(p + "q_b_proj.weight"), load(p + "q_b_proj.weight_scale")
kv_fp8, kv_sc = load(p + "kv_a_proj_with_mqa.weight"), load(p + "kv_a_proj_with_mqa.weight_scale")
gamma_cq = load(p + "q_a_layernorm.weight").to(torch.bfloat16)
gamma_ckv = load(p + "kv_a_layernorm.weight").to(torch.bfloat16)
kv_b = load(p + "kv_b_proj.weight").to(torch.bfloat16)
w_kc = kv_b.view(N, D + 256, Hckv)[:, :D, :].contiguous()

inv = lambda sc: sc.reshape(sc.shape[0], sc.shape[1] // 2, 2).transpose(0, 1).contiguous()

# state after the MXFP8 scheme's process_weights_after_loading: weights [in, out]
fused_in_out = torch.cat([q_a_fp8, kv_fp8], dim=0).t().contiguous().to(dev)     # [He, 1344]
fused_scale_inv = inv(torch.cat([q_a_sc, kv_sc], dim=0)).to(dev)               # [He/64, 1344, 2]
q_b_in_out = q_b_fp8.t().contiguous().to(dev)                                  # [Hcq, 5120]
q_b_scale_inv = inv(q_b_sc).to(dev)                                            # [Hcq/64, 5120, 2]

module = types.SimpleNamespace()
module.q_lora_rank = Hcq
module.qkv_a_proj = types.SimpleNamespace(weight=fused_in_out, weight_scale_inv=fused_scale_inv)
module.q_b_proj = types.SimpleNamespace(weight=q_b_in_out, weight_scale_inv=q_b_scale_inv)
module.q_a_proj_weight = None
module.kv_a_proj_weight = None

prepare_mlapolog_bf16_weights(module)
print("q_a:", tuple(module.q_a_proj_weight.shape), module.q_a_proj_weight.dtype,
      "nz=", torch_npu.get_npu_format(module.q_a_proj_weight) == 29)
print("kv_a:", tuple(module.kv_a_proj_weight.shape), module.kv_a_proj_weight.dtype)
print("q_b:", tuple(module.q_b_proj.weight.shape), module.q_b_proj.weight.dtype,
      "nz=", torch_npu.get_npu_format(module.q_b_proj.weight) == 29)

print("alias registered:", register_custom_alias())
x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T)
freq = torch.outer(pos.float(), 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr)))
cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

got = torch.ops.custom.npu_mla_prolog_v3(
    token_x=x, weight_dq=module.q_a_proj_weight, weight_uq_qr=module.q_b_proj.weight,
    weight_uk=w_kc.to(dev), weight_dkv_kr=module.kv_a_proj_weight,
    rmsnorm_gamma_cq=gamma_cq.to(dev), rmsnorm_gamma_ckv=gamma_ckv.to(dev),
    rope_sin=sin, rope_cos=cos, kv_cache=kvc, kr_cache=krc, cache_index=idx,
    rmsnorm_epsilon_cq=EPS, rmsnorm_epsilon_ckv=EPS,
    cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=1)
torch_npu.npu.synchronize()

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


def cmp(name, a, b):
    a, b = a.cpu().float(), b.cpu().float()
    cs = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0).item()
    print("%-11s cosine=%.7f  max|d|=%.5f" % (name, cs, (a - b).abs().max().item()))


cmp("query", got[0], q_gold)
cmp("query_rope", got[1], qr_gold)
cmp("query_norm", got[3], cq)
cmp("kv_cache", kvc.cpu().float().view(-1, Hckv)[idx.cpu()], ckv)
cmp("kr_cache", krc.cpu().float().view(-1, Dr)[idx.cpu()], kr)
