"""SGLang leaves q_b_proj / w_kc in ND; the shim must still produce correct results.

Simulates the real serving path: qkv_a NZ (as SGLang's mlaprolog_preprocess_weight
does), q_b passed as ND, w_kc as ND, and SGLang's hardcoded weight_quant_mode=1.
"""
import sys

import torch
import torch_npu

sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")
import sgl_kernel_npu  # noqa: F401,E402
from sgl_kernel_npu.mla_prolog_v3_compat import (  # noqa: E402
    clear_format_cache,
    register_custom_alias,
)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 8
torch.manual_seed(0)

print("alias registered:", register_custom_alias())

gq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
gkv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
Wdq_nd = torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02
Wuqqr_nd = torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02
Wdkv_nd = torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T)
inv = 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr))
freq = torch.outer(pos.float(), inv)
cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)

nz = lambda t: torch_npu.npu_format_cast(t.clone(), 29)

# reference: everything NZ, direct call
ref = torch.ops.npu.mla_prolog_v3(
    x, nz(Wdq_nd), nz(Wuqqr_nd), Wuk, nz(Wdkv_nd), gq, gkv, sin, cos,
    torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev),
    torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev),
    cache_index=torch.arange(T, dtype=torch.int64, device=dev),
    cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=0, kv_cache_quant_mode=0)
torch_npu.npu.synchronize()

# serving-like: qkv_a NZ, q_b ND, w_kc ND, mode=1
kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
got = torch.ops.custom.npu_mla_prolog_v3(
    token_x=x, weight_dq=nz(Wdq_nd), weight_uq_qr=Wuqqr_nd, weight_uk=Wuk,
    weight_dkv_kr=nz(Wdkv_nd), rmsnorm_gamma_cq=gq, rmsnorm_gamma_ckv=gkv,
    rope_sin=sin, rope_cos=cos, kv_cache=kvc, kr_cache=krc,
    cache_index=torch.arange(T, dtype=torch.int64, device=dev),
    dequant_scale_w_uq_qr=torch.ones(N * (D + Dr), dtype=torch.float32, device=dev),
    rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5,
    cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=1)
torch_npu.npu.synchronize()

names = ["query", "query_rope", "dequant", "query_norm"]
for name, a, b in zip(names, got[:4], ref[:4]):
    if a.numel() == 0:
        print("%-11s empty" % name)
        continue
    cs = torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()
    print("%-11s bit-identical=%-5s cosine=%.7f" % (name, torch.equal(a, b), cs))

# second call must reuse the cached NZ tensors
t0 = torch.npu.Event(enable_timing=True)
got2 = torch.ops.custom.npu_mla_prolog_v3(
    token_x=x, weight_dq=nz(Wdq_nd), weight_uq_qr=Wuqqr_nd, weight_uk=Wuk,
    weight_dkv_kr=nz(Wdkv_nd), rmsnorm_gamma_cq=gq, rmsnorm_gamma_ckv=gkv,
    rope_sin=sin, rope_cos=cos, kv_cache=kvc, kr_cache=krc,
    cache_index=torch.arange(T, dtype=torch.int64, device=dev),
    cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=1)
torch_npu.npu.synchronize()
print("cached second call identical:", torch.equal(got2[0], got[0]))
clear_format_cache()
