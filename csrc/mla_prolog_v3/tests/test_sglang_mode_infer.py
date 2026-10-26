"""The shim must accept SGLang's hardcoded weight_quant_mode=1 with bf16 weights."""
import sys

import torch
import torch_npu

sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")
import sgl_kernel_npu  # noqa: F401,E402
from sgl_kernel_npu.mla_prolog_v3_compat import (  # noqa: E402
    infer_weight_quant_mode,
    register_custom_alias,
)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 8
torch.manual_seed(0)

# mode detection table
cases = [
    ("bf16 + mode 1 (SGLang default)", torch.bfloat16, None, 1, 0),
    ("bf16 + mode 0", torch.bfloat16, None, 0, 0),
    ("fp8 + e8m0 scale", torch.float8_e4m3fn, "scale", 1, 3),
    ("int8", torch.int8, None, 1, None),
]
for name, dtype, scale, given, want in cases:
    w = torch.empty(1, dtype=dtype)
    sc = torch.empty(1, dtype=torch.uint8) if scale else None
    got = infer_weight_quant_mode(w, sc)
    print("%-32s dtype=%-16s given=%s -> infer=%s %s"
          % (name, str(dtype).replace("torch.", ""), given, got, "OK" if got == want else "MISMATCH"))

print("alias registered:", register_custom_alias())

gq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
gkv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
Wdq = torch_npu.npu_format_cast(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wuqqr = torch_npu.npu_format_cast(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wdkv = torch_npu.npu_format_cast(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T)
inv = 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr))
freq = torch.outer(pos.float(), inv)
cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

# SGLang's kwargs verbatim, including weight_quant_mode=1 and a scale tensor
sglang_kwargs = dict(
    token_x=x, weight_dq=Wdq, weight_uq_qr=Wuqqr, weight_uk=Wuk, weight_dkv_kr=Wdkv,
    rmsnorm_gamma_cq=gq, rmsnorm_gamma_ckv=gkv, rope_sin=sin, rope_cos=cos,
    kv_cache=kvc, kr_cache=krc, cache_index=idx,
    dequant_scale_w_uq_qr=torch.ones(N * (D + Dr), dtype=torch.float32, device=dev),
    rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5,
    cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=1,
)
got = torch.ops.custom.npu_mla_prolog_v3(**sglang_kwargs)
torch_npu.npu.synchronize()

ref = torch.ops.npu.mla_prolog_v3(
    x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
    cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
    weight_quant_mode=0, kv_cache_quant_mode=0)
torch_npu.npu.synchronize()
print("SGLang mode=1 kwargs accepted and matches mode=0:",
      all(torch.equal(a, b) for a, b in zip(got[:4], ref[:4])))
