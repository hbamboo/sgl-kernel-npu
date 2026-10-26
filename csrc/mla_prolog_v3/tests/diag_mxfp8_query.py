"""Localise the MXFP8 'inf' in the query branch.

Reconstructs c_Q from the kernel outputs (query_norm fp8 + dequant_scale_q_norm
e8m0) and compares it against a CPU reference.  If c_Q matches but q is inf, the
problem is the W_UQ_QR dequant path, not the first GEMM.
"""
import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 4
torch.manual_seed(0)


def e8m0(n):
    return torch.full(n, 127, dtype=torch.uint8, device=dev)


x_bf16 = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
Wdq_bf = torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02
x = x_bf16.to(torch.float8_e4m3fn)
Wdq = Wdq_bf.to(torch.float8_e4m3fn)
Wuqqr = (torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02).to(torch.float8_e4m3fn)
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
Wdkv = (torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02).to(torch.float8_e4m3fn)
gq = torch.ones(Hcq, dtype=torch.bfloat16, device=dev)
gkv = torch.ones(Hckv, dtype=torch.bfloat16, device=dev)
sin = torch.zeros(T, Dr, dtype=torch.bfloat16, device=dev)
cos = torch.ones(T, Dr, dtype=torch.bfloat16, device=dev)
kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

Wdq_nz = torch_npu.npu_format_cast(Wdq.clone(), 29)
Wuqqr_nz = torch_npu.npu_format_cast(Wuqqr.clone(), 29)
Wdkv_nz = torch_npu.npu_format_cast(Wdkv.clone(), 29)

import sys
mode = sys.argv[1] if len(sys.argv) > 1 else "e8m0"
nz_scales = mode == "nz"
if nz_scales:
    mode = "e8m0"
use_float_scales = mode in ("float", "mixed")
def sc(n):
    if use_float_scales:
        return torch.ones(n, dtype=torch.float32, device=dev)
    return e8m0(n)

_sx = e8m0((T, He // 32)); _sdq = e8m0((Hcq, He // 32)); _suq = e8m0((N * (D + Dr), Hcq // 32)); _sdkv = e8m0((Hckv + Dr, He // 32))
if nz_scales:
    _sx = torch_npu.npu_format_cast(_sx, 29); _sdq = torch_npu.npu_format_cast(_sdq, 29)
    _suq = torch_npu.npu_format_cast(_suq, 29); _sdkv = torch_npu.npu_format_cast(_sdkv, 29)

q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(
    x, Wdq_nz, Wuqqr_nz, Wuk, Wdkv_nz, gq, gkv, sin, cos, kvc, krc,
    cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
    weight_quant_mode=3, kv_cache_quant_mode=0,
    dequant_scale_x=_sx, dequant_scale_w_dq=_sdq,
    dequant_scale_w_uq_qr=_suq, dequant_scale_w_dkv_kr=_sdkv)
torch_npu.npu.synchronize()

qn_f = qn.float()                      # (T, Hcq) fp8 -> f32
scale = torch.pow(2.0, dqnorm.float() - 127.0) if not use_float_scales else torch.ones_like(dqnorm.float())   # (T, Hcq/32)
scale = scale.repeat_interleave(32, dim=1)       # (T, Hcq)
cq_kernel = qn_f * scale

xr = x_bf16.float()
cq_ref = (xr @ Wdq_bf.float())
cq_ref = cq_ref / torch.sqrt(cq_ref.pow(2).mean(-1, keepdim=True) + 1e-5) * 1.0

print("c_Q kernel mean|.| =", cq_kernel.abs().mean().item())
print("c_Q ref    mean|.| =", cq_ref.abs().mean().item())
print("cosine =", torch.nn.functional.cosine_similarity(cq_kernel.flatten(), cq_ref.flatten(), 0).item())
print("dq_norm e8m0 values (first row, first 6):", dqnorm[0, :6].tolist())
print("q dtype/value sample:", q[0, 0, :4].tolist())
print("q finite ratio:", float(torch.isfinite(q.float()).float().mean()))
print("max |q|:", float(q.float().abs().max()))
