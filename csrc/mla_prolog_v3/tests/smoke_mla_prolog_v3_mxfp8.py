import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 4
torch.manual_seed(0)


def e8m0(n):
    # e8m0 stored as uint8 with the exponent bias of 127 (value = 2^(e-127))
    return torch.full(n, 127, dtype=torch.uint8, device=dev)


x = (torch.randn(T, He, device=dev) * 0.1).to(torch.float8_e4m3fn)
Wdq = (torch.randn(He, Hcq, device=dev) * 0.02).to(torch.float8_e4m3fn)
Wuqqr = (torch.randn(Hcq, N * (D + Dr), device=dev) * 0.02).to(torch.float8_e4m3fn)
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
Wdkv = (torch.randn(He, Hckv + Dr, device=dev) * 0.02).to(torch.float8_e4m3fn)
gq = torch.ones(Hcq, dtype=torch.bfloat16, device=dev)
gkv = torch.ones(Hckv, dtype=torch.bfloat16, device=dev)
sin = torch.zeros(T, Dr, dtype=torch.bfloat16, device=dev)
cos = torch.ones(T, Dr, dtype=torch.bfloat16, device=dev)
kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

sx = e8m0((T, He // 32))
swdq = e8m0((Hcq, He // 32))
swuqqr = e8m0((N * (D + Dr), Hcq // 32))
swdkv = e8m0((Hckv + Dr, He // 32))

print("calling mxfp8 op ...", flush=True)
q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(
    x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
    cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
    weight_quant_mode=3, kv_cache_quant_mode=0,
    dequant_scale_x=sx, dequant_scale_w_dq=swdq, dequant_scale_w_uq_qr=swuqqr,
    dequant_scale_w_dkv_kr=swdkv)
torch_npu.npu.synchronize()
print("ok")
print("q      ", tuple(q.shape), q.dtype, float(q.float().abs().mean()))
print("q_rope ", tuple(qr.shape), qr.dtype, float(qr.float().abs().mean()))
print("q_norm ", tuple(qn.shape), qn.dtype)
print("dq_norm", tuple(dqnorm.shape), dqnorm.dtype)
print("kv_cache nonzero:", int((kvc != 0).sum()), "abs sum", float(kvc.float().abs().sum()))
print("kr_cache nonzero:", int((krc != 0).sum()), "abs sum", float(krc.float().abs().sum()))
