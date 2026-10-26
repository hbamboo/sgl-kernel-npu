"""Probe the exact RoPE element mapping.

gamma_cq = e_k makes c_Q a scaled one-hot vector, so q_rope[head 0] is exactly
rope(scalar * e_k).  Running a couple of k values (and a ramp cos/sin) reveals
which cos/sin elements are used for each output position.
"""
import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 1
torch.manual_seed(11)

x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
Wdq = torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02
Wuqqr = torch.zeros(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev)
Wuqqr[:Dr, D : D + Dr] = torch.eye(Dr, dtype=torch.bfloat16, device=dev)  # head 0 rope = identity
Wuk = torch.zeros(N, D, Hckv, dtype=torch.bfloat16, device=dev)
Wdkv = torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02
gkv = torch.ones(Hckv, dtype=torch.bfloat16, device=dev)

# ramp cos/sin so the index used for each element can be identified
cos = torch.tensor([[1.0 + 0.01 * i for i in range(Dr)]], dtype=torch.bfloat16)
sin = torch.tensor([[0.5 + 0.01 * i for i in range(Dr)]], dtype=torch.bfloat16)
cos = cos.to(dev)
sin = sin.to(dev)

kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.tensor([0], dtype=torch.int64, device=dev)

Wdq_nz = torch_npu.npu_format_cast(Wdq.clone(), 29)
Wuqqr_nz = torch_npu.npu_format_cast(Wuqqr.clone(), 29)
Wdkv_nz = torch_npu.npu_format_cast(Wdkv.clone(), 29)

cq_full = (x.float() @ Wdq.float())
cq_full = cq_full / torch.sqrt(cq_full.pow(2).mean(-1, keepdim=True) + 1e-5)

for k in (0, 1, 32, 33):
    gq = torch.zeros(Hcq, dtype=torch.bfloat16, device=dev)
    gq[k] = 1.0
    q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(
        x, Wdq_nz, Wuqqr_nz, Wuk, Wdkv_nz, gq, gkv, sin, cos, kvc, krc,
        cache_index=idx, cache_mode="PA_BSND", query_norm_flag=False,
        weight_quant_mode=0, kv_cache_quant_mode=0)
    torch_npu.npu.synchronize()
    out = qr[0, 0, :].float()
    xin = float(cq_full[0, k])
    nz = [(j, round(float(out[j]), 5)) for j in range(Dr) if abs(float(out[j])) > 1e-6]
    print("k=%2d  xin=%.5f  nonzero out: %s" % (k, xin, nz))
    # candidate: what does the value look like relative to cos/sin at that index?
    for j, v in nz:
        print("      out[%d]=%.5f -> v/1 = %.3f ; cos[%d]=%.3f sin[%d]=%.3f cos[32+%d]=%s"
              % (j, v, v, j, 1.0 + 0.01 * j, j, 0.5 + 0.01 * j, j, 1.0 + 0.01 * (32 + j) if 32 + j < 64 else "-"))
