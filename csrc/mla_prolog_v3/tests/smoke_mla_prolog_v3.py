import os
import sys

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 4
torch.manual_seed(0)

x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
Wdq = torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02
Wuqqr = torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
Wdkv = torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02
gq = torch.ones(Hcq, dtype=torch.bfloat16, device=dev)
gkv = torch.ones(Hckv, dtype=torch.bfloat16, device=dev)
sin = torch.zeros(T, Dr, dtype=torch.bfloat16, device=dev)
cos = torch.ones(T, Dr, dtype=torch.bfloat16, device=dev)
kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

# weights must be FRACTAL_NZ for the kernel matmuls
Wdq_nd, Wuqqr_nd, Wdkv_nd = Wdq.clone(), Wuqqr.clone(), Wdkv.clone()
Wdq = torch_npu.npu_format_cast(Wdq, 29)
Wuqqr = torch_npu.npu_format_cast(Wuqqr, 29)
Wdkv = torch_npu.npu_format_cast(Wdkv, 29)

print("calling op ...", flush=True)
q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(
    x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
    cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
    weight_quant_mode=0, kv_cache_quant_mode=0)
torch_npu.npu.synchronize()
print("ok")
print("q      ", tuple(q.shape), q.dtype, float(q.float().abs().mean()))
print("q_rope ", tuple(qr.shape), qr.dtype, float(qr.float().abs().mean()))
print("dq_nope", tuple(dqn.shape), dqn.dtype)
print("q_norm ", tuple(qn.shape), qn.dtype, float(qn.float().abs().mean()))
print("dq_norm", tuple(dqnorm.shape), dqnorm.dtype)
print("kv_cache nonzero:", int((kvc != 0).sum()), "sum", float(kvc.float().abs().sum()))
print("kr_cache nonzero:", int((krc != 0).sum()), "sum", float(krc.float().abs().sum()))

torch.save(
    {"x": x.cpu(), "Wdq": Wdq_nd.cpu(), "Wuqqr": Wuqqr_nd.cpu(), "Wuk": Wuk.cpu(), "Wdkv": Wdkv_nd.cpu(),
     "gq": gq.cpu(), "gkv": gkv.cpu(), "sin": sin.cpu(), "cos": cos.cpu(), "idx": idx.cpu(),
     "q": q.cpu(), "qr": qr.cpu(), "qn": qn.cpu(), "kvc": kvc.cpu(), "krc": krc.cpu()},
    "/tmp/mla_prolog_smoke.pt")
print("saved /tmp/mla_prolog_smoke.pt")
