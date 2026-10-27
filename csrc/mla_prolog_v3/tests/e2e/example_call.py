"""Minimal, self-contained example of calling the ported kernel from Python.

Shows all three entry points and asserts they agree, so it doubles as a smoke test:

    python3 csrc/mla_prolog_v3/tests/e2e/example_call.py
"""
import sys

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib"
sys.path.insert(0, LIB)
sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")

torch.ops.load_library(LIB + "/libsgl_kernel_npu.so")
import libsgl_kernel_npu as _C  # noqa: E402
from sgl_kernel_npu.mla_prolog_v3 import mla_prolog_v3 as api  # noqa: E402

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 4
torch.manual_seed(0)

# ---- weights (bf16, FRACTAL_NZ for the three projections, ND for w_kc)
nz = lambda t: torch_npu.npu_format_cast(t, 29)  # noqa: E731
w_dq = nz(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02)
w_uq_qr = nz(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02)
w_dkv_kr = nz(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02)
w_uk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
w_kc = w_uk  # in GLM the "weight_uk" is the absorbed kv_b slice

# ---- inputs
x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T)
freq = torch.outer(pos.float(), 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr)))
cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
kv_cache = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
kr_cache = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
slot = torch.arange(T, dtype=torch.int64, device=dev)

kw = dict(cache_index=slot, cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=0)

# ---- (1) high-level API (allocates, handles dtype/format adaptation)
q1, qr1, dqn1, qn1, dqn2 = api(x, w_dq, w_uq_qr, w_kc, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos,
                              kv_cache, kr_cache, **kw)

# ---- (2) torch.ops entry with caller-provided buffers
q = torch.empty(T, N, Hckv, dtype=torch.bfloat16, device=dev)
qr = torch.empty(T, N, Dr, dtype=torch.bfloat16, device=dev)
dqn = torch.empty(0, dtype=torch.float32, device=dev)
qn = torch.empty(T, Hcq, dtype=torch.bfloat16, device=dev)
dqnorm = torch.empty(0, dtype=torch.float32, device=dev)
torch.ops.npu.mla_prolog_v3_out(x, w_dq, w_uq_qr, w_kc, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos,
                                kv_cache, kr_cache, q, qr, dqn, qn, dqnorm, **kw)

# ---- (3) PyBind direct call (fastest: skips the torch dispatcher)
q3 = torch.empty_like(q)
qr3 = torch.empty_like(qr)
qn3 = torch.empty_like(qn)
_C.mla_prolog_v3_out(x, w_dq, w_uq_qr, w_kc, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos,
                     kv_cache, kr_cache, q3, qr3, torch.empty(0, dtype=torch.float32, device=dev),
                     qn3, torch.empty(0, dtype=torch.float32, device=dev), **kw)
torch_npu.npu.synchronize()

same = (torch.equal(q1, q) and torch.equal(qr1, qr) and torch.equal(qn1, qn)
        and torch.equal(q, q3) and torch.equal(qr, qr3) and torch.equal(qn, qn3))
print("shapes      :", tuple(q1.shape), tuple(qr1.shape), tuple(qn1.shape))
print("three paths bit-identical:", same)
sys.exit(0 if same else 1)
