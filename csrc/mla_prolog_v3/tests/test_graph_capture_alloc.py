"""Is the allocating entry point safe inside an ACL graph capture?

SGLang captures decode graphs; if a caller used ``torch.ops.npu.mla_prolog_v3`` (which
allocates its five outputs) inside the capture, the allocations come from the graph's
private pool.  This checks that capture + replay works and stays numerically identical to
eager execution, so the constraint can be documented either way.
"""
import sys

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 1
torch.manual_seed(0)

nz = lambda t: torch_npu.npu_format_cast(t, 29)  # noqa: E731
w_dq = nz(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02)
w_uq_qr = nz(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02)
w_dkv_kr = nz(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02)
w_uk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
g_cq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
g_ckv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T)
freq = torch.outer(pos.float(), 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr)))
cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
kvc = torch.zeros(2, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(2, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)
kw = dict(cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
          weight_quant_mode=0, kv_cache_quant_mode=0)

ref = torch.ops.npu.mla_prolog_v3(x, w_dq, w_uq_qr, w_uk, w_dkv_kr, g_cq, g_ckv, sin, cos,
                                  kvc, krc, **kw)
torch_npu.npu.synchronize()

graph = torch.npu.NPUGraph()
try:
    with torch.npu.graph(graph):
        out = torch.ops.npu.mla_prolog_v3(x, w_dq, w_uq_qr, w_uk, w_dkv_kr, g_cq, g_ckv, sin, cos,
                                          kvc, krc, **kw)
    torch_npu.npu.synchronize()
    graph.replay()
    torch_npu.npu.synchronize()
    same = all(torch.equal(a, b) for a, b in zip(out[:2] + out[3:4], ref[:2] + ref[3:4]))
    print("capture OK, replay bit-identical to eager:", same)
    print("RESULT:", "OK" if same else "MISMATCH")
except Exception as exc:  # noqa: BLE001
    print("capture FAILED:", str(exc).splitlines()[0][:200])
    print("RESULT: NOT-CAPTURABLE")
sys.exit(0)
