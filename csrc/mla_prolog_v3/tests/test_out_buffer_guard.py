"""A wrong-sized output buffer must raise a clear error, never corrupt device memory.

The kernel writes the five outputs without bounds checks, so `mla_prolog_v3_out` validates
their element counts.  This checks the happy path plus every wrong-size case.
"""
import sys

import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 4
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


def make_exact():
    return [
        torch.empty(T, N, Hckv, dtype=torch.bfloat16, device=dev),
        torch.empty(T, N, Dr, dtype=torch.bfloat16, device=dev),
        torch.empty(0, dtype=torch.float32, device=dev),
        torch.empty(T, Hcq, dtype=torch.bfloat16, device=dev),
        torch.empty(0, dtype=torch.float32, device=dev),
    ]


def make_bad(which, short=8):
    bufs = make_exact()
    out = []
    for i, buf in enumerate(bufs):
        if i == which:
            buf = torch.empty(max(buf.numel() - short, 0), dtype=buf.dtype, device=dev)
        out.append(buf)
    return out


def call(bufs):
    torch.ops.npu.mla_prolog_v3_out(x, w_dq, w_uq_qr, w_uk, w_dkv_kr, g_cq, g_ckv, sin, cos,
                                    kvc, krc, *bufs, **kw)


fails = []
# happy path
try:
    call(make_exact())
    torch_npu.npu.synchronize()
    print("%-46s OK" % "correct sizes")
except Exception as exc:  # noqa: BLE001
    fails.append("happy path failed: %s" % exc)
    print("%-46s FAIL" % "correct sizes")

cases = {
    "query too small": 0,
    "query_rope too small": 1,
    "query_norm too small": 3,
}

for name, which in cases.items():
    bufs = make_bad(which)
    try:
        call(bufs)
        torch_npu.npu.synchronize()
        print("%-46s NO ERROR (bad)" % name)
        fails.append(name)
    except RuntimeError as exc:
        msg = str(exc).splitlines()[0]
        ok = "expected" in msg
        print("%-46s raised: %s" % (name, msg[:70]))
        if not ok:
            fails.append(name)

print()
print("RESULT:", "OK" if not fails else "FAILED: %s" % fails)
sys.exit(0 if not fails else 1)
